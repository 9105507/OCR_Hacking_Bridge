"""Streamlit frontend for the DARDE/DARDO document validation pipeline."""

from __future__ import annotations

import csv as csv_mod
import io
import json
import logging

import streamlit as st

from pipeline import config
from pipeline.ingestion import load_from_bytes
from pipeline.llm_client import OllamaConnectionError, OllamaVisionClient
from pipeline.models import DocumentInput, PipelineResult, ValidationStatus
from pipeline.orchestrator import process_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DARDE/DARDO Validator — ACH",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATUS_COLORS: dict[str, str] = {
    ValidationStatus.VALIDO.value: "🟢",
    ValidationStatus.NO_VALIDO.value: "🔴",
    ValidationStatus.REQUIERE_REVISION.value: "🟡",
}


def _get_client() -> OllamaVisionClient:
    model = st.session_state.get("model", config.OLLAMA_MODEL)
    return OllamaVisionClient(model=model)


def _results_to_dataframe(results: list[PipelineResult]):
    import pandas as pd

    rows = [r.model_dump() for r in results]
    df = pd.DataFrame(rows)
    df.rename(
        columns={
            "id_documento": "ID Documento",
            "fecha_inscripcion": "Fecha Inscripción",
            "fecha_renovacion": "Fecha Renovación",
            "estado_validacion": "Estado",
            "motivo": "Motivo",
            "nivel_confianza": "Confianza",
        },
        inplace=True,
    )
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuración")

    model_name = st.text_input(
        "Modelo Ollama",
        value=config.OLLAMA_MODEL,
        help="Nombre del modelo vision en Ollama (ej. llama3.2-vision, qwen2.5-vl)",
    )
    st.session_state["model"] = model_name

    st.divider()
    st.markdown(
        f"**Fecha inicio programa:** `{config.PROGRAM_START_DATE.isoformat()}`"
    )
    st.markdown(
        f"**Umbral clasificación:** `{config.CLASSIFICATION_CONFIDENCE_THRESHOLD}`"
    )
    st.markdown(
        f"**Umbral extracción:** `{config.EXTRACTION_CONFIDENCE_THRESHOLD}`"
    )

    st.divider()
    st.caption("DARDE/DARDO Validator · ACH Hackathon")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📄 Validador de Documentos DARDE / DARDO")
st.markdown(
    "Sube uno o varios documentos de desempleo (DARDE o DARDO) en formato **PDF** o **JPG/PNG**. "
    "El sistema extraerá las fechas clave y comprobará si cumplen la regla de elegibilidad del programa."
)

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Arrastra o selecciona documentos DARDE/DARDO",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
    help="Formatos aceptados: JPEG, PNG o PDF (se procesará cada página del PDF por separado).",
)

if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} archivo(s) seleccionado(s)")

    # Show previews for images (PDFs can't be previewed as thumbnails easily)
    image_files = [f for f in uploaded_files if f.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    if image_files:
        cols = st.columns(min(len(image_files), 5))
        for idx, uf in enumerate(image_files):
            with cols[idx % len(cols)]:
                st.image(uf, caption=uf.name, use_container_width=True)

    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
    if pdf_files:
        st.markdown(f"📑 {len(pdf_files)} PDF(s): " + ", ".join(f"`{f.name}`" for f in pdf_files))

    # --- Process button ---
    if st.button("🚀 Procesar documentos", type="primary"):
        # 1. Ingest all files into DocumentInput list
        all_docs: list[DocumentInput] = []
        ingest_errors: list[str] = []

        for uf in uploaded_files:
            raw = uf.read()
            uf.seek(0)
            try:
                docs = load_from_bytes(raw, uf.name)
                all_docs.extend(docs)
            except ValueError as exc:
                ingest_errors.append(f"⚠️ {uf.name}: {exc}")

        if ingest_errors:
            for err in ingest_errors:
                st.warning(err)

        if not all_docs:
            st.error("No se pudo procesar ningún documento válido.")
            st.stop()

        st.info(f"🔍 {len(all_docs)} página(s)/imagen(es) a procesar…")

        # 2. Run pipeline
        results: list[PipelineResult] = []
        progress = st.progress(0, text="Iniciando…")

        try:
            client = _get_client()
        except Exception as exc:
            st.error(f"No se pudo conectar a Ollama: {exc}")
            st.stop()

        for i, doc in enumerate(all_docs):
            progress.progress(
                i / len(all_docs),
                text=f"Procesando {doc.id} ({i + 1}/{len(all_docs)})…",
            )
            try:
                result = process_document(client, doc)
            except OllamaConnectionError:
                st.error(
                    "❌ No se puede conectar a Ollama. "
                    "¿Está ejecutándose? (`ollama serve`)"
                )
                client.close()
                st.stop()
            except Exception as exc:
                result = PipelineResult(
                    id_documento=doc.id,
                    estado_validacion=ValidationStatus.REQUIERE_REVISION.value,
                    motivo=f"Error: {exc}",
                    nivel_confianza=0.0,
                )
            results.append(result)

        progress.progress(1.0, text="✅ Completado")
        client.close()

        st.session_state["last_results"] = results

# ---------------------------------------------------------------------------
# Show results
# ---------------------------------------------------------------------------

if "last_results" in st.session_state and st.session_state["last_results"]:
    results = st.session_state["last_results"]
    st.divider()
    st.subheader("📊 Resultados")

    # Summary metrics
    n_valid = sum(1 for r in results if r.estado_validacion == ValidationStatus.VALIDO.value)
    n_invalid = sum(1 for r in results if r.estado_validacion == ValidationStatus.NO_VALIDO.value)
    n_review = sum(1 for r in results if r.estado_validacion == ValidationStatus.REQUIERE_REVISION.value)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(results))
    col2.metric("🟢 Válido", n_valid)
    col3.metric("🔴 No Válido", n_invalid)
    col4.metric("🟡 Revisión", n_review)

    # Table
    df = _results_to_dataframe(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-document detail
    for r in results:
        icon = STATUS_COLORS.get(r.estado_validacion, "⚪")
        with st.expander(f"{icon} {r.id_documento} — {r.estado_validacion}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Fecha Inscripción:** `{r.fecha_inscripcion or '—'}`")
            c2.markdown(f"**Fecha Renovación:** `{r.fecha_renovacion or '—'}`")
            st.markdown(f"**Motivo:** {r.motivo}")
            st.markdown(f"**Confianza:** {r.nivel_confianza:.0%}")

    # Download buttons
    st.divider()
    col_csv, col_json = st.columns(2)
    with col_csv:
        buf = io.StringIO()
        writer = csv_mod.DictWriter(
            buf,
            fieldnames=["id_documento", "fecha_inscripcion", "fecha_renovacion",
                         "estado_validacion", "motivo", "nivel_confianza"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.model_dump())
        st.download_button(
            "⬇️ Descargar CSV",
            data=buf.getvalue(),
            file_name="resultados.csv",
            mime="text/csv",
        )
    with col_json:
        json_str = json.dumps(
            [r.model_dump() for r in results], indent=2, ensure_ascii=False,
        )
        st.download_button(
            "⬇️ Descargar JSON",
            data=json_str,
            file_name="resultados.json",
            mime="application/json",
        )
