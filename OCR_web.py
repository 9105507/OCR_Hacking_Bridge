# uv run streamlit run OCR_web.py
"""Streamlit frontend para validar documentos DARDE/DARDO con OCR local."""

from __future__ import annotations

import csv as csv_mod
import io
import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st

# Para PDF -> imagen
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PROGRAM_START_DATE = datetime.strptime("01/03/2025", "%d/%m/%Y")
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(
    page_title="Validador DARDE/DARDO OCR",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_LOGO_PATH = "assets/logo.svg"

st.markdown(
    """
    <style>
    .stApp > header {
        background-color: var(--primary-color, #E63946);
    }
    .thumb-preview img {
        max-height: 150px;
        object-fit: contain;
        border-radius: 6px;
        border: 1px solid #ddd;
    }
    .stButton > button[kind="primary"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODELOS
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    id_documento: str
    tipo_documento: str
    fecha_inscripcion: Optional[str]
    fecha_renovacion: Optional[str]
    fecha_emision: Optional[str]
    estado_validacion: str
    motivo: str
    nivel_confianza: float
    texto_ocr: str


STATUS_VALIDO = "VALIDO"
STATUS_NO_VALIDO = "NO_VALIDO"
STATUS_REVISION = "REQUIERE_REVISION"

STATUS_COLORS: dict[str, str] = {
    STATUS_VALIDO: "🟢",
    STATUS_NO_VALIDO: "🔴",
    STATUS_REVISION: "🟡",
}

# ---------------------------------------------------------------------------
# FUNCIONES OCR / PROCESADO
# ---------------------------------------------------------------------------

def normalizar_texto(texto: str) -> str:
    """
    Pasa el texto a minúsculas y elimina tildes para facilitar búsquedas.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def extraer_campos(texto: str) -> dict:
    """
    Extrae del texto OCR:
      - fecha de inscripción
      - fecha de renovación (la siguiente a la fecha de inscripción)
      - fecha de emisión (formato: '18 de julio de 2025')

    Devuelve un diccionario con los campos encontrados.
    """
    texto_norm = normalizar_texto(texto)

    patron_fecha_num = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

    patron_fecha_texto = (
        r"\b\d{1,2}\s+de\s+"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s+de\s+\d{4}\b"
    )

    resultado = {
        "fecha_inscripcion": None,
        "fecha_renovacion": None,
        "fecha_emision": None,
    }

    # Buscar desde "inscripcion" y sacar las 2 primeras fechas numéricas
    m_bloque = re.search(
        r"inscrip(?:cion)?(.{0,300})",
        texto_norm,
        re.IGNORECASE | re.DOTALL
    )
    if m_bloque:
        fechas = re.findall(patron_fecha_num, m_bloque.group(1), re.IGNORECASE)
        if len(fechas) >= 1:
            resultado["fecha_inscripcion"] = fechas[0]
        if len(fechas) >= 2:
            resultado["fecha_renovacion"] = fechas[1]

    # Fecha de emisión en formato texto
    m_emi = re.search(
        rf"fecha\s+de\s+emision\D{{0,20}}({patron_fecha_texto})",
        texto_norm,
        re.IGNORECASE
    )
    if m_emi:
        resultado["fecha_emision"] = m_emi.group(1)

    return resultado


def parsear_fecha(fecha_str: Optional[str]):
    """
    Convierte una fecha en formato dd/mm/yyyy o dd-mm-yyyy a datetime.
    Devuelve None si la fecha no es válida o viene vacía.
    """
    if not fecha_str:
        return None

    fecha_str = fecha_str.strip().replace("-", "/")

    for formato in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(fecha_str, formato)
        except ValueError:
            pass

    return None


def evaluar_validez_documento(fecha_inscripcion: Optional[str], fecha_renovacion: Optional[str]) -> dict:
    """
    Evalúa si el documento es válido según la condición:
        inscripción < 01/03/2025 < renovación

    Devuelve un diccionario con:
      - valido: True / False / None
      - motivo: explicación del resultado
    """
    fecha_referencia = PROGRAM_START_DATE

    fi = parsear_fecha(fecha_inscripcion)
    fr = parsear_fecha(fecha_renovacion)

    if fi is None and fr is None:
        return {
            "valido": None,
            "motivo": "No se pudieron interpretar ni la fecha de inscripción ni la de renovación."
        }

    if fi is None:
        return {
            "valido": None,
            "motivo": "No se pudo interpretar la fecha de inscripción."
        }

    if fr is None:
        return {
            "valido": None,
            "motivo": "No se pudo interpretar la fecha de renovación."
        }

    valido = fi < fecha_referencia < fr

    return {
        "valido": valido,
        "motivo": (
            "El documento es válido."
            if valido
            else "El documento no es válido porque no cumple: inscripción < 01/03/2025 < renovación."
        )
    }


def configurar_tesseract(path_tesseract: str):
    """
    Configura la ruta de Tesseract si existe.
    """
    if path_tesseract and os.path.exists(path_tesseract):
        pytesseract.pytesseract.tesseract_cmd = path_tesseract


def preprocess_image(image_bgr: np.ndarray) -> dict:
    """
    Aplica preprocesado básico a una imagen OpenCV BGR.
    Devuelve distintas fases por si quieres mostrarlas en Streamlit.
    """
    gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gris_blur = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(gris_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "original": image_bgr,
        "gris": gris,
        "binaria": binaria,
    }


def aplicar_ocr_desde_array(image_bgr: np.ndarray, lang: str = "spa") -> tuple[str, dict]:
    """
    Recibe una imagen OpenCV (BGR), aplica preprocesado y OCR.
    """
    fases = preprocess_image(image_bgr)
    texto = pytesseract.image_to_string(fases["binaria"], lang=lang)
    return texto, fases


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """
    Convierte un archivo subido (imagen) a np.ndarray BGR para OpenCV.
    """
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    uploaded_file.seek(0)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo decodificar la imagen: {uploaded_file.name}")
    return image


def pdf_to_bgr_pages(pdf_bytes: bytes) -> list[np.ndarray]:
    """
    Convierte un PDF en una lista de páginas como imágenes BGR.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image no está instalado. Instálalo con: pip install pdf2image"
        )

    pil_pages = convert_from_bytes(pdf_bytes)
    pages_bgr = []

    for page in pil_pages:
        page_rgb = np.array(page)
        page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
        pages_bgr.append(page_bgr)

    return pages_bgr


def clasificar_estado(validacion: dict) -> tuple[str, float]:
    """
    Convierte el resultado de validación a estado y confianza aproximada.
    """
    if validacion["valido"] is True:
        return STATUS_VALIDO, 1.0
    if validacion["valido"] is False:
        return STATUS_NO_VALIDO, 1.0
    return STATUS_REVISION, 0.0


def procesar_imagen_documento(doc_id: str, image_bgr: np.ndarray, lang: str = "spa") -> tuple[PipelineResult, dict]:
    """
    Procesa una imagen individual: OCR + extracción + validación.
    """
    texto, fases = aplicar_ocr_desde_array(image_bgr, lang=lang)
    campos = extraer_campos(texto)
    validacion = evaluar_validez_documento(
        campos["fecha_inscripcion"],
        campos["fecha_renovacion"]
    )
    estado, confianza = clasificar_estado(validacion)

    result = PipelineResult(
        id_documento=doc_id,
        tipo_documento="DARDE/DARDO",
        fecha_inscripcion=campos["fecha_inscripcion"],
        fecha_renovacion=campos["fecha_renovacion"],
        fecha_emision=campos["fecha_emision"],
        estado_validacion=estado,
        motivo=validacion["motivo"],
        nivel_confianza=confianza,
        texto_ocr=texto,
    )
    return result, fases


def _results_to_dataframe(results: list[PipelineResult]) -> pd.DataFrame:
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    df.rename(
        columns={
            "id_documento": "ID Documento",
            "tipo_documento": "Tipo",
            "fecha_inscripcion": "Fecha Inscripción",
            "fecha_renovacion": "Fecha Renovación",
            "fecha_emision": "Fecha Emisión",
            "estado_validacion": "Estado",
            "motivo": "Motivo",
            "nivel_confianza": "Confianza",
            "texto_ocr": "Texto OCR",
        },
        inplace=True,
    )
    return df

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    if os.path.isfile(_LOGO_PATH):
        st.image(_LOGO_PATH, use_container_width=True)
    else:
        st.markdown(
            "<p style='text-align:center;color:#888;'>Coloca tu logo en<br><code>assets/logo.svg</code></p>",
            unsafe_allow_html=True
        )

    st.title("⚙️ Configuración")

    tesseract_path = st.text_input(
        "Ruta de Tesseract",
        value=DEFAULT_TESSERACT_PATH,
        help="Ruta al ejecutable tesseract.exe",
    )
    st.session_state["tesseract_path"] = tesseract_path

    ocr_lang = st.text_input(
        "Idioma OCR",
        value="spa",
        help="Ejemplos: spa, eng, spa+eng",
    )
    st.session_state["ocr_lang"] = ocr_lang

    mostrar_preproceso = st.checkbox(
        "Mostrar imágenes preprocesadas",
        value=False,
    )
    st.session_state["mostrar_preproceso"] = mostrar_preproceso

    st.divider()
    st.markdown(
        f"**Fecha referencia:** `{PROGRAM_START_DATE.strftime('%d/%m/%Y')}`"
    )
    st.caption("Regla: inscripción < fecha referencia < renovación")

    st.divider()
    st.caption("Validador DARDE/DARDO · OCR local con Tesseract")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

st.title("📄 Validador de Documentos DARDE / DARDO")
st.markdown(
    "Sube uno o varios documentos en formato **PDF** o **JPG/PNG**. "
    "El sistema aplicará OCR, extraerá las fechas clave y comprobará la validez del documento."
)

uploaded_files = st.file_uploader(
    "Arrastra o selecciona documentos DARDE/DARDO",
    type=["jpg", "jpeg", "png", "pdf", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help="Formatos aceptados: JPEG, PNG, BMP, TIFF o PDF.",
)

if uploaded_files:
    st.info(f"📎 {len(uploaded_files)} archivo(s) seleccionado(s)")

    image_files = [f for f in uploaded_files if f.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
    if image_files:
        cols = st.columns(min(len(image_files), 5))
        for idx, uf in enumerate(image_files):
            with cols[idx % len(cols)]:
                st.markdown('<div class="thumb-preview">', unsafe_allow_html=True)
                st.image(uf, caption=uf.name, width=150)
                st.markdown('</div>', unsafe_allow_html=True)

    pdf_files = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
    if pdf_files:
        st.markdown(f"📑 {len(pdf_files)} PDF(s): " + ", ".join(f"`{f.name}`" for f in pdf_files))

    if st.button("🚀 Procesar documentos", type="primary"):
        configurar_tesseract(st.session_state["tesseract_path"])

        all_items: list[tuple[str, np.ndarray]] = []
        ingest_errors: list[str] = []

        for uf in uploaded_files:
            try:
                if uf.name.lower().endswith(".pdf"):
                    pdf_bytes = uf.read()
                    uf.seek(0)

                    pages = pdf_to_bgr_pages(pdf_bytes)
                    for page_idx, page_bgr in enumerate(pages, start=1):
                        doc_id = f"{uf.name} - página {page_idx}"
                        all_items.append((doc_id, page_bgr))
                else:
                    image_bgr = uploaded_file_to_bgr(uf)
                    all_items.append((uf.name, image_bgr))
            except Exception as exc:
                ingest_errors.append(f"⚠️ {uf.name}: {exc}")

        if ingest_errors:
            for err in ingest_errors:
                st.warning(err)

        if not all_items:
            st.error("No se pudo procesar ningún documento válido.")
            st.stop()

        st.info(f"🔍 {len(all_items)} imagen(es)/página(s) a procesar…")

        results: list[PipelineResult] = []
        debug_images: dict[str, dict] = {}

        progress = st.progress(0, text="Iniciando…")

        for i, (doc_id, image_bgr) in enumerate(all_items):
            progress.progress(
                i / len(all_items),
                text=f"Procesando {doc_id} ({i + 1}/{len(all_items)})…",
            )
            try:
                result, fases = procesar_imagen_documento(
                    doc_id=doc_id,
                    image_bgr=image_bgr,
                    lang=st.session_state["ocr_lang"],
                )
                results.append(result)
                debug_images[doc_id] = fases
            except Exception as exc:
                results.append(
                    PipelineResult(
                        id_documento=doc_id,
                        tipo_documento="DARDE/DARDO",
                        fecha_inscripcion=None,
                        fecha_renovacion=None,
                        fecha_emision=None,
                        estado_validacion=STATUS_REVISION,
                        motivo=f"Error: {exc}",
                        nivel_confianza=0.0,
                        texto_ocr="",
                    )
                )

        progress.progress(1.0, text="✅ Completado")

        st.session_state["last_results"] = results
        st.session_state["debug_images"] = debug_images

# ---------------------------------------------------------------------------
# RESULTADOS
# ---------------------------------------------------------------------------

if "last_results" in st.session_state and st.session_state["last_results"]:
    results = st.session_state["last_results"]
    debug_images = st.session_state.get("debug_images", {})

    st.divider()
    st.subheader("📊 Resultados")

    n_valid = sum(1 for r in results if r.estado_validacion == STATUS_VALIDO)
    n_invalid = sum(1 for r in results if r.estado_validacion == STATUS_NO_VALIDO)
    n_review = sum(1 for r in results if r.estado_validacion == STATUS_REVISION)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(results))
    col2.metric("🟢 Válido", n_valid)
    col3.metric("🔴 No válido", n_invalid)
    col4.metric("🟡 Revisión", n_review)

    df = _results_to_dataframe(results)
    st.dataframe(df, width="stretch", hide_index=True)

    for r in results:
        icon = STATUS_COLORS.get(r.estado_validacion, "⚪")
        with st.expander(f"{icon} {r.id_documento} — {r.estado_validacion}"):
            st.markdown(f"**Tipo documento:** `{r.tipo_documento}`")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Fecha Inscripción:** `{r.fecha_inscripcion or '—'}`")
            c2.markdown(f"**Fecha Renovación:** `{r.fecha_renovacion or '—'}`")
            c3.markdown(f"**Fecha Emisión:** `{r.fecha_emision or '—'}`")

            st.markdown(f"**Motivo:** {r.motivo}")
            st.markdown(f"**Confianza:** {r.nivel_confianza:.0%}")

            tab1, tab2 = st.tabs(["Texto OCR", "Preprocesado"])
            with tab1:
                st.text_area(
                    f"OCR - {r.id_documento}",
                    value=r.texto_ocr,
                    height=250,
                    key=f"ocr_{r.id_documento}",
                )

            with tab2:
                if st.session_state.get("mostrar_preproceso", False):
                    fases = debug_images.get(r.id_documento, {})
                    if fases:
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.markdown("**Original**")
                            st.image(
                                cv2.cvtColor(fases["original"], cv2.COLOR_BGR2RGB),
                                use_container_width=True
                            )

                        with col_b:
                            st.markdown("**Gris**")
                            st.image(
                                fases["gris"],
                                use_container_width=True,
                                clamp=True
                            )

                        with col_c:
                            st.markdown("**Binaria**")
                            st.image(
                                fases["binaria"],
                                use_container_width=True,
                                clamp=True
                            )
                    else:
                        st.info("No hay imágenes de depuración disponibles.")
                else:
                    st.info("Activa 'Mostrar imágenes preprocesadas' en la barra lateral.")

    st.divider()
    col_csv, col_json = st.columns(2)

    with col_csv:
        buf = io.StringIO()
        writer = csv_mod.DictWriter(
            buf,
            fieldnames=[
                "id_documento",
                "tipo_documento",
                "fecha_inscripcion",
                "fecha_renovacion",
                "fecha_emision",
                "estado_validacion",
                "motivo",
                "nivel_confianza",
                "texto_ocr",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

        st.download_button(
            "⬇️ Descargar CSV",
            data=buf.getvalue(),
            file_name="resultados.csv",
            mime="text/csv",
        )

    with col_json:
        json_str = json.dumps(
            [asdict(r) for r in results],
            indent=2,
            ensure_ascii=False,
        )
        st.download_button(
            "⬇️ Descargar JSON",
            data=json_str,
            file_name="resultados.json",
            mime="application/json",
        )