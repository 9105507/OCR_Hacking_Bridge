"""Document ingestion: convert uploaded files (JPG/PNG/PDF) into pipeline-ready inputs."""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

from pipeline.models import DocumentInput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_id(name: str) -> str:
    """Turn a filename into a clean document ID."""
    stem = Path(name).stem
    match = re.match(r"(DARD[EO])\s*(\d+)", stem, re.IGNORECASE)
    if match:
        doc_type = match.group(1).upper()
        number = int(match.group(2))
        return f"{doc_type}_{number:02d}"
    return re.sub(r"\s+", "_", stem)


def _image_bytes_to_base64_jpeg(raw: bytes) -> str:
    """Ensure *raw* image bytes are re-encoded as JPEG and return base64."""
    with Image.open(io.BytesIO(raw)) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Public API: work with in-memory bytes (upload-oriented)
# ---------------------------------------------------------------------------

def load_from_bytes(
    raw: bytes,
    filename: str,
) -> list[DocumentInput]:
    """Convert uploaded file bytes into one or more ``DocumentInput`` objects.

    Supports:
    - **JPEG / PNG**: Returns a single ``DocumentInput``.
    - **PDF**: Renders every page as a JPEG image and returns one
      ``DocumentInput`` per page.

    Raises:
        ValueError: If the file format is unrecognised or corrupt.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return _load_pdf(raw, filename)

    if lower.endswith((".jpg", ".jpeg", ".png")):
        return [_load_image_bytes(raw, filename)]

    raise ValueError(f"Formato no soportado: {filename}")


def _load_image_bytes(raw: bytes, filename: str) -> DocumentInput:
    """Validate and wrap raw image bytes."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception as exc:
        raise ValueError(f"Imagen no válida: {filename}") from exc

    b64 = _image_bytes_to_base64_jpeg(raw)
    doc_id = _sanitise_id(filename)

    logger.info("Loaded image %s -> id=%s", filename, doc_id)
    return DocumentInput(id=doc_id, file_path=Path(filename), base64_image=b64)


def _load_pdf(raw: bytes, filename: str) -> list[DocumentInput]:
    """Render each page of a PDF as a JPEG ``DocumentInput``."""
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"PDF no válido: {filename}") from exc

    docs: list[DocumentInput] = []
    base_id = _sanitise_id(filename)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        # Render at 200 DPI for good VLM readability
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("jpeg")
        b64 = base64.b64encode(img_bytes).decode("ascii")

        page_id = f"{base_id}_p{page_num + 1}" if len(pdf) > 1 else base_id
        docs.append(
            DocumentInput(
                id=page_id,
                file_path=Path(filename),
                base64_image=b64,
            )
        )

    pdf.close()
    logger.info("Loaded PDF %s (%d pages) -> %d documents", filename, len(docs), len(docs))
    return docs
