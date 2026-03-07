"""Pydantic data models for every stage of the pipeline."""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class DocumentInput(BaseModel):
    """Raw document loaded from disk, ready for LLM processing."""

    id: str = Field(description="Unique document identifier derived from filename.")
    file_path: Path
    base64_image: str = Field(repr=False, description="Base64-encoded JPEG bytes.")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Output of the document classifier stage."""

    is_valid_document: bool = False
    is_legible: bool = False
    document_type: Literal["DARDE", "DARDO", "UNKNOWN"] = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    """Dates extracted by the VLM (raw strings, not yet normalised)."""

    fecha_inscripcion_raw: str | None = None
    fecha_renovacion_raw: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_llm_response: str = Field(default="", repr=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    VALIDO = "VÁLIDO"
    NO_VALIDO = "NO VÁLIDO"
    REQUIERE_REVISION = "REQUIERE REVISIÓN HUMANA"


class ValidationResult(BaseModel):
    """Normalised dates + business-rule evaluation."""

    fecha_inscripcion: date | None = None
    fecha_renovacion: date | None = None
    status: ValidationStatus = ValidationStatus.REQUIERE_REVISION
    motivo: str = ""


# ---------------------------------------------------------------------------
# Final pipeline output (one row per document)
# ---------------------------------------------------------------------------

class PipelineResult(BaseModel):
    """Single row in the exported CSV / JSON."""

    id_documento: str
    fecha_inscripcion: str | None = None
    fecha_renovacion: str | None = None
    estado_validacion: str
    motivo: str
    nivel_confianza: float = Field(ge=0.0, le=1.0)
