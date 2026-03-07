"""Core pipeline orchestration — shared by CLI and Streamlit frontend."""

from __future__ import annotations

import logging

from pipeline import config
from pipeline.classifier import classify_document
from pipeline.extractor import extract_dates
from pipeline.llm_client import OllamaVisionClient
from pipeline.models import (
    ClassificationResult,
    DocumentInput,
    ExtractionResult,
    PipelineResult,
    ValidationResult,
    ValidationStatus,
)
from pipeline.validator import validate_document

logger = logging.getLogger(__name__)


def process_document(
    client: OllamaVisionClient,
    doc: DocumentInput,
) -> PipelineResult:
    """Run classify → extract → validate for a single document."""

    # --- 1. Classification (fail-fast) ---
    logger.info("[PIPELINE] ===== Processing %s =====", doc.id)
    classification: ClassificationResult = classify_document(client, doc)
    logger.info("[PIPELINE] Classification result: type=%s, valid=%s, legible=%s, conf=%.2f",
                classification.document_type, classification.is_valid_document,
                classification.is_legible, classification.confidence)

    if not classification.is_valid_document or not classification.is_legible:
        return PipelineResult(
            id_documento=doc.id,
            tipo_documento=classification.document_type,
            estado_validacion=ValidationStatus.REQUIERE_REVISION.value,
            motivo=classification.reason or "Documento no válido o ilegible.",
            nivel_confianza=classification.confidence,
        )

    if classification.confidence < config.CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return PipelineResult(
            id_documento=doc.id,
            tipo_documento=classification.document_type,
            estado_validacion=ValidationStatus.REQUIERE_REVISION.value,
            motivo=(
                f"Confianza de clasificación baja ({classification.confidence:.2f}). "
                "Se requiere revisión manual."
            ),
            nivel_confianza=classification.confidence,
        )

    # --- 2. Date extraction ---
    logger.info("[PIPELINE] Starting date extraction for %s ...", doc.id)
    extraction: ExtractionResult = extract_dates(
        client, doc, doc_type=classification.document_type,
    )
    logger.info("[PIPELINE] Extraction result: insc=%s, reno=%s, conf=%.2f",
                extraction.fecha_inscripcion_raw, extraction.fecha_renovacion_raw,
                extraction.confidence)

    if extraction.confidence < config.EXTRACTION_CONFIDENCE_THRESHOLD:
        return PipelineResult(
            id_documento=doc.id,
            tipo_documento=classification.document_type,
            fecha_inscripcion=extraction.fecha_inscripcion_raw,
            fecha_renovacion=extraction.fecha_renovacion_raw,
            estado_validacion=ValidationStatus.REQUIERE_REVISION.value,
            motivo=(
                f"Confianza de extracción baja ({extraction.confidence:.2f}). "
                "Se requiere revisión manual."
            ),
            nivel_confianza=extraction.confidence,
        )

    # --- 3. Validation ---
    logger.info("[PIPELINE] Starting validation for %s ...", doc.id)
    validation: ValidationResult = validate_document(extraction)
    logger.info("[PIPELINE] Validation result: status=%s, motivo=%s",
                validation.status.value, validation.motivo)

    return PipelineResult(
        id_documento=doc.id,
        tipo_documento=classification.document_type,
        fecha_inscripcion=(
            validation.fecha_inscripcion.isoformat()
            if validation.fecha_inscripcion
            else extraction.fecha_inscripcion_raw
        ),
        fecha_renovacion=(
            validation.fecha_renovacion.isoformat()
            if validation.fecha_renovacion
            else extraction.fecha_renovacion_raw
        ),
        estado_validacion=validation.status.value,
        motivo=validation.motivo,
        nivel_confianza=extraction.confidence,
    )
