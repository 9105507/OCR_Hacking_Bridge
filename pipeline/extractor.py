"""Extract inscription and renewal dates from a classified document via VLM."""

from __future__ import annotations

import json
import logging
import re

from pipeline.llm_client import OllamaVisionClient
from pipeline.models import DocumentInput, ExtractionResult

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT_TEMPLATE = """\
This image is a Spanish {doc_type} unemployment document.
Extract ONLY these two dates — do NOT extract any names, DNI numbers, or addresses:

1. "fecha_inscripcion": The initial registration / inscription date (Fecha de Inscripción).
2. "fecha_renovacion": The first renewal date (Fecha de la próxima renovación / Próxima renovación de la demanda).

Dates may appear as "dd/mm/yyyy", "dd de mes de yyyy", "dd-mm-yyyy", or similar.

Reply ONLY with a JSON object, no other text:
{{"fecha_inscripcion": "...", "fecha_renovacion": "...", "confidence": 0.0 to 1.0}}

If you cannot find one of the dates, set its value to null.
"""


def _parse_extraction_json(raw: str) -> dict:
    """Extract a JSON object from the LLM response, tolerant of markdown fences."""
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def extract_dates(
    client: OllamaVisionClient,
    doc: DocumentInput,
    doc_type: str = "DARDE",
) -> ExtractionResult:
    """Send the document image to the VLM and extract the two key dates.

    Returns an ``ExtractionResult`` — always succeeds (never raises).
    """
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(doc_type=doc_type)

    try:
        raw_response = client.generate(prompt, doc.base64_image)
        logger.debug("Extraction raw response for %s: %s", doc.id, raw_response)
    except Exception:
        logger.exception("LLM call failed during extraction of %s", doc.id)
        return ExtractionResult(raw_llm_response="LLM call failed.")

    data = _parse_extraction_json(raw_response)
    if not data:
        logger.warning("Could not parse extraction JSON for %s", doc.id)
        return ExtractionResult(raw_llm_response=raw_response[:500])

    fecha_insc = data.get("fecha_inscripcion")
    fecha_reno = data.get("fecha_renovacion")

    return ExtractionResult(
        fecha_inscripcion_raw=fecha_insc if fecha_insc else None,
        fecha_renovacion_raw=fecha_reno if fecha_reno else None,
        confidence=float(data.get("confidence", 0.0)),
        raw_llm_response=raw_response[:500],
    )
