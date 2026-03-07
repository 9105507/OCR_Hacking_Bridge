"""Classify whether an image is a legible DARDE/DARDO document (fail-fast)."""

from __future__ import annotations

import json
import logging
import re

from pipeline.llm_client import OllamaVisionClient
from pipeline.models import ClassificationResult, DocumentInput

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """\
Analyze this image and determine:
1. Is it a legible photograph or scan (not blurry/cut off)?
2. Is it a Spanish official unemployment document?
   - DARDE: from Comunidad de Madrid (in Spanish/Castilian).
   - DARDO: from Cataluña / Catalonia (may contain Catalan text).

Reply ONLY with a JSON object, no extra text:
{"is_document": true/false, "is_legible": true/false, "type": "DARDE" or "DARDO" or "UNKNOWN", "confidence": 0.0 to 1.0}
"""


def _parse_classification_json(raw: str) -> dict:
    """Extract a JSON object from the LLM response, tolerant of markdown fences."""
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences and retry
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find first { … }
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def classify_document(
    client: OllamaVisionClient,
    doc: DocumentInput,
) -> ClassificationResult:
    """Ask the VLM whether *doc* is a valid, legible DARDE/DARDO document.

    Returns a ``ClassificationResult`` — always succeeds (never raises).
    On LLM failure the result defaults to ``is_valid_document=False``.
    """
    try:
        logger.info("[CLASSIFY] Starting classification for %s ...", doc.id)
        raw_response = client.generate(CLASSIFICATION_PROMPT, doc.base64_image)
        logger.info("[CLASSIFY] Classification done for %s.", doc.id)
        logger.debug("Classification raw response for %s: %s", doc.id, raw_response)
    except Exception:
        logger.exception("LLM call failed during classification of %s", doc.id)
        return ClassificationResult(
            reason="LLM call failed during classification.",
        )

    data = _parse_classification_json(raw_response)
    if not data:
        logger.warning("Could not parse classification JSON for %s", doc.id)
        return ClassificationResult(
            reason=f"Unparseable LLM response: {raw_response[:200]}",
        )

    doc_type_raw = str(data.get("type", "UNKNOWN")).upper()
    if doc_type_raw not in {"DARDE", "DARDO"}:
        doc_type_raw = "UNKNOWN"

    return ClassificationResult(
        is_valid_document=bool(data.get("is_document", False)),
        is_legible=bool(data.get("is_legible", False)),
        document_type=doc_type_raw,  # type: ignore[arg-type]
        confidence=float(data.get("confidence", 0.0)),
    )
