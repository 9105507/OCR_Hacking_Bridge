"""Normalise extracted date strings and apply the core business rule."""

from __future__ import annotations

import logging
import re
from datetime import date

from dateutil import parser as dateutil_parser

from pipeline import config
from pipeline.models import ExtractionResult, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

# Mapping of Spanish and Catalan month names → numeric month (lowercase).
_MONTH_NAMES: dict[str, str] = {
    # Spanish
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
    # Catalan
    "gener": "01",
    "febrer": "02",
    "març": "03",
    "maig": "05",
    "juny": "06",
    "juliol": "07",
    "agost": "08",
    "setembre": "09",
    "novembre": "11",
    "desembre": "12",
    # Common abbreviations (shared / Spanish)
    "ene": "01",
    "feb": "02",
    "mar": "03",
    "abr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dic": "12",
    # Catalan abbreviations
    "gen": "01",
    "des": "12",
    "set": "09",
}

# Pre-compiled pattern: "12 de mayo de 2024", "12 mayo 2024", "12 de maig de 2024"
_DATE_RE = re.compile(
    r"(\d{1,2})\s*(?:de\s+|d')?\s*("
    + "|".join(_MONTH_NAMES.keys())
    + r")\s*(?:de\s+|del\s+)?(\d{4})",
    re.IGNORECASE,
)


def normalize_date(raw: str | None) -> date | None:
    """Best-effort conversion of a raw date string to ``datetime.date``.

    Handles formats like:
    - ``12/05/2024``
    - ``12-05-2024``
    - ``12 de mayo de 2024``
    - ``2024-05-12`` (ISO)

    Returns ``None`` when the string cannot be parsed.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()

    # --- Try Spanish / Catalan textual months first ---
    match = _DATE_RE.search(raw.lower())
    if match:
        day, month_name, year = match.groups()
        month_num = _MONTH_NAMES.get(month_name.lower())
        if month_num:
            try:
                return date(int(year), int(month_num), int(day))
            except ValueError:
                pass

    # --- Fallback to dateutil with dayfirst=True (European convention) ---
    try:
        return dateutil_parser.parse(raw, dayfirst=True).date()
    except (ValueError, OverflowError):
        logger.warning("Cannot parse date: %r", raw)
        return None


def validate_document(
    extraction: ExtractionResult,
    program_start: date = config.PROGRAM_START_DATE,
) -> ValidationResult:
    """Normalise dates and apply the business rule.

    Rule: ``fecha_inscripcion < program_start < fecha_renovacion``

    Status outcomes:
    - **VÁLIDO** — both dates present and rule satisfied.
    - **NO VÁLIDO** — both dates present but rule NOT satisfied.
    - **REQUIERE REVISIÓN HUMANA** — at least one date missing / unparseable.
    """
    f_insc = normalize_date(extraction.fecha_inscripcion_raw)
    f_reno = normalize_date(extraction.fecha_renovacion_raw)

    if f_insc is None or f_reno is None:
        motivo_parts: list[str] = []
        if f_insc is None:
            motivo_parts.append(
                f"Fecha de inscripción no reconocida (raw={extraction.fecha_inscripcion_raw!r})"
            )
        if f_reno is None:
            motivo_parts.append(
                f"Fecha de renovación no reconocida (raw={extraction.fecha_renovacion_raw!r})"
            )
        return ValidationResult(
            fecha_inscripcion=f_insc,
            fecha_renovacion=f_reno,
            status=ValidationStatus.REQUIERE_REVISION,
            motivo="; ".join(motivo_parts),
        )

    rule_ok = f_insc < program_start < f_reno

    if rule_ok:
        return ValidationResult(
            fecha_inscripcion=f_insc,
            fecha_renovacion=f_reno,
            status=ValidationStatus.VALIDO,
            motivo=(
                f"Inscripción ({f_insc.isoformat()}) < "
                f"Inicio programa ({program_start.isoformat()}) < "
                f"Renovación ({f_reno.isoformat()})"
            ),
        )

    return ValidationResult(
        fecha_inscripcion=f_insc,
        fecha_renovacion=f_reno,
        status=ValidationStatus.NO_VALIDO,
        motivo=(
            f"Regla no cumplida: inscripción={f_insc.isoformat()}, "
            f"renovación={f_reno.isoformat()}, "
            f"inicio_programa={program_start.isoformat()}"
        ),
    )
