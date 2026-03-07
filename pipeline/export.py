"""Export pipeline results to CSV and JSON."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from pipeline.models import PipelineResult

logger = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "id_documento",
    "fecha_inscripcion",
    "fecha_renovacion",
    "estado_validacion",
    "motivo",
    "nivel_confianza",
]


def export_results(results: list[PipelineResult], output_dir: Path) -> None:
    """Write *results* as ``resultados.csv`` and ``resultados.json``.

    Creates *output_dir* if it does not exist. Files are UTF-8 encoded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(results, output_dir / "resultados.csv")
    _write_json(results, output_dir / "resultados.json")


def _write_csv(results: list[PipelineResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.model_dump())
    logger.info("CSV written to %s (%d rows)", path, len(results))


def _write_json(results: list[PipelineResult], path: Path) -> None:
    payload = [r.model_dump() for r in results]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("JSON written to %s (%d entries)", path, len(results))
