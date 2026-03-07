"""Centralized configuration for the pipeline."""

from datetime import date
from pathlib import Path

# --- Business Rule ---
PROGRAM_START_DATE: date = date(2025, 3, 1)

# --- Ollama Settings ---
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "qwen3-vl:latest"
OLLAMA_TIMEOUT_SECONDS: int = 120

# --- Paths ---
OUTPUT_DIR: Path = Path("output")

# --- Confidence Thresholds ---
CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.6
EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.5
