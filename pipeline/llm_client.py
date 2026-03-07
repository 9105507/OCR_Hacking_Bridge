"""Thin wrapper around the Ollama Vision API."""

from __future__ import annotations

import json
import logging
import re

import httpx

from pipeline import config

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server is unreachable."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not pulled locally."""


class OllamaVisionClient:
    """Synchronous client for Ollama's ``/api/generate`` endpoint with image support."""

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout: int = config.OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, image_base64: str) -> str:
        """Send a vision prompt and return the full text response.

        Args:
            prompt: The text prompt sent alongside the image.
            image_base64: Base64-encoded JPEG image.

        Returns:
            The concatenated text response from the model.

        Raises:
            OllamaConnectionError: Server unreachable.
            OllamaModelNotFoundError: Model not available locally.
            OllamaError: Any other Ollama-related failure.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": True,
            "think": False,
        }

        logger.info("[LLM] Sending request to %s (model=%s) ...", url, self.model)
        try:
            with self._client.stream("POST", url, json=payload) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.model}' not found. "
                        f"Run: ollama pull {self.model}"
                    )
                response.raise_for_status()
                result = self._read_streamed_response(response)
                logger.info("[LLM] Response received (%d chars).", len(result))
                return result
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the server running? (ollama serve)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise OllamaError("Ollama request timed out") from exc

    def close(self) -> None:
        self._client.close()

    # Context-manager support
    def __enter__(self) -> OllamaVisionClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _read_streamed_response(response: httpx.Response) -> str:
        """Concatenate ``response`` fields from Ollama's NDJSON stream."""
        chunks: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    chunks.append(chunk)
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON line from Ollama: %s", line[:120])
        text = "".join(chunks)
        # Strip Qwen3 <think>…</think> reasoning blocks if present
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
