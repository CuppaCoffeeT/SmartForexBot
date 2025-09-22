
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the local LLM call fails."""


@dataclass
class LLMResponse:
    text: str


class OllamaClient:
    """Simple wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT", "6"))

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        options = {
            "temperature": max(0.0, float(temperature)),
        }
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response")
            if not text:
                raise LLMClientError("Ollama returned an empty response")
            return LLMResponse(text=text.strip())
        except requests.RequestException as exc:
            raise LLMClientError(f"Ollama request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMClientError("Failed to decode Ollama response") from exc


_client: Optional[OllamaClient] = None
_client_checked = False


def get_client() -> Optional[OllamaClient]:
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    enabled_flag = os.getenv("LLM_ENABLED")
    if enabled_flag is not None and enabled_flag.lower() in {"0", "false", "no"}:
        logger.info("Local LLM explicitly disabled via LLM_ENABLED")
        _client = None
        return _client

    if provider and provider not in {"ollama"}:
        logger.warning("Unsupported LLM provider '%s'; disabling LLM integration", provider)
        _client = None
        return _client

    try:
        _client = OllamaClient()
    except Exception as exc:  # noqa: BLE001 - defensive guard around env configuration
        logger.warning("Failed to initialise Ollama client: %s", exc)
        _client = None
    return _client


def has_client() -> bool:
    return get_client() is not None


def generate_text(
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> Optional[str]:
    client = get_client()
    if not client:
        return None
    try:
        result = client.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return result.text
    except LLMClientError as exc:
        logger.warning("Local LLM call failed: %s", exc)
        return None

