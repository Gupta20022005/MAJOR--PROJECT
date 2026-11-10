"""
Common LLM-call interface, shared by every provider (Gemini, Ollama, or any
future one). The point: query/pipeline.py calls `generate_with_retry(prompt,
provider)` exactly the same way regardless of which provider it's holding —
switching providers is a config/constructor change, never a call-site change.

Each provider only has to implement two things:
    - `_generate(prompt) -> str`         the actual API/local call
    - `is_transient(error) -> bool`      whether this error is worth retrying
    - `is_rate_limited(error) -> bool`   whether this error deserves the longer,
                                          rate-limit-style backoff (optional;
                                          defaults to False)

`generate_with_retry` implements the retry/backoff loop once, generically,
against that interface — so retry logic isn't duplicated per provider.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

from config import LLM_PROVIDER, GEMINI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger("query.llm")


class LLMError(Exception):
    """Raised when a provider's call fails after exhausting all retries."""


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
class BaseLLMProvider:
    name = "base"

    def _generate(self, prompt: str) -> str:
        raise NotImplementedError

    def is_transient(self, error: Exception) -> bool:
        """Whether this error is worth retrying at all (vs. failing fast)."""
        raise NotImplementedError

    def is_rate_limited(self, error: Exception) -> bool:
        """Whether this error should get the longer, rate-limit-style backoff.
        Providers with no concept of rate limiting (e.g. local inference)
        can leave this as the default `False`."""
        return False


# ---------------------------------------------------------------------------
# Gemini (cloud API) — transient errors are mostly rate limits / 5xx / network
# ---------------------------------------------------------------------------
class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    RATE_LIMIT_SUBSTRINGS = ("rate limit", "429", "quota", "resource_exhausted")
    TRANSIENT_SUBSTRINGS = RATE_LIMIT_SUBSTRINGS + (
        "timeout", "timed out", "500", "502", "503", "504", "overloaded", "unavailable", "connection",
    )

    def __init__(self, model: str = GEMINI_MODEL):
        self.model = model

    def _generate(self, prompt: str) -> str:
        from google import genai
        client = genai.Client()  # reads GEMINI_API_KEY from env
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def is_transient(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(sub in text for sub in self.TRANSIENT_SUBSTRINGS)

    def is_rate_limited(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(sub in text for sub in self.RATE_LIMIT_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Ollama (local inference) — transient errors are server-not-up, cold-start
# timeouts, or the model still loading. No concept of rate limiting.
# ---------------------------------------------------------------------------
class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    TRANSIENT_SUBSTRINGS = (
        "timeout", "timed out", "connection", "refused", "loading model",
        "model is loading", "500", "502", "503", "not found", "unavailable",
    )

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _generate(self, prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def is_transient(self, error: Exception) -> bool:
        text = str(error).lower()
        return any(sub in text for sub in self.TRANSIENT_SUBSTRINGS)

    def is_rate_limited(self, error: Exception) -> bool:
        return False  # local inference has no rate limits, just retries as normal transient


# ---------------------------------------------------------------------------
# Factory — resolves LLM_PROVIDER (config/env) to a provider instance
# ---------------------------------------------------------------------------
def get_llm_provider(name: Optional[str] = None) -> BaseLLMProvider:
    """
    Returns the configured provider. `name` overrides config.LLM_PROVIDER
    (which itself reads the LLM_PROVIDER env var, default "gemini").
    This is the ONLY place that needs to know about specific providers —
    everything downstream just calls `generate_with_retry(prompt, provider)`.
    """
    resolved = (name or LLM_PROVIDER).lower()
    if resolved == "gemini":
        return GeminiProvider()
    if resolved == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{resolved}'. Expected 'gemini' or 'ollama'.")


# ---------------------------------------------------------------------------
# Shared retry loop — identical logic regardless of provider
# ---------------------------------------------------------------------------
def generate_with_retry(
    prompt: str,
    provider: BaseLLMProvider,
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
) -> str:
    """
    Calls `provider._generate(prompt)` with retry + exponential backoff.
    Errors the provider flags as rate-limited get a longer backoff; other
    transient errors get normal backoff; non-transient errors fail immediately.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return provider._generate(prompt)
        except Exception as e:  # noqa: BLE001 - providers raise varied exception types
            last_error = e

            if attempt >= max_retries:
                break

            if not provider.is_transient(e):
                logger.error(f"[{provider.name}] non-transient error, not retrying: {e}")
                break

            extra_backoff = 2 if provider.is_rate_limited(e) else 1
            delay = min(base_delay * (2 ** attempt) * extra_backoff, max_delay)
            kind = "rate limited" if provider.is_rate_limited(e) else "transient error"
            logger.warning(
                f"[{provider.name}] {kind} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    raise LLMError(f"[{provider.name}] call failed after {max_retries + 1} attempt(s): {last_error}") from last_error
