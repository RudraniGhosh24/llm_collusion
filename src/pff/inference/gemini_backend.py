"""Real inference via Google Gemini (frontier-model generalization arm).

A drop-in :class:`InferenceBackend` for the "does the method generalize beyond small
open-weight models?" experiment. Uses Gemini's OpenAI-compatible endpoint so it reuses
the same chat-completions call shape as the vLLM backend.

Caveats for the paper (state honestly):
- Gemini is a hosted API: the exact checkpoint is not pinned and may change, and the
  provider's sampling seed is not guaranteed honored. So for a Gemini arm the Q3
  "innocent parallelism" control is "same API model, no agreement" rather than the
  byte-identical same-checkpoint control available for self-hosted open weights. It is a
  valid but slightly weaker negative control.
"""

from __future__ import annotations

import os
import time

from .interface import Completion, InferenceBackend

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiBackend(InferenceBackend):
    """Chat backend backed by the Gemini OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        model_aliases: dict[str, str] | None = None,
        max_retries: int = 5,
        request_timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("set GEMINI_API_KEY (or pass api_key=) for the Gemini backend")
        self._client = OpenAI(base_url=GEMINI_OPENAI_BASE, api_key=key, timeout=request_timeout)
        self._model_name = model_name
        self._aliases = dict(model_aliases or {})
        self._max_retries = max_retries

    def _resolve(self, model: str) -> str:
        return self._aliases.get(model, self._model_name)

    def available_models(self) -> list[str]:
        return [self._model_name]

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> Completion:
        served = self._resolve(model)
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=served,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    # Gemini's OpenAI shim accepts seed but may not honor it; sent for parity.
                    seed=seed,
                )
                choice = resp.choices[0]
                return Completion(
                    text=choice.message.content or "",
                    model=model,
                    finish_reason=choice.finish_reason or "stop",
                    seed=seed,
                    temperature=temperature,
                )
            except Exception as e:  # noqa: BLE001 - API transient / rate limit
                last_err = e
                time.sleep(min(2.0 * (attempt + 1), 15.0))
        raise RuntimeError(
            f"Gemini chat failed after {self._max_retries} attempts for {served!r}: {last_err}"
        )
