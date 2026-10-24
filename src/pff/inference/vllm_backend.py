"""Real vLLM-backed inference (design 5.2).

A drop-in :class:`InferenceBackend` that talks to a vLLM OpenAI-compatible server
running locally (same machine/GPU, e.g. a Kaggle T4 kernel). It implements the exact
same surface as :class:`pff.inference.mock_backend.MockBackend`, so the env, agents,
quadrant generator and detectors run against it unchanged.

Unlike the mock, this backend sends the full prompt (including the ``<<PFF_CONTROL>>``
block, which a real model simply treats as context) to an actual LLM and returns its
real completion. This is the GPU-spending path used to generate real trajectories.

Usage (inside a Kaggle notebook after `vllm serve` is up on :8000)::

    backend = VLLMBackend(base_url="http://localhost:8000/v1",
                          model_aliases={"mock/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct"})
"""

from __future__ import annotations

import time

from .interface import Completion, InferenceBackend


class VLLMBackend(InferenceBackend):
    """Chat backend backed by a vLLM OpenAI-compatible HTTP server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        *,
        model_aliases: dict[str, str] | None = None,
        max_retries: int = 4,
        request_timeout: float = 120.0,
    ) -> None:
        # Imported lazily so the package stays importable without `openai` installed
        # (the mock-only test environment does not need it).
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        # Map the quadrant table's logical model ids (which may be "mock/...") to the
        # real served model names. Anything not in the map is passed through unchanged.
        self._aliases = dict(model_aliases or {})
        self._max_retries = max_retries
        self._served: list[str] | None = None

    def _resolve(self, model: str) -> str:
        return self._aliases.get(model, model)

    def available_models(self) -> list[str]:
        if self._served is None:
            self._served = [m.id for m in self._client.models.list().data]
        return list(self._served)

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
        served_model = self._resolve(model)
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=served_model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                choice = resp.choices[0]
                return Completion(
                    text=choice.message.content or "",
                    model=model,
                    finish_reason=choice.finish_reason or "stop",
                    seed=seed,
                    temperature=temperature,
                )
            except Exception as e:  # noqa: BLE001 - network/server transient
                last_err = e
                time.sleep(min(2.0 * (attempt + 1), 8.0))
        raise RuntimeError(
            f"vLLM chat failed after {self._max_retries} attempts for model "
            f"{served_model!r}: {last_err}"
        )
