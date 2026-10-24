"""Modal/vLLM real inference backend (FR-3, design 5.2 / 5.4 / 9).

A drop-in :class:`InferenceBackend` that talks to the Modal-hosted vLLM
OpenAI-compatible server (:mod:`modal_app.vllm_server`) via the ``openai`` client. It
implements the identical surface as :class:`pff.inference.mock_backend.MockBackend`, so
the env, agents, quadrant generator and detectors run against it unchanged -- this is the
single abstraction seam the whole pipeline depends on.

Unlike the mock, this backend sends the full prompt (including the ``<<PFF_CONTROL>>``
block, which a real model simply treats as context) to an actual LLM and returns its real
completion. This is the GPU-spending path used to generate real trajectories, and it only
runs after GATE 3 (the real Q1 positive control) passes.

Error handling (design 9): network / timeout / 5xx are retried with exponential backoff
(max 3 retries by default); on exhaustion a :class:`BackendError` is raised (fatal to that
run). Missing Modal auth / an unreachable endpoint also raise :class:`BackendError` before
any sweep spend.

Model aliasing: the quadrant table uses logical ids (``mock/Qwen2.5-7B-Instruct``,
``mock/Llama-3.1-8B-Instruct``); ``model_aliases`` maps them to the real served names
(``Qwen/Qwen2.5-7B-Instruct``, ``meta-llama/Llama-3.1-8B-Instruct``).
"""

from __future__ import annotations

import time

from .interface import Completion, InferenceBackend

# Default logical-id -> served-name map (design 6 / quadrants.config model ids).
DEFAULT_MODEL_ALIASES = {
    "mock/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "mock/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


class BackendError(RuntimeError):
    """Fatal real-backend failure (design 9: raised on retry exhaustion / bad config)."""


class ModalVLLMBackend(InferenceBackend):
    """Real chat backend over the Modal-hosted vLLM OpenAI server.

    Parameters
    ----------
    base_url:
        The Modal web endpoint URL with the ``/v1`` suffix, e.g.
        ``https://<workspace>--plus-factor-forensics-vllm-serve.modal.run/v1``.
    api_key:
        vLLM's OpenAI server ignores the key but the client requires a non-empty value.
    model_aliases:
        Logical-id -> served-name map (defaults to :data:`DEFAULT_MODEL_ALIASES`); any id
        not present is passed through unchanged.
    max_retries / backoff_base / backoff_cap:
        Exponential-backoff retry policy (design 9).
    request_timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        *,
        model_aliases: dict[str, str] | None = None,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        backoff_cap: float = 12.0,
        request_timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - openai is a pinned dep
            raise BackendError(
                "the 'openai' client is required for the real Modal backend"
            ) from exc
        if not base_url:
            raise BackendError("base_url (the Modal vLLM endpoint) is required")
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self._aliases = {**DEFAULT_MODEL_ALIASES, **(model_aliases or {})}
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._served: list[str] | None = None

    def _resolve(self, model: str) -> str:
        return self._aliases.get(model, model)

    def available_models(self) -> list[str]:
        """List the served model ids (cached). Raises :class:`BackendError` if offline."""
        if self._served is None:
            try:
                self._served = [m.id for m in self._client.models.list().data]
            except Exception as exc:  # noqa: BLE001 - surface any transport error as fatal
                raise BackendError(f"could not reach the Modal vLLM endpoint: {exc}") from exc
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
        """Return a real :class:`Completion`, retrying transient failures (design 9)."""
        served_model = self._resolve(model)
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
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
                    model=model,  # echo the logical id, not the served name
                    finish_reason=choice.finish_reason or "stop",
                    seed=seed,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001 - network/server transient
                last_err = exc
                if attempt < self._max_retries:
                    time.sleep(min(self._backoff_base ** (attempt + 1), self._backoff_cap))
        raise BackendError(
            f"Modal vLLM chat failed after {self._max_retries + 1} attempts for model "
            f"{served_model!r}: {last_err}"
        )
