"""Real inference via the Kaggle Benchmarks LLM proxy (frontier-model arm).

Kaggle Benchmarks injects an ``llm`` object whose ``prompt()`` method calls a SOTA model
(Gemini, Claude, DeepSeek, Qwen, ...) and spends the account's benchmark AI credit. The
model is NOT chosen in code: it is selected on the task page ("Evaluate More Models"),
and the task is re-run once per selected model.

This adapter implements our :class:`InferenceBackend` surface on top of that injected
``llm`` object, using an isolated chat per call so our stateless per-round ``chat()``
contract holds and context does not accumulate across agents/rounds. That lets the
existing Bertrand env, quadrant generator and detectors run unchanged on a frontier model.

Caveats for the paper (state honestly):
- Hosted proxy: the exact checkpoint is not pinned and the sampling seed is not honored,
  so for this arm Q3 ("innocent parallelism") is "same proxied model, no agreement"
  rather than the byte-identical same-checkpoint control used for self-hosted open weights.
  Valid but slightly weaker negative control.
"""

from __future__ import annotations

from .interface import Completion, InferenceBackend


class KbenchBackend(InferenceBackend):
    """Adapter from Kaggle Benchmarks' injected ``llm`` to our InferenceBackend seam."""

    def __init__(self, kbench_llm, model_label: str = "kbench-proxy") -> None:
        # kbench_llm is the object injected into a @kbench.task function (kbench.llm).
        self._llm = kbench_llm
        self._label = model_label
        # Imported lazily; only present inside a Kaggle Benchmarks runtime.
        import kaggle_benchmarks as kbench  # noqa: F401

        self._kbench = kbench
        self._chat_counter = 0

    def available_models(self) -> list[str]:
        return [self._label]

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
        # Flatten system+user into a single prompt (our prompts already carry all state;
        # the control block is ignored by a real model). Use an isolated chat so no
        # history accumulates across calls — matches the stateless chat() contract.
        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        prompt = (system + "\n\n" + user).strip() if system else user

        self._chat_counter += 1
        chat_name = f"pff_{seed}_{self._chat_counter}"
        try:
            with self._kbench.chats.new(chat_name):
                text = self._llm.prompt(prompt)
        except Exception:
            # Fall back to a plain prompt if isolated-chat context mgmt is unavailable.
            text = self._llm.prompt(prompt)

        text = text if isinstance(text, str) else str(text)
        if stop:
            for s in stop:
                i = text.find(s)
                if i != -1:
                    text = text[:i]
        return Completion(
            text=text.strip(),
            model=model,
            finish_reason="stop",
            seed=seed,
            temperature=temperature if temperature is not None else 0.0,
        )
