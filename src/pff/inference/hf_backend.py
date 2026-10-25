"""Real inference via HuggingFace transformers (no server, no vLLM).

This backend loads an instruct model with transformers and calls ``.generate()``
directly. It is the robust path for constrained environments (e.g. Kaggle free T4)
where vLLM's CUDA-kernel compilation fights the preinstalled torch/CUDA stack: it
uses whatever torch is already present and compiles nothing.

It implements the exact :class:`InferenceBackend` surface, so the env, agents,
quadrant generator and detectors run against it unchanged. Outputs are short (one
``PRICE=<idx>`` line plus optional CHANNEL/PLAN), so per-call latency is acceptable
even without vLLM's batching.

Determinism: with ``temperature == 0`` generation is greedy (deterministic); otherwise
the per-call ``seed`` is set via ``transformers.set_seed`` before sampling, giving the
Q3 same-checkpoint control reproducible-per-seed behavior.
"""

from __future__ import annotations

from .interface import Completion, InferenceBackend


class HFBackend(InferenceBackend):
    """Chat backend backed by a locally loaded transformers model."""

    def __init__(
        self,
        model_name: str,
        *,
        model_aliases: dict[str, str] | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens_cap: int = 96,
    ) -> None:
        import torch  # noqa: F401 - ensure torch present
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model_name = model_name
        self._aliases = dict(model_aliases or {})
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        self._model.eval()
        self._cap = max_new_tokens_cap

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
        import torch
        from transformers import set_seed

        set_seed(int(seed))
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        do_sample = temperature is not None and temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=min(max_tokens, self._cap),
            do_sample=do_sample,
            pad_token_id=self._tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        text = self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # Honor simple stop strings if provided.
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
