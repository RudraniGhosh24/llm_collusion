"""AC-3 (GPU-gated): the Modal vLLM backend serves both model families.

Skipped unless a real Modal vLLM endpoint is wired via ``PFF_MODAL_VLLM_URL``. Also
asserts (always, at $0) that ``modal_app/vllm_server.py`` imports and that the real client
constructs and resolves logical model ids to served names without a GPU.

GPU ACCESS NOTE: Modal GPU functions on this account require a payment method, so the live
part is skipped in the committed $0 run.
"""

from __future__ import annotations

import os

import pytest

from pff.inference.modal_client import DEFAULT_MODEL_ALIASES, BackendError, ModalVLLMBackend

MODAL_URL = os.environ.get("PFF_MODAL_VLLM_URL")
GPU_REASON = (
    "real Modal vLLM endpoint not wired (set PFF_MODAL_VLLM_URL); Modal GPU requires a "
    "payment method on this account"
)


def test_modal_app_imports():
    """The Modal app module imports cleanly (no GPU needed to define the app)."""
    import modal_app.vllm_server as srv

    assert srv.app.name == "plus-factor-forensics-vllm"
    assert "Qwen/Qwen2.5-7B-Instruct" in srv.SERVED_MODELS
    assert "meta-llama/Llama-3.1-8B-Instruct" in srv.SERVED_MODELS


def test_real_client_resolves_logical_ids_without_gpu():
    """The client maps the quadrant table's logical ids to real served names."""
    backend = ModalVLLMBackend(base_url="http://example.invalid/v1")
    assert backend._resolve("mock/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert backend._resolve("mock/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"
    # Unknown ids pass through unchanged.
    assert backend._resolve("Qwen/Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert set(DEFAULT_MODEL_ALIASES.values()) == {
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    }


def test_real_client_requires_base_url():
    with pytest.raises(BackendError):
        ModalVLLMBackend(base_url="")


@pytest.mark.skipif(MODAL_URL is None, reason=GPU_REASON)
def test_modal_serves_both_models():
    """AC-3: the live endpoint serves both Qwen and Llama and returns real completions."""
    backend = ModalVLLMBackend(base_url=MODAL_URL)
    served = backend.available_models()
    assert any("Qwen2.5-7B-Instruct" in m for m in served)
    for model in ("mock/Qwen2.5-7B-Instruct", "mock/Llama-3.1-8B-Instruct"):
        comp = backend.chat(
            model,
            [
                {"role": "system", "content": "Reply with exactly: PRICE=5"},
                {"role": "user", "content": "Round 1. Choose your price index."},
            ],
            temperature=0.0,
            seed=0,
            max_tokens=16,
        )
        assert isinstance(comp.text, str) and comp.text != ""
