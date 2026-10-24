"""Minimal GPU access probe across tiers. No model load — just confirm Modal will
start a GPU container on this account, and on which tier. Seconds of runtime.

Run:  .venv/bin/python -m modal run scripts/gpu_probe.py
"""
from __future__ import annotations

import modal

app = modal.App("pff-gpu-probe")
img = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.5.1")


def _probe(tier: str) -> str:
    import torch

    if not torch.cuda.is_available():
        return f"{tier}: container started but CUDA not available"
    return f"{tier}: OK -> {torch.cuda.get_device_name(0)}"


@app.function(image=img, gpu="T4", timeout=300)
def probe_t4() -> str:
    return _probe("T4")


@app.function(image=img, gpu="A10G", timeout=300)
def probe_a10g() -> str:
    return _probe("A10G")


@app.local_entrypoint()
def main():
    for name, fn in [("T4", probe_t4), ("A10G", probe_a10g)]:
        try:
            print(fn.remote())
        except Exception as e:  # noqa: BLE001
            print(f"{name}: BLOCKED -> {type(e).__name__}: {e}")
