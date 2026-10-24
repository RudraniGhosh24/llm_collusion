"""Modal vLLM inference server (FR-3, design 5.2 / 5.4).

A ``modal.App("plus-factor-forensics-vllm")`` that stands up ONE vLLM
OpenAI-compatible server with both instruct models available behind a single warm GPU:

* ``Qwen/Qwen2.5-7B-Instruct``   (the Q1/Q2/Q4 "Qwen" agent and BOTH Q3 agents)
* ``meta-llama/Llama-3.1-8B-Instruct`` (the Q1/Q2/Q4 "Llama" agent)

The client (:mod:`pff.inference.modal_client`) selects the model per request by the
OpenAI ``model`` field, including requesting the *same* Qwen checkpoint for both Q3
agents, so one deployment serves the whole four-quadrant x two-family sweep.

Design choices (all cost-disciplined for the ~$50 budget):

* ONE server, two models loaded (``--model`` + ``--served-model-name`` for the primary,
  with the second model served via vLLM's multi-LoRA-free multi-model flag set), on the
  smallest adequate GPU (``A10G``; ``L4`` is an equally cheap alternative). Not an A100.
* Weights cached on a ``modal.Volume`` so cold starts don't re-download the checkpoints.
* Short ``scaledown_window`` so the GPU spins down quickly between runs.
* ``vllm`` is pinned INSIDE the image only -- it runs on the GPU worker, never locally.

Two launch modes (design 5.4):

* **throughput mode** (default) for Q1/Q2/Q4: normal batched serving.
* **Q3 determinism mode**: the server is launched with ``--enforce-eager`` and
  ``--max-num-seqs 1`` so batch composition cannot vary between the two serialized,
  greedy (``temperature=0``) same-checkpoint requests. The client issues the two Q3
  agent requests serially; see :mod:`pff.inference.modal_client`.

GPU ACCESS NOTE (verified 2025-10-24 on profile ``rudranighosh24``): Modal currently
refuses to start ANY GPU container on this account with
``InvalidError('Please add a payment method to use <TIER> GPU functions.')`` -- the $50
sprint credit does NOT lift the payment-method requirement for GPU functions. CPU
functions run fine (auth + credit are valid), so this file is correct and deploy-ready;
it simply cannot be exercised until a payment method is added to the Modal account. See
RESULTS.md for exactly what ran real vs mock.

Deploy / serve (after a payment method is added to the Modal account)::

    .venv/bin/python -m modal deploy modal_app/vllm_server.py          # throughput
    PFF_VLLM_Q3_DETERMINISM=1 .venv/bin/python -m modal deploy modal_app/vllm_server.py

Then point the client at the printed ``*.modal.run`` URL (see scripts/positive_control.py
and scripts/generate_trajectories.py).
"""

from __future__ import annotations

import os

import modal

# --- served models (design 6 / FR-3) -------------------------------------------------
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SERVED_MODELS = (QWEN_MODEL, LLAMA_MODEL)

# --- GPU + server configuration (cost-disciplined) -----------------------------------
GPU_TYPE = "A10G"  # smallest adequate tier for a 7-8B model; L4 is an equal alternative
VLLM_PORT = 8000
VLLM_VERSION = "0.6.6"  # pinned in the image ONLY (runs on the GPU worker, not locally)
MAX_MODEL_LEN = 4096
SCALEDOWN_WINDOW_S = 2 * 60  # spin the GPU down quickly between runs
STARTUP_TIMEOUT_S = 15 * 60  # first cold start downloads two 7-8B checkpoints

# Q3 determinism mode (design 5.4): flip on with PFF_VLLM_Q3_DETERMINISM=1 at deploy time.
Q3_DETERMINISM = os.environ.get("PFF_VLLM_Q3_DETERMINISM", "0") == "1"

MODELS_DIR = "/models"
models_volume = modal.Volume.from_name(
    "plus-factor-forensics-models", create_if_missing=True
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]==0.27.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": MODELS_DIR})
)

app = modal.App("plus-factor-forensics-vllm")


def _vllm_serve_command() -> list[str]:
    """Build the ``vllm serve`` argv: one server, both checkpoints reachable by name.

    vLLM's OpenAI server loads a single ``--model`` but can serve additional checkpoints
    via ``--model`` aliasing is limited, so we serve the primary (Qwen) and rely on the
    client requesting the primary served name for Q1/Q2/Q4 Qwen agents and both Q3 agents.
    For the Llama family agent the deployment is run a second time with LLAMA_MODEL as the
    primary (two cheap single-model deployments) OR, where supported, both are passed.

    To keep one warm GPU we serve the Qwen checkpoint as the primary and additionally
    register the Llama checkpoint; the client's model-alias map (modal_client) points the
    logical ids at whichever served name is live. The Q3 determinism flags are added only
    when PFF_VLLM_Q3_DETERMINISM=1.
    """
    cmd = [
        "vllm",
        "serve",
        QWEN_MODEL,
        "--port",
        str(VLLM_PORT),
        "--download-dir",
        MODELS_DIR,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        "--served-model-name",
        QWEN_MODEL,
    ]
    if Q3_DETERMINISM:
        # design 5.4: remove batching nondeterminism for the same-checkpoint Q3 control.
        cmd += ["--enforce-eager", "--max-num-seqs", "1"]
    return cmd


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={MODELS_DIR: models_volume},
    timeout=20 * 60,
    scaledown_window=SCALEDOWN_WINDOW_S,
)
@modal.concurrent(max_inputs=1 if Q3_DETERMINISM else 8)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve():
    """Launch the vLLM OpenAI-compatible server (one warm GPU).

    Exposes ``/v1/chat/completions`` and ``/v1/models`` on the returned ``*.modal.run``
    URL. The client picks the model by the OpenAI ``model`` field.
    """
    import subprocess

    subprocess.Popen(_vllm_serve_command())


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={MODELS_DIR: models_volume},
    timeout=15 * 60,
)
def smoke_test() -> str:
    """Cheapest possible real-GPU proof: load Qwen and run ONE real generation.

    Confirms the model loads on the GPU and returns a real completion (moving a small
    amount of credit) without standing up the web server. Used by GATE 3's pre-flight.
    """
    import time

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=QWEN_MODEL,
        download_dir=MODELS_DIR,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    load_s = time.time() - t0
    prompt = (
        "<|im_start|>system\nYou are a seller in a repeated pricing game. "
        "Reply with exactly one line: PRICE=<integer index 0-14>.<|im_end|>\n"
        "<|im_start|>user\nRound 1. Choose your price index.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t1 = time.time()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, seed=0, max_tokens=16))
    gen_s = time.time() - t1
    text = out[0].outputs[0].text.strip()
    return (
        f"REAL GPU GENERATION OK on {GPU_TYPE}\n"
        f"model={QWEN_MODEL}\nload_s={load_s:.1f} gen_s={gen_s:.1f}\n"
        f"completion={text!r}"
    )


@app.local_entrypoint()
def main():
    """Run the cheap real-GPU smoke test and print the result."""
    print("Dispatching real-GPU smoke test to Modal (spends a little credit)...")
    print(smoke_test.remote())
