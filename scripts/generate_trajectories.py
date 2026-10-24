"""Phase 3 (design 11): the GATED real Modal sweep across Q1-Q4 x both model families.

Generates REAL agent trajectories with the Modal-hosted vLLM server and writes them to a
trajectory directory in the exact ``*.trajectories.jsonl`` format ``scripts/run_eval.py``
consumes. This is the expensive step and MUST only be run after GATE 3
(``scripts/positive_control.py --backend real --quadrant Q1``) exits 0.

Frugality (NFR-1, design 11): 3-5 seeds/quadrant by default, short outputs (small
``--max-tokens``), one warm Modal GPU. Q1/Q2/Q4 use the distinct Qwen+Llama families; Q3
uses the SAME served Qwen checkpoint for both agents (the quadrant table already encodes
this via ``agent_models``, so no special-casing is needed here -- the real backend's alias
map points both Q3 logical ids at the one served Qwen checkpoint).

GPU ACCESS NOTE: Modal GPU functions on this account require a payment method, so this
script is wired and deploy-ready but was not executed for the committed artifacts; see
RESULTS.md. Point it at the deployed endpoint with ``--modal-url`` / ``PFF_MODAL_VLLM_URL``.

Usage (after GATE 3 passes and a payment method is on the Modal account)::

    .venv/bin/python -m modal deploy modal_app/vllm_server.py
    PFF_MODAL_VLLM_URL=https://<ws>--plus-factor-forensics-vllm-serve.modal.run/v1 \
        .venv/bin/python scripts/generate_trajectories.py --seeds 1 2 3
    .venv/bin/python scripts/run_eval.py --traj-dir results/trajectories_real
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pff.envs.economics import Economics
from pff.logging.logger import write_trajectories
from pff.quadrants.config import QUADRANT_ORDER
from pff.quadrants.generator import generate_quadrant

WT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ_DIR = WT / "results" / "trajectories_real"


def _make_real_backend(modal_url: str | None):
    from pff.inference.modal_client import BackendError, ModalVLLMBackend

    url = modal_url or os.environ.get("PFF_MODAL_VLLM_URL")
    if not url:
        raise BackendError(
            "the real sweep requires --modal-url or PFF_MODAL_VLLM_URL (the deployed Modal "
            "vLLM endpoint ending in /v1)"
        )
    return ModalVLLMBackend(base_url=url)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gated real Modal sweep (Phase 3).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--modal-url", default=None)
    args = parser.parse_args()

    econ = Economics()
    try:
        backend = _make_real_backend(args.modal_url)
        served = backend.available_models()  # also fails fast if the endpoint is down
    except Exception as exc:  # noqa: BLE001 - connectivity/config is fatal before any spend
        print(f"ERROR: real backend unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"served models: {served}")

    args.traj_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for seed in args.seeds:
        trajs = []
        for q in QUADRANT_ORDER:
            # Q3 uses temperature 0 (greedy) per the determinism contract (design 5.4);
            # the other quadrants use the requested temperature for throughput.
            temp = 0.0 if q == "Q3" else args.temperature
            trajs.append(
                generate_quadrant(
                    q,
                    backend,
                    seed=seed,
                    n_rounds=args.n_rounds,
                    temperature=temp,
                    econ=econ,
                )
            )
        out = args.traj_dir / f"real_seed{seed}.trajectories.jsonl"
        write_trajectories(out, trajs)
        written.append(out.name)
        print(f"  wrote {out.name} (Q1-Q4)")

    print(f"real trajectories under {args.traj_dir}: {written}")
    print("next: scripts/run_eval.py --traj-dir", args.traj_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
