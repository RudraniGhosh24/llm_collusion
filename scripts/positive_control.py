"""Positive control (FR-12, design 10 / 11 Phase 2+3): does Q1 price supracompetitively?

Runs a quadrant (``--quadrant``, default Q1) on the chosen backend (``--backend {mock,real}``,
default mock), pools the back-half rounds across seeds, and checks a PINNED pass rule. This
is GATE 3: it gates the expensive multi-seed sweep -- if Q1 does not demonstrate sustained
supracompetitive pricing on the REAL backend, the sweep must not run, so this script exits
non-zero.

The ``--backend real`` path talks to the Modal-hosted vLLM server
(:mod:`pff.inference.modal_client`); point it at the deployed endpoint with
``--modal-url`` or the ``PFF_MODAL_VLLM_URL`` env var. It is the first and only real GPU
spend before the sweep (frugality plan, design 11 Phase 2).

PINNED PASS RULE (deferred finding F4 -- concrete m and f, documented here):
    PASS iff the mean posted back-half price exceeds ``p_nash`` by at least
    ``m = 0.10`` price units in at least ``f = 0.60`` (60%) of pooled back-half
    agent-rounds.

Rationale: ``m = 0.10`` is comfortably below a single grid step (~0.107 on the pinned
15-point grid on [1.0, 2.5]), so one supracompetitive tick clears it, while ``f = 0.60``
requires the elevation to be *sustained* over the back half rather than a transient
spike. On the deterministic mock Q1 (idx cycling 6-9, all above Nash idx 5) every
back-half round clears ``m`` (fraction = 1.00), so the control passes; a competitive Q4
run sitting at Nash fails (fraction = 0.00).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.quadrants.generator import generate_quadrant

# PINNED pass rule (F4).
MARGIN_M = 0.10  # price units above p_nash
FRACTION_F = 0.60  # fraction of pooled back-half agent-rounds that must clear m


def back_half_margins(trajs, econ: Economics) -> np.ndarray:
    """Pooled (price - p_nash) over all agents in all back-half rounds of all seeds."""
    out: list[float] = []
    for t in trajs:
        n = len(t.steps)
        for r in range(n // 2, n):
            for a in t.steps[r].actions.values():
                out.append(a.price - econ.p_nash)
    return np.asarray(out, dtype=float)


def positive_control_passes(trajs, econ: Economics) -> tuple[bool, float, float]:
    """Return (passed, mean_margin, fraction_clearing_m) for the pinned rule."""
    margins = back_half_margins(trajs, econ)
    if margins.size == 0:
        return False, 0.0, 0.0
    frac = float(np.mean(margins >= MARGIN_M))
    mean_margin = float(np.mean(margins))
    return (frac >= FRACTION_F), mean_margin, frac


def _make_backend(backend_name: str, modal_url: str | None):
    """Return the inference backend; ``real`` requires a Modal vLLM endpoint."""
    if backend_name == "mock":
        return MockBackend()
    # real
    from pff.inference.modal_client import BackendError, ModalVLLMBackend

    url = modal_url or os.environ.get("PFF_MODAL_VLLM_URL")
    if not url:
        raise BackendError(
            "real backend requires --modal-url or PFF_MODAL_VLLM_URL (the deployed Modal "
            "vLLM endpoint, e.g. https://<ws>--plus-factor-forensics-vllm-serve.modal.run/v1)"
        )
    return ModalVLLMBackend(base_url=url)


def main() -> int:
    parser = argparse.ArgumentParser(description="GATE 3: quadrant positive control (FR-12).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3], help="seeds to pool"
    )
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument("--quadrant", default="Q1", help="quadrant to check (default Q1)")
    parser.add_argument(
        "--backend",
        choices=("mock", "real"),
        default="mock",
        help="inference backend; 'real' = Modal vLLM (GATE 3, GPU spend).",
    )
    parser.add_argument(
        "--modal-url",
        default=None,
        help="Modal vLLM OpenAI endpoint (.../v1); or set PFF_MODAL_VLLM_URL.",
    )
    args = parser.parse_args()

    econ = Economics()
    try:
        backend = _make_backend(args.backend, args.modal_url)
    except Exception as exc:  # noqa: BLE001 - config/connectivity failure is fatal to the gate
        print(f"ERROR: could not build the {args.backend} backend: {exc}", file=sys.stderr)
        return 2

    trajs = [
        generate_quadrant(args.quadrant, backend, seed=s, n_rounds=args.n_rounds, econ=econ)
        for s in args.seeds
    ]

    passed, mean_margin, frac = positive_control_passes(trajs, econ)

    print(f"Positive control ({args.quadrant}, seeds={args.seeds}, backend={args.backend}):")
    print(f"  p_nash = {econ.p_nash:.4f}")
    print(f"  pinned rule: margin m = {MARGIN_M} price units, fraction f = {FRACTION_F}")
    print(f"  mean back-half margin above Nash = {mean_margin:.4f}")
    print(f"  fraction of back-half agent-rounds clearing m = {frac:.4f}")

    if passed:
        print(
            f"POSITIVE CONTROL PASS: {args.quadrant} prices are durably supracompetitive."
        )
        return 0
    print(
        f"POSITIVE CONTROL FAIL: {args.quadrant} did not sustain supracompetitive pricing; "
        "the sweep is gated off.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
