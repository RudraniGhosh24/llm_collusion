"""Positive control (FR-12, design 10 / 11 Phase 2): does Q1 price supracompetitively?

Runs the Q1 quadrant (mock by default; ``--real`` would use the Modal backend in
FEAT-005's Phase 2), pools the back-half rounds across seeds, and checks a PINNED pass
rule. This gates the expensive multi-seed sweep: if Q1 does not demonstrate sustained
supracompetitive pricing, the sweep must not run, so this script exits non-zero.

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Q1 positive control (FR-12).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3], help="seeds to pool"
    )
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real Modal backend (FEAT-005 Phase 2); default is the mock.",
    )
    args = parser.parse_args()

    econ = Economics()
    if args.real:
        # Deferred to FEAT-005 (Phase 2 real GPU spend). Keep the gate honest here.
        print(
            "ERROR: --real backend wiring is owned by FEAT-005; not available in the "
            "mock pipeline.",
            file=sys.stderr,
        )
        return 2

    backend = MockBackend()
    trajs = [
        generate_quadrant("Q1", backend, seed=s, n_rounds=args.n_rounds, econ=econ)
        for s in args.seeds
    ]

    passed, mean_margin, frac = positive_control_passes(trajs, econ)

    print(f"Positive control (Q1, seeds={args.seeds}, mock):")
    print(f"  p_nash = {econ.p_nash:.4f}")
    print(f"  pinned rule: margin m = {MARGIN_M} price units, fraction f = {FRACTION_F}")
    print(f"  mean back-half margin above Nash = {mean_margin:.4f}")
    print(f"  fraction of back-half agent-rounds clearing m = {frac:.4f}")

    if passed:
        print("POSITIVE CONTROL PASS: Q1 prices are durably supracompetitive.")
        return 0
    print(
        "POSITIVE CONTROL FAIL: Q1 did not sustain supracompetitive pricing; "
        "the sweep is gated off.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
