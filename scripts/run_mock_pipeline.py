"""GATE 2 -- full mock end-to-end pipeline (design 11, Phase 1). $0, no GPU, no network.

Phase 1 of the frugality plan:
  1. Generate all four quadrants (Q1-Q4) on the MOCK backend across the default seeds,
     writing one ``<quadrant>.trajectories.jsonl`` per quadrant under
     ``results/trajectories/``.
  2. Invoke ``run_eval`` end-to-end: score both detectors, write
     ``results/audit_table.csv`` and the five figures under ``results/figures/``.

GATE 2 PASSES iff all four quadrants generate, both detectors score, the audit table +
five figures are written, and the headline confusion contrast holds (naive flags Q3; the
auditor keeps Q1/Q2 recall and rejects Q3/Q4). Exits non-zero otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import write_trajectories
from pff.quadrants.config import QUADRANT_ORDER
from pff.quadrants.generator import generate_quadrant

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import run_eval  # noqa: E402

WT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [1, 2, 3]
TRAJ_DIR = WT / "results" / "trajectories"
RESULTS_DIR = WT / "results"
FIGURE_NAMES = (
    "fig_confusion.png",
    "fig_pf1_gap.png",
    "fig_influence_validation.png",
    "fig_price_paths.png",
    "fig_naive_vs_auditor_q3.png",
)


def generate_all(seeds, n_rounds, traj_dir: Path) -> dict[str, int]:
    """Generate every quadrant across seeds; return {quadrant: n_trajectories}."""
    econ = Economics()
    backend = MockBackend()
    traj_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for quadrant in QUADRANT_ORDER:
        trajs = [
            generate_quadrant(quadrant, backend, seed=s, n_rounds=n_rounds, econ=econ)
            for s in seeds
        ]
        write_trajectories(traj_dir / f"{quadrant}.trajectories.jsonl", trajs)
        counts[quadrant] = len(trajs)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="GATE 2 full mock pipeline.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--n-rounds", type=int, default=25)
    args = parser.parse_args()

    print(f"GATE 2 Phase 1: generating Q1-Q4 on the mock backend (seeds={args.seeds}).")
    counts = generate_all(args.seeds, args.n_rounds, TRAJ_DIR)
    for q in QUADRANT_ORDER:
        print(f"  {q}: {counts[q]} trajectories -> {TRAJ_DIR / (q + '.trajectories.jsonl')}")
    if not all(counts.get(q, 0) == len(args.seeds) for q in QUADRANT_ORDER):
        print("GATE 2 FAIL: not all four quadrants generated.", file=sys.stderr)
        return 1

    print("\nGATE 2 Phase 1 (eval): scoring both detectors + writing figures.")
    rc = run_eval(TRAJ_DIR, RESULTS_DIR, seed=0)
    if rc != 0:
        print("GATE 2 FAIL: evaluation did not produce the expected contrast.", file=sys.stderr)
        return rc

    # Confirm artifacts exist.
    table = RESULTS_DIR / "audit_table.csv"
    figs = [RESULTS_DIR / "figures" / n for n in FIGURE_NAMES]
    missing = [str(p) for p in [table, *figs] if not p.exists()]
    if missing:
        print(f"GATE 2 FAIL: missing artifacts: {missing}", file=sys.stderr)
        return 1

    print("\nGATE 2 PASS: four quadrants generated, both detectors scored, "
          "audit_table.csv + five figures written, headline contrast holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
