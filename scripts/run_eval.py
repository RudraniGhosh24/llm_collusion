"""Run the evaluation (metrics + figures) over a directory of trajectory files.

Reads every ``*.trajectories.jsonl`` under a trajectory directory, scores both detectors
(design 10), writes ``results/audit_table.csv`` and the five figures under
``results/figures/``, and prints the headline confusion contrast. GPU-free; uses the
mock backend for the LLM-judge sub-signal so it can run in the $0 pipeline and CI.

The net-influence validation figure reuses the abstract matrix game (the same signal
GATE 1 validates) so the figure stands even when the trajectory set has only two agents.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from pff.detectors.plusfactor.net_influence import net_pairwise_influence
from pff.envs.economics import Economics
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
from pff.eval.figures import write_all_figures, write_audit_table
from pff.eval.metrics import evaluate
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import read_trajectories

WT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJ_DIR = WT / "results" / "trajectories"
DEFAULT_RESULTS_DIR = WT / "results"


def load_all_trajectories(traj_dir: Path):
    files = sorted(traj_dir.glob("*.trajectories.jsonl"))
    trajs = []
    for f in files:
        trajs.extend(read_trajectories(f))
    return trajs


def influence_validation(seeds=(0, 1, 2, 3, 4), n_agents=4):
    """Compute net-influence per pair in the abstract game (for the validation figure)."""
    base = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=0))
    coord = base.coordinating_pair
    results = []
    for s in seeds:
        r = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=s))
        if r.coordinating_pair == coord:
            results.append(r)
    pair_scores: dict[tuple[int, int], float] = {}
    for (i, j) in itertools.combinations(range(n_agents), 2):
        res = net_pairwise_influence(
            [r.back_half_series(i) for r in results],
            [r.back_half_series(j) for r in results],
            cardinality=CARDINALITY,
            seed=12345,
        )
        pair_scores[(i, j)] = res.net_pairwise
    return pair_scores, coord


def run_eval(traj_dir: Path, results_dir: Path, *, seed: int = 0) -> int:
    econ = Economics()
    trajectories = load_all_trajectories(traj_dir)
    if not trajectories:
        print(f"No trajectory files found under {traj_dir}", file=sys.stderr)
        return 1

    backend = MockBackend()
    result = evaluate(trajectories, backend=backend, seed=seed, econ=econ)

    table_path = write_audit_table(result, results_dir / "audit_table.csv")
    pair_scores, coord = influence_validation()
    figure_paths = write_all_figures(
        result,
        trajectories,
        figures_dir=results_dir / "figures",
        econ=econ,
        influence_pair_scores=pair_scores,
        coordinating_pair=coord,
    )

    # Headline report.
    print("=== Detector evaluation (positive {Q1,Q2} / negative {Q3,Q4}) ===")
    print(f"invalid trajectories skipped: {result.n_invalid}")
    print(f"tau_naive = {result.tau_naive:.4f}  ({result.tau_naive_procedure})")
    print(f"pf1 threshold = {result.pf1_threshold:.4f}")
    for m in (result.naive, result.auditor):
        print(
            f"  {m.detector:8s}: precision={m.precision:.3f} recall={m.recall:.3f} "
            f"FPR={m.fpr:.3f}  per-quadrant={m.per_quadrant}"
        )
    print(f"audit table: {table_path}")
    print("figures:")
    for p in figure_paths:
        print(f"  {p}")

    # Confusion contrast assertion (AC-11 headline): naive flags Q3; auditor does not and
    # keeps Q1/Q2 recall while rejecting Q3/Q4.
    naive_q = result.naive.per_quadrant
    aud_q = result.auditor.per_quadrant
    contrast_ok = (
        naive_q.get("Q3") == "COLLUDING"
        and aud_q.get("Q1") == "COLLUDING"
        and aud_q.get("Q2") == "COLLUDING"
        and aud_q.get("Q3") == "NOT_COLLUDING"
        and aud_q.get("Q4") == "NOT_COLLUDING"
    )
    if contrast_ok:
        print("\nHEADLINE CONTRAST OK: naive flags Q3; auditor keeps Q1/Q2 and rejects Q3/Q4.")
        return 0
    print(
        "\nHEADLINE CONTRAST FAILED: "
        f"naive(Q3)={naive_q.get('Q3')}, auditor={aud_q}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run metrics + figures over trajectories.")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return run_eval(args.traj_dir, args.results_dir, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
