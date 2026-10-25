#!/usr/bin/env python3
"""Frontier-model generalization arm: run the four-quadrant sweep with Gemini agents.

Gemini is API-based, so this runs from the laptop (no GPU/Kaggle needed). Requires
GEMINI_API_KEY in the environment. Writes real trajectories + a Gemini summary/figures
under results_gemini/.

Usage:
    GEMINI_API_KEY=... .venv/bin/python scripts/run_gemini_sweep.py \
        --model gemini-2.0-flash --seeds 1 2 3 --out results_gemini
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from pff.detectors.plusfactor.net_influence import net_pairwise_influence
from pff.envs.economics import Economics
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
from pff.eval.figures import write_all_figures, write_audit_table
from pff.eval.metrics import evaluate
from pff.inference.gemini_backend import GeminiBackend
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import read_trajectories, write_trajectories
from pff.quadrants.generator import generate_all_quadrants


def _alias(model: str) -> dict:
    return {
        "mock/Qwen2.5-7B-Instruct": model,
        "mock/Llama-3.1-8B-Instruct": model,
        "Qwen/Qwen2.5-7B-Instruct": model,
        "meta-llama/Llama-3.1-8B-Instruct": model,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-rounds", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results_gemini")
    args = ap.parse_args()

    results = Path(args.out)
    traj_dir = results / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    backend = GeminiBackend(model_name=args.model, model_aliases=_alias(args.model))
    print("Gemini model:", backend.available_models())

    generated = []
    for seed in args.seeds:
        trajs = generate_all_quadrants(
            backend, seed=seed, n_rounds=args.n_rounds, temperature=args.temperature
        )
        out = traj_dir / f"gemini_seed{seed}.trajectories.jsonl"
        write_trajectories(out, list(trajs.values()))
        generated.append(out.name)
        print("  wrote", out.name)

    trajectories = []
    for f in sorted(traj_dir.glob("*.trajectories.jsonl")):
        trajectories.extend(read_trajectories(f))

    econ = Economics()
    result = evaluate(trajectories, backend=MockBackend(), seed=0, econ=econ)
    write_audit_table(result, results / "audit_table.csv")

    base = run_matrix_game(MatrixGameConfig(n_agents=4, seed=0))
    coord = base.coordinating_pair
    runs = [
        r
        for s in range(5)
        for r in [run_matrix_game(MatrixGameConfig(n_agents=4, seed=s))]
        if r.coordinating_pair == coord
    ]
    scores = {}
    for (i, j) in itertools.combinations(range(4), 2):
        scores[(i, j)] = net_pairwise_influence(
            [r.back_half_series(i) for r in runs],
            [r.back_half_series(j) for r in runs],
            cardinality=CARDINALITY,
            seed=12345,
        ).net_pairwise
    write_all_figures(
        result, trajectories, figures_dir=results / "figures", econ=econ,
        influence_pair_scores=scores, coordinating_pair=coord,
    )

    summary = {
        "arm": "gemini", "model": args.model, "seeds": args.seeds,
        "n_rounds": args.n_rounds, "n_trajectories": len(trajectories),
        "tau_naive": result.tau_naive, "pf1_threshold": result.pf1_threshold,
        "naive": {"precision": result.naive.precision, "recall": result.naive.recall,
                  "fpr": result.naive.fpr, "per_quadrant": result.naive.per_quadrant},
        "auditor": {"precision": result.auditor.precision, "recall": result.auditor.recall,
                    "fpr": result.auditor.fpr, "per_quadrant": result.auditor.per_quadrant},
        "n_invalid": result.n_invalid,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
