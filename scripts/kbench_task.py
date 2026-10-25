"""Kaggle Benchmarks task: Plus-Factor Forensics frontier-model collusion sweep.

Pushed and run via the CLI (credit + Model Proxy pre-wired on Kaggle infra):

    kaggle b t push pff-frontier-collusion-sweep -f scripts/kbench_task.py \
        -d rudranighosh/pff-0-1-0-tar-gz --wait
    kaggle b t run pff-frontier-collusion-sweep \
        -m gemini-2.5-pro -m claude-sonnet-4-6-default -m gpt-5.5-2026-04-23 \
        -m deepseek-r1-0528 --wait
    kaggle b t download pff-frontier-collusion-sweep -o artifacts/kaggle/frontier

Runs the 4-quadrant sweep with the injected frontier model as both agents, scores
both detectors, writes summary.json + audit_table.csv + figures to /kaggle/working.
Short on purpose (2 seeds x 15 rounds) to keep credit spend modest.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import kaggle_benchmarks as kbench


def _ensure_pff():
    """Install/locate the attached pff package (Kaggle auto-extracts the sdist)."""
    try:
        import pff  # noqa: F401
        return
    except Exception:
        pass
    cands = glob.glob("/kaggle/input/**/pff-0.1.0/pyproject.toml", recursive=True)
    if not cands:
        cands = glob.glob("/kaggle/input/**/src/pff/__init__.py", recursive=True)
        if cands:
            src_dir = os.path.dirname(os.path.dirname(cands[0]))
            sys.path.insert(0, src_dir)
            import pff  # noqa: F401
            return
        raise RuntimeError("pff package not found under /kaggle/input")
    pkg_dir = os.path.dirname(cands[0])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pydantic>=2", "scipy>=1.11", "matplotlib>=3.7"], check=False)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg_dir])
    sys.path.insert(0, os.path.join(pkg_dir, "src"))
    import pff  # noqa: F401


SEEDS = [1, 2]
N_ROUNDS = 15
TEMPERATURE = 0.7


def _model_tag(llm) -> str:
    for attr in ("model", "model_name", "name", "id"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return "".join(c if c.isalnum() else "-" for c in v)[:48]
    return "model-" + str(int(time.time()))


@kbench.task(name="pff-frontier-collusion-sweep")
def pff_sweep(llm):
    _ensure_pff()

    from pff.inference.kbench_backend import KbenchBackend
    from pff.quadrants.generator import generate_all_quadrants
    from pff.logging.logger import write_trajectories, read_trajectories
    from pff.eval.metrics import evaluate
    from pff.eval.figures import write_all_figures, write_audit_table
    from pff.envs.economics import Economics
    from pff.inference.mock_backend import MockBackend
    from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
    from pff.detectors.plusfactor.net_influence import net_pairwise_influence

    tag = _model_tag(llm)
    results = Path(f"/kaggle/working/results_frontier/{tag}")
    traj = results / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    backend = KbenchBackend(llm, model_label=tag)

    for seed in SEEDS:
        try:
            trajs = generate_all_quadrants(backend, seed=seed, n_rounds=N_ROUNDS,
                                           temperature=TEMPERATURE)
        except Exception as e:  # noqa: BLE001
            print(f"seed {seed} failed: {type(e).__name__}: {e}", flush=True)
            continue
        out = traj / f"{tag}_seed{seed}.trajectories.jsonl"
        write_trajectories(out, list(trajs.values()))
        print("wrote", out.name, flush=True)

    trajectories = []
    for f in sorted(traj.glob("*.trajectories.jsonl")):
        trajectories.extend(read_trajectories(f))
    kbench.assertions.assert_true(len(trajectories) > 0,
                                  expectation="sweep produced real trajectories")

    econ = Economics()
    result = evaluate(trajectories, backend=MockBackend(), seed=0, econ=econ)
    write_audit_table(result, results / "audit_table.csv")

    base = run_matrix_game(MatrixGameConfig(n_agents=4, seed=0))
    coord = base.coordinating_pair
    runs = [r for s in range(5)
            for r in [run_matrix_game(MatrixGameConfig(n_agents=4, seed=s))]
            if r.coordinating_pair == coord]
    scores = {}
    for (i, j) in itertools.combinations(range(4), 2):
        scores[(i, j)] = net_pairwise_influence(
            [r.back_half_series(i) for r in runs],
            [r.back_half_series(j) for r in runs],
            cardinality=CARDINALITY, seed=12345).net_pairwise
    write_all_figures(result, trajectories, figures_dir=results / "figures", econ=econ,
                      influence_pair_scores=scores, coordinating_pair=coord)

    summary = {"arm": "frontier-kbench", "model_tag": tag, "seeds": SEEDS,
               "n_rounds": N_ROUNDS, "n_trajectories": len(trajectories),
               "tau_naive": result.tau_naive, "pf1_threshold": result.pf1_threshold,
               "naive": {"precision": result.naive.precision, "recall": result.naive.recall,
                         "fpr": result.naive.fpr, "per_quadrant": result.naive.per_quadrant},
               "auditor": {"precision": result.auditor.precision, "recall": result.auditor.recall,
                           "fpr": result.auditor.fpr, "per_quadrant": result.auditor.per_quadrant}}
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SUMMARY", tag, json.dumps(summary, default=str), flush=True)


# The push/run harness executes this file; it must produce a run via .run().
# On push-validation kbench provides a default llm; on `tasks run -m <model>` the
# selected model is injected. This top-level call is what generates run output.
pff_sweep.run(llm=kbench.llm)
