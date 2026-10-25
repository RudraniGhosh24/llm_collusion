# ======================================================================
# Plus-Factor Forensics — FRONTIER-MODEL arm via Kaggle Benchmarks
# Runs the 4-quadrant collusion sweep on SOTA models (Gemini, Claude, ChatGPT,
# DeepSeek) through Kaggle Benchmarks' injected LLM, spending the benchmark AI credit.
# ======================================================================
# JOB IS KEPT SHORT ON PURPOSE: 2 seeds x 4 quadrants x ~15 rounds x 2 agents
# ≈ 240 short calls per model. Across 4 models that's ~1k calls — cheap on the
# $10/day benchmark credit while still giving a real per-model confusion result.
#
# HOW TO RUN
#   1. kaggle.com -> New Notebook. Internet ON. (No GPU needed.)
#   2. "+ Add Input" -> Upload -> pff-0.1.0.tar.gz.
#   3. Paste the cells. Run All -> creates a Benchmarks task + runs the default model.
#   4. On the task page click "Evaluate More Models" and select:
#        Gemini, Claude, ChatGPT (OpenAI), DeepSeek
#      (pick whichever Kaggle offers; the code is model-agnostic). Each selection
#      re-runs the sweep on that model and spends the benchmark credit.
#   5. Each run writes results_frontier/<run>/ ; download pff_frontier_results.zip
#      after each model, or re-zip at the end, and send them back.
# ======================================================================

# ---- CELL 1: install -------------------------------------------------
import glob, subprocess, sys
cands = glob.glob("/kaggle/input/**/pff-0.1.0.tar.gz", recursive=True)
assert cands, "Upload pff-0.1.0.tar.gz as a notebook input first."
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "kaggle-benchmarks", cands[0], "pydantic>=2", "scipy>=1.11",
                "matplotlib>=3.7"], check=True)
print("installs done")

# ---- CELL 2: define the benchmark task that runs our sweep -----------
import json, itertools, os, time
from pathlib import Path
import kaggle_benchmarks as kbench

from pff.inference.kbench_backend import KbenchBackend
from pff.quadrants.generator import generate_all_quadrants
from pff.logging.logger import write_trajectories, read_trajectories
from pff.eval.metrics import evaluate
from pff.eval.figures import write_all_figures, write_audit_table
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
from pff.detectors.plusfactor.net_influence import net_pairwise_influence

# SHORT job: keep frontier credit spend modest.
SEEDS = [1, 2]
N_ROUNDS = 15
TEMPERATURE = 0.7

def _model_tag(llm) -> str:
    # Best-effort label of the model currently injected, so multi-model runs don't
    # overwrite each other. Falls back to a timestamp if the attribute isn't exposed.
    for attr in ("model", "model_name", "name", "id"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return "".join(c if c.isalnum() else "-" for c in v)[:40]
    return "model-" + str(int(time.time()))

@kbench.task(name="pff-frontier-collusion-sweep")
def pff_sweep(llm):
    tag = _model_tag(llm)
    results = Path(f"/kaggle/working/results_frontier/{tag}")
    traj = results / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    backend = KbenchBackend(llm, model_label=tag)

    generated = []
    for seed in SEEDS:
        try:
            trajs = generate_all_quadrants(backend, seed=seed, n_rounds=N_ROUNDS,
                                           temperature=TEMPERATURE)
        except Exception as e:
            print(f"seed {seed} failed: {type(e).__name__}: {e}", flush=True)
            continue
        out = traj / f"{tag}_seed{seed}.trajectories.jsonl"
        write_trajectories(out, list(trajs.values()))
        generated.append(out.name)
        print("wrote", out.name, flush=True)

    trajectories = []
    for f in sorted(traj.glob("*.trajectories.jsonl")):
        trajectories.extend(read_trajectories(f))
    if not trajectories:
        kbench.assertions.assert_true(False, expectation="sweep produced trajectories")
        return

    econ = Economics()
    result = evaluate(trajectories, backend=MockBackend(), seed=0, econ=econ)
    write_audit_table(result, results / "audit_table.csv")

    base = run_matrix_game(MatrixGameConfig(n_agents=4, seed=0)); coord = base.coordinating_pair
    runs = [r for s in range(5) for r in [run_matrix_game(MatrixGameConfig(n_agents=4, seed=s))]
            if r.coordinating_pair == coord]
    scores = {}
    for (i, j) in itertools.combinations(range(4), 2):
        scores[(i, j)] = net_pairwise_influence(
            [r.back_half_series(i) for r in runs], [r.back_half_series(j) for r in runs],
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
    print("SUMMARY", tag, json.dumps(summary, indent=2, default=str), flush=True)
    kbench.assertions.assert_true(len(trajectories) > 0,
                                  expectation="sweep produced trajectories")

# ---- CELL 3: run default model, then zip whatever exists so far ------
pff_sweep.run(llm=kbench.llm)

import shutil
root = Path("/kaggle/working/results_frontier")
if root.exists():
    shutil.make_archive("/kaggle/working/pff_frontier_results", "zip", root)
print("DONE. Now use 'Evaluate More Models' on the task page to run "
      "Gemini / Claude / ChatGPT / DeepSeek; re-download pff_frontier_results.zip "
      "after the models finish.")
