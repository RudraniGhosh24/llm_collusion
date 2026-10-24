# ======================================================================
# Plus-Factor Forensics — REAL LLM sweep on Kaggle free GPU (T4)
# ======================================================================
# WHAT THIS DOES
#   Generates REAL agent trajectories with real Qwen2.5-7B-Instruct and
#   Llama-3.1-8B-Instruct served locally by vLLM on Kaggle's free T4, using the
#   exact tested pff code, then runs the real eval (both detectors + 5 figures +
#   headline confusion contrast). Produces pff_results.zip for you to send back.
#
# HOW TO RUN (browser, no credit card)
#   1. kaggle.com -> Create -> New Notebook.
#   2. Settings -> Accelerator -> "GPU T4 x2".   Settings -> Internet -> ON.
#   3. "+ Add Input" -> Upload dataset -> upload  pff-0.1.0.tar.gz
#      (it will appear under /kaggle/input/<name>/pff-0.1.0.tar.gz).
#   4. Paste each CELL below into its own cell, in order.  Run All.
#   5. Download /kaggle/working/pff_results.zip (Output tab) and send it to me.
#
# TIME: ~25-45 min (model downloads dominate). To go faster, reduce SEEDS in CELL 3.
# ======================================================================


# ---- CELL 1: install vLLM + our tested package -----------------------
import glob, subprocess, sys

cands = glob.glob("/kaggle/input/**/pff-0.1.0.tar.gz", recursive=True)
assert cands, "Upload pff-0.1.0.tar.gz as a notebook input (+ Add Input) first."
PFF_SDIST = cands[0]
print("found package:", PFF_SDIST)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "vllm==0.6.6", "openai>=1.40"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", PFF_SDIST], check=True)
print("installs done")


# ---- CELL 2: vLLM server start/stop helpers --------------------------
import time, requests

MODELS = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
PORT = 8000

def start_server(hf_model: str):
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", hf_model, "--port", str(PORT),
         "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
         "--disable-log-requests"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{PORT}/v1/models"
    for _ in range(150):               # up to ~25 min for first-time model download
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print("server up:", hf_model); return proc
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError(f"vLLM server for {hf_model} did not start in time")

def stop_server(proc):
    proc.terminate()
    try: proc.wait(timeout=60)
    except Exception: proc.kill()
    time.sleep(5)


# ---- CELL 3: generate REAL trajectories (uses tested pff code) -------
# For each model we generate all four quadrants per seed and WRITE them with the
# real logger to results/trajectories/<model>_seed<seed>.trajectories.jsonl, the exact
# format scripts/run_eval.py expects. Q3 (innocent-parallelism control) uses the SAME
# served checkpoint for both agents with identical decode params, as the table requires.
from pathlib import Path
from pff.inference.vllm_backend import VLLMBackend
from pff.quadrants.generator import generate_all_quadrants
from pff.logging.logger import write_trajectories

SEEDS = [1, 2, 3, 4, 5]        # raise to 10-20 if the kernel has time left
N_ROUNDS = 25
TEMPERATURE = 0.7

RESULTS = Path("/kaggle/working/results")
TRAJ_DIR = RESULTS / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

def alias_map(served_hf_model: str) -> dict:
    # The quadrant table uses logical ids ("mock/Qwen...", "mock/Llama...").
    # Point them all at the currently-served real model so each pass is real inference.
    return {
        "mock/Qwen2.5-7B-Instruct": served_hf_model,
        "mock/Llama-3.1-8B-Instruct": served_hf_model,
        "Qwen/Qwen2.5-7B-Instruct": served_hf_model,
        "meta-llama/Llama-3.1-8B-Instruct": served_hf_model,
    }

for model_key, hf_model in MODELS.items():
    proc = start_server(hf_model)
    try:
        backend = VLLMBackend(base_url=f"http://localhost:{PORT}/v1",
                              model_aliases=alias_map(hf_model))
        print("served:", backend.available_models())
        for seed in SEEDS:
            trajs = generate_all_quadrants(
                backend, seed=seed, n_rounds=N_ROUNDS, temperature=TEMPERATURE
            )
            out = TRAJ_DIR / f"{model_key}_seed{seed}.trajectories.jsonl"
            write_trajectories(out, list(trajs.values()))
            print(f"  wrote {out.name}  (Q1-Q4)")
    finally:
        stop_server(proc)

print("trajectory files:", sorted(p.name for p in TRAJ_DIR.glob('*.jsonl')))


# ---- CELL 4: run the REAL eval (both detectors + figures + contrast) -
# This calls the exact tested eval path from scripts/run_eval.py. The naive detector's
# LLM-judge sub-signal uses the MockBackend judge (as in the tested pipeline); all agent
# BEHAVIOR being scored is REAL LLM output generated in CELL 3.
from pff.eval.metrics import evaluate
from pff.eval.figures import write_all_figures, write_audit_table
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import read_trajectories
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
from pff.detectors.plusfactor.net_influence import net_pairwise_influence
import itertools, json

trajectories = []
for f in sorted(TRAJ_DIR.glob("*.trajectories.jsonl")):
    trajectories.extend(read_trajectories(f))
print("loaded trajectories:", len(trajectories))

econ = Economics()
result = evaluate(trajectories, backend=MockBackend(), seed=0, econ=econ)

write_audit_table(result, RESULTS / "audit_table.csv")

# net-influence validation figure (reuses the abstract game, same as run_eval.py)
def influence_validation(seeds=(0,1,2,3,4), n_agents=4):
    base = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=0))
    coord = base.coordinating_pair
    runs = [r for s in seeds for r in [run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=s))]
            if r.coordinating_pair == coord]
    scores = {}
    for (i, j) in itertools.combinations(range(n_agents), 2):
        scores[(i, j)] = net_pairwise_influence(
            [r.back_half_series(i) for r in runs],
            [r.back_half_series(j) for r in runs],
            cardinality=CARDINALITY, seed=12345).net_pairwise
    return scores, coord

pair_scores, coord = influence_validation()
figs = write_all_figures(result, trajectories, figures_dir=RESULTS / "figures",
                         econ=econ, influence_pair_scores=pair_scores,
                         coordinating_pair=coord)

summary = {
    "n_trajectories": len(trajectories),
    "seeds": SEEDS, "n_rounds": N_ROUNDS, "models": MODELS,
    "tau_naive": result.tau_naive, "pf1_threshold": result.pf1_threshold,
    "naive": {"precision": result.naive.precision, "recall": result.naive.recall,
              "fpr": result.naive.fpr, "per_quadrant": result.naive.per_quadrant},
    "auditor": {"precision": result.auditor.precision, "recall": result.auditor.recall,
                "fpr": result.auditor.fpr, "per_quadrant": result.auditor.per_quadrant},
    "n_invalid": result.n_invalid,
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))


# ---- CELL 5: zip for download ----------------------------------------
import shutil
shutil.make_archive("/kaggle/working/pff_results", "zip", RESULTS)
print("DONE -> download /kaggle/working/pff_results.zip from the Output tab")
