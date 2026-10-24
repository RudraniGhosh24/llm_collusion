#!/usr/bin/env python3
"""Emit a single self-contained Kaggle job script for the real LLM sweep.

A Kaggle *script* kernel ships exactly one .py file, so we embed the whole ``pff``
package as a base64 zip blob (same pattern as the molecular project's bundle_job.py),
followed by a prologue that unpacks it, pip-installs vLLM, serves Qwen/Llama locally,
generates REAL trajectories, runs the real eval, and writes pff_results.zip to
/kaggle/working.

Usage:
    python scripts/build_kaggle_job.py --out artifacts/kaggle/pff_sweep_job.py
Then push with the molecular project's runner:
    <molecular>/.venv/bin/python <molecular>/scripts/kaggle_run.py run \
        --script artifacts/kaggle/pff_sweep_job.py --name pff-llm-sweep \
        --gpu --gpu-shape NvidiaTeslaT4 --enable-internet --wait \
        -o artifacts/kaggle/out
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import zipfile
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = WT / "src" / "pff"


def build_blob(package_dir: Path = PACKAGE_DIR) -> bytes:
    sources = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    if not sources:
        raise FileNotFoundError(f"no .py under {package_dir}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sources:
            arc = str(Path("pff") / path.relative_to(package_dir))
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return base64.b64encode(buf.getvalue())


PROLOGUE = '''# AUTO-GENERATED Kaggle job: Plus-Factor Forensics real LLM sweep. Do not edit by hand.
import base64, io, os, subprocess, sys, time, zipfile, json, itertools
from pathlib import Path

# 1) unpack embedded pff package to /tmp and put it on sys.path
_BUNDLE = Path("/tmp/pff_bundle")
_BUNDLE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_PFF_BLOB))) as zf:
    zf.extractall(_BUNDLE)
sys.path.insert(0, str(_BUNDLE))

# 2) install runtime deps. Kaggle's image is Python 3.13 with a recent torch, so we do
# NOT pin vllm (old pins drag in torchvision builds that don't exist for 3.13). Let pip
# pick a vllm compatible with the preinstalled torch; print the resolved version.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "openai>=1.40", "pydantic>=2", "scipy>=1.11", "matplotlib>=3.7"],
               check=True)
_vllm_ok = False
for _spec in ["vllm", "vllm<0.7", "vllm==0.6.3.post1"]:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", _spec])
    if r.returncode == 0:
        _vllm_ok = True
        print("installed vllm via spec:", _spec, flush=True)
        break
if not _vllm_ok:
    raise SystemExit("could not install a Python-3.13-compatible vllm on Kaggle")
import vllm as _v
print("resolved vllm", getattr(_v, "__version__", "?"), flush=True)

import requests  # Kaggle preinstalls

MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
PORT = 8000
SEEDS = [1, 2, 3, 4, 5]
N_ROUNDS = 25
TEMPERATURE = 0.7

RESULTS = Path("/kaggle/working/results"); TRAJ = RESULTS / "trajectories"
TRAJ.mkdir(parents=True, exist_ok=True)

def start_server(hf_model):
    p = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", hf_model, "--port", str(PORT),
         "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
         "--disable-log-requests"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            if requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).status_code == 200:
                print("server up:", hf_model, flush=True); return p
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError("vLLM did not start for " + hf_model)

def stop_server(p):
    p.terminate()
    try: p.wait(timeout=60)
    except Exception: p.kill()
    time.sleep(5)

from pff.inference.vllm_backend import VLLMBackend
from pff.quadrants.generator import generate_all_quadrants
from pff.logging.logger import write_trajectories, read_trajectories
from pff.eval.metrics import evaluate
from pff.eval.figures import write_all_figures, write_audit_table
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game
from pff.detectors.plusfactor.net_influence import net_pairwise_influence

def alias(served):
    return {"mock/Qwen2.5-7B-Instruct": served, "mock/Llama-3.1-8B-Instruct": served,
            "Qwen/Qwen2.5-7B-Instruct": served, "meta-llama/Llama-3.1-8B-Instruct": served}

generated = []
for key, hf in MODELS.items():
    try:
        proc = start_server(hf)
    except Exception as e:
        print("SKIP model", key, "->", e, flush=True)
        continue
    try:
        backend = VLLMBackend(base_url=f"http://localhost:{PORT}/v1", model_aliases=alias(hf))
        print("served:", backend.available_models(), flush=True)
        for seed in SEEDS:
            trajs = generate_all_quadrants(backend, seed=seed, n_rounds=N_ROUNDS, temperature=TEMPERATURE)
            out = TRAJ / f"{key}_seed{seed}.trajectories.jsonl"
            write_trajectories(out, list(trajs.values()))
            generated.append(out.name)
            print("  wrote", out.name, flush=True)
    finally:
        stop_server(proc)

print("generated files:", generated, flush=True)

# eval on whatever real trajectories we produced
trajectories = []
for f in sorted(TRAJ.glob("*.trajectories.jsonl")):
    trajectories.extend(read_trajectories(f))
assert trajectories, "no real trajectories generated"

econ = Economics()
result = evaluate(trajectories, backend=MockBackend(), seed=0, econ=econ)
write_audit_table(result, RESULTS / "audit_table.csv")

def influence_validation(seeds=(0,1,2,3,4), n_agents=4):
    base = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=0)); coord = base.coordinating_pair
    runs = [r for s in seeds for r in [run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=s))] if r.coordinating_pair == coord]
    sc = {}
    for (i,j) in itertools.combinations(range(n_agents), 2):
        sc[(i,j)] = net_pairwise_influence([r.back_half_series(i) for r in runs],
                                           [r.back_half_series(j) for r in runs],
                                           cardinality=CARDINALITY, seed=12345).net_pairwise
    return sc, coord
ps, coord = influence_validation()
write_all_figures(result, trajectories, figures_dir=RESULTS / "figures", econ=econ,
                  influence_pair_scores=ps, coordinating_pair=coord)

summary = {"n_trajectories": len(trajectories), "seeds": SEEDS, "n_rounds": N_ROUNDS,
           "models_attempted": MODELS, "files": generated,
           "tau_naive": result.tau_naive, "pf1_threshold": result.pf1_threshold,
           "naive": {"precision": result.naive.precision, "recall": result.naive.recall,
                     "fpr": result.naive.fpr, "per_quadrant": result.naive.per_quadrant},
           "auditor": {"precision": result.auditor.precision, "recall": result.auditor.recall,
                       "fpr": result.auditor.fpr, "per_quadrant": result.auditor.per_quadrant},
           "n_invalid": result.n_invalid}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("SUMMARY", json.dumps(summary, indent=2, default=str), flush=True)

import shutil
shutil.make_archive("/kaggle/working/pff_results", "zip", RESULTS)
print("DONE -> /kaggle/working/pff_results.zip", flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blob = build_blob()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"_PFF_BLOB = {blob!r}\n\n")
        fh.write(PROLOGUE)
    size = out.stat().st_size
    print(f"wrote {out} ({size/1024:.0f} KB, blob {len(blob)/1024:.0f} KB)")
    if size > 900 * 1024:
        print("WARNING: job > 900KB; Kaggle may reject. Consider a dataset.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
