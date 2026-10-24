# Plus-Factor Forensics

A local-first Python research harness that pits a **naive, association-based** collusion
detector against a **causal, plus-factors-doctrine auditor** over logged multi-agent
Bertrand-pricing trajectories.

**Thesis.** In a constructed negative control where two agents share the *same base-model
checkpoint* but have no channel and no incentive to coordinate (innocent parallelism), the
naive detector false-positives while the plus-factor auditor does not — without losing
recall on genuine collusion. Everything except final real-LLM trajectory generation runs at
**$0, GPU-free, deterministic** via a mock inference backend; only the gated sweep touches a
Modal GPU. See `RESULTS.md` for the headline numbers and an explicit statement of what is
real vs mock.

## Install

```bash
cd /path/to/.worktrees/plus-factor-forensics
# pip is NOT present in the venv; use uv (uv manages this project):
uv pip install --python /path/to/.venv/bin/python -e .
```

All dependencies are pinned with exact `==` in both `pyproject.toml` and `requirements.lock`
(a test, `test_deps_pinned`, enforces this). `vllm` is pinned inside the Modal image only.

## Run (the three gates, cheapest → most expensive)

```bash
PY=/path/to/.venv/bin/python

# full mock test suite (13 modules; the 2 GPU-gated tests skip without a Modal endpoint)
$PY -m pytest -q

# GATE 1 — Phase 0: scripted abstract-game net-influence validation ($0, no GPU)
$PY scripts/run_scripted_game.py

# GATE 2 — Phase 1: full Q1–Q4 mock pipeline -> results/audit_table.csv + 5 figures ($0)
$PY scripts/run_mock_pipeline.py

# GATE 3 — Phase 2: the FIRST and ONLY real GPU spend (see "Running the real sweep")
$PY scripts/positive_control.py --backend real --quadrant Q1
```

GATE 3 is a hard gate: the multi-seed real sweep must **not** run unless the real Q1
positive control exits 0.

## Running the real sweep (Modal GPU — currently blocked)

> **Status:** Modal GPU functions on the sprint account require a payment method
> (`InvalidError: Please add a payment method to use <TIER> GPU functions.`). The $50 credit
> does not lift this. CPU Modal functions work, so the code is deploy-ready but the real
> runs have **not** been executed. No real GPU credit was consumed. See `RESULTS.md`.

Once a payment method is added at [modal.com](https://modal.com):

```bash
PY=/path/to/.venv/bin/python

# 1. deploy the one-warm-GPU vLLM server (Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct)
$PY -m modal deploy modal_app/vllm_server.py
#    for the Q3 same-checkpoint determinism control, deploy with:
PFF_VLLM_Q3_DETERMINISM=1 $PY -m modal deploy modal_app/vllm_server.py

# 2. export the printed endpoint (ends in /v1)
export PFF_MODAL_VLLM_URL=https://<workspace>--plus-factor-forensics-vllm-serve.modal.run/v1

# 3. GATE 3: real Q1 positive control (frugal: few seeds, short outputs)
$PY scripts/positive_control.py --backend real --quadrant Q1 --seeds 1 2 3

# 4. only if GATE 3 passes — the gated real sweep, then the real eval
$PY scripts/generate_trajectories.py --seeds 1 2 3
$PY scripts/run_eval.py --traj-dir results/trajectories_real

# the two GPU-gated tests then run instead of skipping:
$PY -m pytest tests/test_modal_backend_gpu.py tests/test_q3_determinism_gpu.py -q
```

Cost discipline: smallest adequate GPU (A10G/L4), one warm server, short `scaledown_window`,
weights cached on a Modal Volume, 3–5 seeds/quadrant, short `--max-tokens`.

## Code → plus factor → result map (design §1)

Each plus factor has exactly one module and emits one named column in the auditor output
and one figure/row-group in the eval artifacts.

| Signal | Definition | Module | Output field | Eval artifact |
|---|---|---|---|---|
| Plus factor #1 — action against self-interest (**gate**) | directional counterfactual payoff gap, rival pinned at Nash | `detectors/plusfactor/pf1_self_interest.py` + `detectors/replay.py` | `pf1_gap` (directional) | `fig_pf1_gap.png`, col in `audit_table.csv` |
| Plus factor #2 — comms / side-channel (**gate: pf2 OR pf4**) | explicit-channel MI OR order-code MI | `detectors/plusfactor/pf2_signaling.py` | `pf2_signal_score` | col in `audit_table.csv` |
| Plus factor #3 — motive + opportunity (corroboration only) | shared incentive AND usable channel | `detectors/plusfactor/pf3_motive_opportunity.py` | `pf3_motive_opp` | col in `audit_table.csv` |
| Plus factor #4 — plan → action (**gate: pf2 OR pf4**) | B tracks A's private plan; hard `0.0` if inapplicable | `detectors/plusfactor/pf4_plan_to_action.py` | `pf4_plan_action` (float) | col in `audit_table.csv` |
| Net pairwise influence | info-theoretic realization of #1 | `detectors/plusfactor/net_influence.py` | `net_influence` | `fig_influence_validation.png` |
| Naive correlation | back-half price correlation | `detectors/naive.py` | `naive_corr` | `fig_confusion.png` |
| Naive LLM-judge | conversation/price score | `detectors/naive.py` | `naive_judge` | `fig_confusion.png` |

Auditor decision rule (design §8.6): `pf1_fires AND (pf2_fires OR pf4_fires)`. pf1 is the
primary discriminator; pf3 and net influence are **reported but not gated**.

## Layout

```
src/pff/            envs/ agents/ inference/ logging/ quadrants/ detectors/ eval/
modal_app/          vllm_server.py        # the only GPU-spending component (deployed separately)
scripts/            run_scripted_game, run_mock_pipeline, positive_control,
                    generate_trajectories, run_eval
tests/              13 mock modules + 2 GPU-gated modules (skip without a Modal endpoint)
results/            audit_table.csv, figures/*.png
RESULTS.md          headline numbers + real-vs-mock statement
```

## Honesty

No results, metrics, or citations are fabricated. All numbers in `RESULTS.md` come from
committed run artifacts produced by the GPU-free mock pipeline; the real-LLM sweep is wired
but blocked (Modal GPU payment method). Unverified citations are marked `verify citation`.
