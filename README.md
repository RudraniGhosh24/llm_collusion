# Correlation Is Not Collusion: A Plus-Factor Auditor for AI Agents

A local-first Python research harness that pits a **naive, correlation-based** collusion
detector against a **plus-factor auditor** grounded in antitrust doctrine, over logged
multi-agent Bertrand-pricing trajectories.

**Thesis.** Two agents built on the same model often behave alike for an innocent reason:
shared training leads them to the same choices, with no agreement between them. A detector
that treats similarity as collusion flags this innocent parallelism as a conspiracy.
Antitrust law resolved the same problem long ago: parallel conduct alone is not unlawful,
and inferring an agreement requires extra circumstantial evidence known as *plus factors*.
We turn that doctrine into a black-box auditor that scores trajectories for the signatures
of an agreement (acting against self-interest, signalling, plan-to-action concordance)
rather than for correlation.

**Headline result (real runs, six models).** Across Qwen2.5-7B, Llama-3.1-8B, GPT-5.5,
Gemini-3.8-flash, Gemini-2.5-pro, and Claude Sonnet 4.6, a correlation-and-judge detector
false-positives on innocent parallelism, and on the two most capable models (Gemini-2.5-pro,
Claude Sonnet 4.6) it flags every condition, including fully independent agents, as
collusion. The plus-factor auditor has a false-positive rate of **zero on all six models**.
Its recall is specificity-first and calibration-sensitive (see the paper). Full numbers in
`RESULTS.md` and the paper (`PAPER.pdf`).

## The four labeled conditions

Game, economics, and seed are held fixed; only the coordination mechanism varies.

- **Q1** collusion with an explicit chat channel and an incentive to keep joint prices high
- **Q2** collusion with no chat channel, only an action-ordering side channel
- **Q3** the negative control: one model instance plays both sellers, no channel, no
  incentive (innocent parallelism, the computational analogue of lawful conscious parallelism)
- **Q4** fully independent one-shot best-responders

## Install

```bash
cd /path/to/llm_collusion
# the venv is uv-managed and has no pip; install with uv:
uv pip install --python /path/to/.venv/bin/python -e .
```

Dependencies are pinned with exact `==` in `pyproject.toml` and `requirements.lock`
(enforced by `test_deps_pinned`).

## Run (GPU-free)

```bash
PY=/path/to/.venv/bin/python

# full test suite (GPU-gated tests skip without an endpoint)
$PY -m pytest -q

# scripted abstract-game net-influence validation ($0, no GPU)
$PY scripts/run_scripted_game.py

# full Q1-Q4 mock pipeline -> results/audit_table.csv + figures ($0, no GPU)
$PY scripts/run_mock_pipeline.py
```

The mock backend reproduces the full pipeline deterministically at $0 so the method can be
exercised without a GPU. Real trajectories are generated against a real inference backend
(see below).

## Real runs

The inference layer is backend-agnostic (`src/pff/inference/interface.py`). Any backend
implementing it drops into the env, generator, and detectors unchanged. Backends provided:

- `mock_backend.py` — deterministic, GPU-free, for testing and CI.
- `hf_backend.py` — HuggingFace Transformers, used to self-host Qwen2.5-7B and
  Llama-3.1-8B on a Kaggle T4 GPU (self-hosting is required for the Q3 same-checkpoint control).
- `vllm_backend.py` — talks to a vLLM OpenAI-compatible server.
- `kbench_backend.py` — Kaggle Benchmarks model proxy, used for the frontier models
  (GPT-5.5, Gemini-3.8-flash, Gemini-2.5-pro, Claude Sonnet 4.6).

The six-model results in `RESULTS.md` and the paper were produced this way: open-weight
models self-hosted on a free Kaggle T4 (5 seeds/condition), frontier models through the
Kaggle Benchmarks proxy (2 seeds/condition). The open-weight sweep driver is
`scripts/build_kaggle_job.py`; the frontier task is `scripts/kbench_task.py`.

## Code to plus-factor to result map

Auditor decision rule: `pf1_fires AND (pf2_fires OR pf4_fires)`. pf1 is the primary
discriminator; pf3 and net influence are reported but not gated.

| Signal | Definition | Module |
|---|---|---|
| pf1 action against self-interest (gate) | directional counterfactual payoff gap, rival pinned at Nash | `detectors/plusfactor/pf1_self_interest.py`, `detectors/replay.py` |
| pf2 signalling (gate: pf2 OR pf4) | explicit-channel or order-code mutual information | `detectors/plusfactor/pf2_signaling.py` |
| pf3 motive + opportunity (corroboration) | shared incentive and a usable channel | `detectors/plusfactor/pf3_motive_opportunity.py` |
| pf4 plan to action (gate: pf2 OR pf4) | B tracks A's private plan | `detectors/plusfactor/pf4_plan_to_action.py` |
| net pairwise influence | information-theoretic realization of pf1 | `detectors/plusfactor/net_influence.py` |
| naive correlation + LLM judge | back-half price correlation, transcript score | `detectors/naive.py` |

## Layout

```
src/pff/      envs/ agents/ inference/ logging/ quadrants/ detectors/ eval/
scripts/      run_scripted_game, run_mock_pipeline, run_eval,
              build_kaggle_job (open-weight sweep), kbench_task (frontier sweep)
tests/        mock modules + GPU-gated modules
paper/        build_pdf.py, make_results_figure.py
results/      audit_table.csv, figures/
PAPER.md / PAPER.pdf   the write-up
RESULTS.md    headline numbers
RESEARCH_LOG.md   full chronological record of decisions, failures, and results
```

## Originality and credits

The environment, four-condition generator, both detectors, and evaluation were written from
scratch. We did not clone or vendor the hackathon starter repository or any other collusion
codebase. Models: Qwen2.5-7B-Instruct (Alibaba/Qwen), Llama-3.1-8B-Instruct (Meta, via the
public NousResearch Hugging Face mirror), and frontier models via the Kaggle Benchmarks
proxy. Tools: vLLM, Hugging Face Transformers, Kaggle Notebooks and Kaggle Benchmarks,
numpy, scipy, matplotlib, pydantic. Economics follow the standard logit-demand
repeated-Bertrand setup (Calvano et al., 2020; Fish et al., 2024).

## Honesty

No results, metrics, or citations are fabricated. Numbers come from committed run artifacts.
DeepSeek-R1 could not be run to completion because the provider rate-limited it; this is
reported, not papered over. A few 2026 arXiv citations are flagged in the paper for a final
author-list confirmation.
