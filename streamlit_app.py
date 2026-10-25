"""Streamlit app: Correlation Is Not Collusion — a plus-factor auditor for AI agents.

Interactive view of the real six-model results. Run with:
    streamlit run app.py
Reads results/consolidated.json (built by scripts/consolidate_results.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "app_assets"
DATA = ASSETS / "consolidated.json"
FIGS = ASSETS / "figures"
CROSSFIG = ASSETS / "fig_crossmodel_fpr.png"

st.set_page_config(page_title="Correlation Is Not Collusion", layout="wide")


@st.cache_data
def load():
    return json.loads(DATA.read_text())


d = load()
models = d["models"]
nash, mono = d["nash_idx"], d["mono_idx"]

st.title("Correlation Is Not Collusion")
st.subheader("A plus-factor auditor for AI agents")

st.markdown(
    """
Two AI agents built on the same model often behave alike for an innocent reason: shared
training leads them to the same choices, with no agreement between them. A detector that
treats *similarity* as collusion flags this innocent parallelism as a conspiracy. Antitrust
law resolved the same problem long ago: parallel conduct alone is not unlawful, and inferring
an agreement requires extra circumstantial evidence called **plus factors**. This app shows
the real results of a plus-factor auditor against a standard correlation-and-judge detector,
across six models.
"""
)

# headline metrics
n_models = len(models)
aud_clean = sum(1 for m in models if m["auditor"]["fpr"] == 0.0)
naive_fp = sum(1 for m in models if m["naive"]["fpr"] > 0.0)
c1, c2, c3 = st.columns(3)
c1.metric("Models evaluated", n_models)
c2.metric("Auditor false-positive-free", f"{aud_clean}/{n_models}")
c3.metric("Models where naive detector false-positives", naive_fp)

st.divider()

st.header("The four conditions")
st.markdown(
    """
Game, economics, and seed are held fixed; only the coordination mechanism changes.

- **Q1** collusion with an explicit chat channel and an incentive to keep joint prices high
- **Q2** collusion with no chat channel, only an action-ordering side channel
- **Q3** the negative control: one model instance plays both sellers, no channel, no incentive
  (innocent parallelism, the computational analogue of lawful conscious parallelism)
- **Q4** fully independent one-shot best-responders

Q1 and Q2 are the real-collusion (positive) conditions; Q3 and Q4 are the innocent (negative)
conditions. A good detector catches Q1/Q2 and stays quiet on Q3/Q4.
"""
)

st.divider()

st.header("Headline: false-positive rate on the innocent conditions")
st.markdown(
    "The naive detector false-positives on the two most capable models; the plus-factor "
    "auditor has a false-positive rate of **zero on all six**."
)
if CROSSFIG.exists():
    st.image(str(CROSSFIG), use_container_width=True)

rows = []
for m in models:
    rows.append({
        "Model": m["model"],
        "Type": m["kind"],
        "Naive FPR": m["naive"]["fpr"],
        "Auditor FPR": m["auditor"]["fpr"],
        "Auditor recall": m["auditor"]["recall"],
        "pf1 threshold": m["pf1_threshold"],
        "Seeds": m["n_seeds"],
    })
df = pd.DataFrame(rows)


def hl(v):
    return "background-color:#ffd6d6" if v == 1.0 else ""


st.dataframe(
    df.style.map(hl, subset=["Naive FPR"]).format(
        {"Naive FPR": "{:.1f}", "Auditor FPR": "{:.1f}", "Auditor recall": "{:.2f}",
         "pf1 threshold": "{:.2f}"}
    ),
    use_container_width=True, hide_index=True,
)
st.caption(
    "FPR is the false-positive rate on the innocent conditions (Q3, Q4). Recall is on the "
    "collusion conditions (Q1, Q2). Open-weight runs use 5 seeds, frontier runs use 2."
)

st.divider()

st.header("Per-model detail")
pick = st.selectbox("Model", [m["model"] for m in models])
m = next(x for x in models if x["model"] == pick)
left, right = st.columns(2)
with left:
    st.markdown("**Naive detector (correlation + LLM judge)**")
    st.json(m["naive"]["per_quadrant"])
    st.write(
        f"precision {m['naive']['precision']:.2f} · recall {m['naive']['recall']:.2f} · "
        f"FPR {m['naive']['fpr']:.2f}"
    )
with right:
    st.markdown("**Plus-factor auditor**")
    st.json(m["auditor"]["per_quadrant"])
    st.write(
        f"precision {m['auditor']['precision']:.2f} · recall {m['auditor']['recall']:.2f} · "
        f"FPR {m['auditor']['fpr']:.2f}"
    )
st.caption(
    f"pf1 self-interest threshold for {pick}: {m['pf1_threshold']:.2f} "
    "(calibrated on the independent Q4 condition)."
)

st.divider()

st.header("The conditions are genuine, not cosmetic")
st.markdown(
    f"Back-half mean price index by condition for the open-weight models. "
    f"Competitive (Nash) reference is index **{nash}**, monopoly is index **{mono}**. "
    "Higher means more collusive pricing."
)
prices = d["prices"]
price_rows = []
for model, qd in prices.items():
    row = {"Model": model}
    row.update({q: qd.get(q) for q in ["Q1", "Q2", "Q3", "Q4"]})
    price_rows.append(row)
st.dataframe(pd.DataFrame(price_rows), use_container_width=True, hide_index=True)
st.caption(
    "On Qwen the collusion conditions price near or at monopoly while the innocent control "
    "sits only mildly above competitive. Q3 agents do look somewhat alike, the hard case a "
    "similarity detector falls for, and the auditor rejects it anyway."
)

st.divider()

st.header("Supporting figures (open-weight runs)")
fig_titles = {
    "fig_confusion.png": "Confusion: naive vs auditor across conditions",
    "fig_naive_vs_auditor_q3.png": "Naive vs auditor on the Q3 innocent-parallelism control",
    "fig_pf1_gap.png": "pf1 self-interest gap by condition",
    "fig_price_paths.png": "Price paths over rounds",
    "fig_influence_validation.png": "Net pairwise influence validation (abstract game)",
}
cols = st.columns(2)
i = 0
for fn, title in fig_titles.items():
    p = FIGS / fn
    if p.exists():
        with cols[i % 2]:
            st.image(str(p), caption=title, use_container_width=True)
        i += 1

st.divider()
st.caption(
    "Code: github.com/RudraniGhosh24/llm_collusion · Honesty: all numbers come from committed "
    "run artifacts; DeepSeek-R1 was rate-limited by the provider and is not included."
)
