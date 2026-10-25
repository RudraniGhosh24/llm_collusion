#!/usr/bin/env python3
"""Generate Figure 1: cross-model false-positive comparison (naive vs auditor)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

models = ["Qwen2.5-7B", "Llama-3.1-8B", "GPT-5.5", "Gemini-3.8-flash", "Gemini-2.5-pro", "Claude Sonnet 4.6"]
naive_fpr = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
auditor_fpr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x = np.arange(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 3.6))
b1 = ax.bar(x - w/2, naive_fpr, w, label="Naive (correlation + judge)", color="#d1495b")
b2 = ax.bar(x + w/2, auditor_fpr, w, label="Plus-factor auditor", color="#2e7d87")
ax.set_ylabel("False-positive rate on\ninnocent conditions (Q3, Q4)")
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
ax.axhspan(0, 0.001, color="none")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_title("Naive detector false-positives on innocent parallelism; the auditor never does",
             fontsize=10.5)
for b in b1:
    h = b.get_height()
    ax.text(b.get_x()+b.get_width()/2, h+0.03, f"{h:.0f}", ha="center", fontsize=8,
            color="#d1495b", fontweight="bold")
for b in b2:
    ax.text(b.get_x()+b.get_width()/2, 0.03, "0", ha="center", fontsize=8,
            color="#2e7d87", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = Path(__file__).resolve().parent / "fig_crossmodel_fpr.png"
fig.savefig(out, dpi=200)
print("wrote", out)
