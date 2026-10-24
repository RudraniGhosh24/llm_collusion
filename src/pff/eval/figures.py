"""Figure + audit-table writers (design 10).

Writes five PNGs under ``<WT>/results/figures/`` plus ``results/audit_table.csv``. All
plotting uses the non-interactive Agg backend, so this runs headless / GPU-free in the
mock pipeline and in CI. Nothing here reads a GPU or the network.

Figures:
  * ``fig_confusion.png``        -- precision / recall / FPR bars, naive vs auditor.
  * ``fig_pf1_gap.png``          -- pf1 counterfactual gap distribution by quadrant, with
                                    the ~1.35 monopoly reference line.
  * ``fig_influence_validation.png`` -- seeded coordinating pair vs others (abstract game).
  * ``fig_price_paths.png``      -- mean price per quadrant vs p_nash / p_mono.
  * ``fig_naive_vs_auditor_q3.png`` -- the money shot: naive flags Q3, auditor does not.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display, no GPU.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..detectors.plusfactor.pf1_self_interest import pf1_trajectory_gap  # noqa: E402
from ..envs.economics import Economics  # noqa: E402
from ..logging.schema import Trajectory  # noqa: E402
from .metrics import EvalResult  # noqa: E402

FIGURE_NAMES = (
    "fig_confusion.png",
    "fig_pf1_gap.png",
    "fig_influence_validation.png",
    "fig_price_paths.png",
    "fig_naive_vs_auditor_q3.png",
)

# Normalized monopoly reference for the pf1 gap axis (design: ~1.35 at full monopoly).
MONOPOLY_PF1_REFERENCE = 1.35
QUADRANT_ORDER = ("Q1", "Q2", "Q3", "Q4")


def _quadrant_groups(
    trajectories: list[Trajectory],
) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        if not t.steps or len(t.steps[0].actions) < 2:
            continue
        groups.setdefault(t.meta.quadrant, []).append(t)
    return groups


# ---------------------------------------------------------------------------
# fig_confusion.png
# ---------------------------------------------------------------------------
def plot_confusion(result: EvalResult, path: Path) -> Path:
    labels = ["precision", "recall", "FPR"]
    naive_vals = [result.naive.precision, result.naive.recall, result.naive.fpr]
    aud_vals = [result.auditor.precision, result.auditor.recall, result.auditor.fpr]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, naive_vals, width, label="naive baseline", color="#c0504d")
    ax.bar(x + width / 2, aud_vals, width, label="plus-factor auditor", color="#4f81bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Detector comparison: precision / recall / FPR")
    for i, v in enumerate(naive_vals):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(aud_vals):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig_pf1_gap.png
# ---------------------------------------------------------------------------
def plot_pf1_gap(
    groups: dict[str, list[Trajectory]],
    path: Path,
    *,
    econ: Economics,
    pf1_threshold: float,
) -> Path:
    quadrants = [q for q in QUADRANT_ORDER if q in groups]
    per_q_gaps = [
        [pf1_trajectory_gap(t, econ=econ) for t in groups[q]] for q in quadrants
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(quadrants))
    for i, gaps in enumerate(per_q_gaps):
        jitter = (np.arange(len(gaps)) - (len(gaps) - 1) / 2) * 0.04
        ax.scatter(
            np.full(len(gaps), positions[i]) + jitter,
            gaps,
            color="#4f81bd",
            s=28,
            zorder=3,
        )
        ax.scatter(positions[i], float(np.mean(gaps)), color="#1f3864", marker="_", s=600, zorder=4)
    ax.axhline(
        pf1_threshold,
        color="#999999",
        linestyle="--",
        label=f"pf1 firing threshold ({pf1_threshold:.3f})",
    )
    ax.axhline(
        MONOPOLY_PF1_REFERENCE,
        color="#c0504d",
        linestyle=":",
        label=f"~monopoly reference ({MONOPOLY_PF1_REFERENCE})",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(quadrants)
    ax.set_ylabel(
        "directional foregone-payoff ratio (pf1 gap), rival pinned at Nash"
    )
    ax.set_title("pf1 counterfactual payoff gap by quadrant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig_influence_validation.png
# ---------------------------------------------------------------------------
def plot_influence_validation(
    pair_scores: dict[tuple[int, int], float],
    coordinating_pair: tuple[int, int],
    path: Path,
) -> Path:
    pairs = list(pair_scores.keys())
    labels = [f"{p[0]}-{p[1]}" for p in pairs]
    vals = [pair_scores[p] for p in pairs]
    colors = [
        "#4f81bd" if p == coordinating_pair else "#bfbfbf" for p in pairs
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("net pairwise influence (nats)")
    ax.set_title(
        f"Net-influence validation: seeded pair {coordinating_pair} vs others"
    )
    # Legend proxy.
    ax.bar([], [], color="#4f81bd", label="seeded coordinating pair")
    ax.bar([], [], color="#bfbfbf", label="other pairs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig_price_paths.png
# ---------------------------------------------------------------------------
def _mean_price_path(trajs: list[Trajectory]) -> np.ndarray:
    """Mean (over agents and seeds) posted price per round."""
    n_rounds = min(len(t.steps) for t in trajs)
    per_round = np.zeros(n_rounds, dtype=float)
    for r in range(n_rounds):
        vals = []
        for t in trajs:
            for a in t.steps[r].actions.values():
                vals.append(a.price)
        per_round[r] = float(np.mean(vals))
    return per_round


def plot_price_paths(
    groups: dict[str, list[Trajectory]], path: Path, *, econ: Economics
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"Q1": "#c0504d", "Q2": "#e8973a", "Q3": "#4f81bd", "Q4": "#5b9b5b"}
    for q in QUADRANT_ORDER:
        if q not in groups:
            continue
        path_vals = _mean_price_path(groups[q])
        ax.plot(
            range(len(path_vals)),
            path_vals,
            marker="o",
            markersize=3,
            label=q,
            color=colors.get(q, None),
        )
    ax.axhline(econ.p_nash, color="#333333", linestyle="--", label=f"p_nash ({econ.p_nash:.3f})")
    ax.axhline(econ.p_mono, color="#333333", linestyle=":", label=f"p_mono ({econ.p_mono:.3f})")
    ax.set_xlabel("round")
    ax.set_ylabel("mean posted price")
    ax.set_title("Mean price trajectory per quadrant vs Nash / monopoly")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# fig_naive_vs_auditor_q3.png
# ---------------------------------------------------------------------------
def plot_naive_vs_auditor_q3(result: EvalResult, path: Path) -> Path:
    quadrants = [q for q in QUADRANT_ORDER if q in result.naive.per_quadrant]
    naive_flags = [
        1 if result.naive.per_quadrant.get(q) == "COLLUDING" else 0 for q in quadrants
    ]
    aud_flags = [
        1 if result.auditor.per_quadrant.get(q) == "COLLUDING" else 0 for q in quadrants
    ]
    x = np.arange(len(quadrants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, naive_flags, width, label="naive baseline", color="#c0504d")
    ax.bar(x + width / 2, aud_flags, width, label="plus-factor auditor", color="#4f81bd")
    ax.set_xticks(x)
    ax.set_xticklabels(quadrants)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["NOT_COLLUDING", "COLLUDING"])
    ax.set_title("The money shot: naive flags Q3, auditor does not")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# audit_table.csv
# ---------------------------------------------------------------------------
def write_audit_table(result: EvalResult, path: Path) -> Path:
    """Write the per-quadrant, per-detector confusion/decision table to CSV.

    No NaN ever appears: scores come from safe estimators and rates use ``_rate``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "detector",
                "quadrant",
                "ground_truth",
                "decision",
                "correct",
                "score",
                "n_seeds",
            ]
        )
        for d in result.decisions:
            correct = (d.decision == d.ground_truth)
            writer.writerow(
                [
                    d.detector,
                    d.quadrant,
                    d.ground_truth,
                    d.decision,
                    int(correct),
                    f"{d.score:.6f}",
                    d.n_seeds,
                ]
            )
        writer.writerow([])
        writer.writerow(
            ["detector", "precision", "recall", "fpr", "tp", "fp", "tn", "fn"]
        )
        for m in (result.naive, result.auditor):
            writer.writerow(
                [
                    m.detector,
                    f"{m.precision:.6f}",
                    f"{m.recall:.6f}",
                    f"{m.fpr:.6f}",
                    m.tp,
                    m.fp,
                    m.tn,
                    m.fn,
                ]
            )
    return path


def write_all_figures(
    result: EvalResult,
    trajectories: list[Trajectory],
    *,
    figures_dir: Path,
    econ: Economics,
    influence_pair_scores: dict[tuple[int, int], float],
    coordinating_pair: tuple[int, int],
) -> list[Path]:
    """Write all five figures to ``figures_dir`` and return the paths (in FIGURE order)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    groups = _quadrant_groups(trajectories)
    paths = [
        plot_confusion(result, figures_dir / "fig_confusion.png"),
        plot_pf1_gap(
            groups,
            figures_dir / "fig_pf1_gap.png",
            econ=econ,
            pf1_threshold=result.pf1_threshold,
        ),
        plot_influence_validation(
            influence_pair_scores,
            coordinating_pair,
            figures_dir / "fig_influence_validation.png",
        ),
        plot_price_paths(groups, figures_dir / "fig_price_paths.png", econ=econ),
        plot_naive_vs_auditor_q3(
            result, figures_dir / "fig_naive_vs_auditor_q3.png"
        ),
    ]
    return paths
