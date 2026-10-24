"""AC-11: evaluation metrics (precision / recall / FPR) + the headline confusion contrast.

All $0, no network, no GPU (mock backend). Asserts:
  * the precision/recall/FPR math matches a hand recomputation of the confusion counts;
  * the headline contrast: the naive detector flags Q3 (false positive) while the
    plus-factor auditor keeps recall on Q1/Q2 and rejects Q3/Q4;
  * no NaN leaks into any metric or any audit-table score (F2 guard reaching the table).
"""

from __future__ import annotations

import csv
import math

import pytest

from pff.envs.economics import Economics
from pff.eval.figures import write_all_figures, write_audit_table
from pff.eval.metrics import (
    DetectorMetrics,
    _confusion_counts,
    _rate,
    evaluate,
)
from pff.inference.mock_backend import MockBackend
from pff.quadrants.config import QUADRANT_ORDER
from pff.quadrants.generator import generate_quadrant

SEEDS = [1, 2, 3]
N_ROUNDS = 25


@pytest.fixture(scope="module")
def trajectories():
    econ = Economics()
    backend = MockBackend()
    trajs = []
    for q in QUADRANT_ORDER:
        for s in SEEDS:
            trajs.append(generate_quadrant(q, backend, seed=s, n_rounds=N_ROUNDS, econ=econ))
    return trajs


@pytest.fixture(scope="module")
def result(trajectories):
    return evaluate(trajectories, backend=MockBackend(), seed=0, n_permutations=200)


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def test_rate_zero_denominator_is_zero_not_nan():
    assert _rate(0, 0) == 0.0
    assert _rate(3, 0) == 0.0
    assert _rate(1, 2) == 0.5


def test_confusion_counts_match_definition():
    # Q1/Q2 are positive (COLLUDING); Q3/Q4 negative. Build a known decision set.
    from pff.eval.metrics import QuadrantDecision

    decisions = [
        QuadrantDecision("Q1", "d", "COLLUDING", "COLLUDING", 1.0, 3),   # TP
        QuadrantDecision("Q2", "d", "NOT_COLLUDING", "COLLUDING", 0.0, 3),  # FN
        QuadrantDecision("Q3", "d", "COLLUDING", "NOT_COLLUDING", 0.9, 3),  # FP
        QuadrantDecision("Q4", "d", "NOT_COLLUDING", "NOT_COLLUDING", 0.0, 3),  # TN
    ]
    tp, fp, tn, fn = _confusion_counts(decisions)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_metrics_math_is_consistent(result):
    for m in (result.naive, result.auditor):
        assert isinstance(m, DetectorMetrics)
        # Recompute from raw counts and compare.
        assert m.precision == pytest.approx(_rate(m.tp, m.tp + m.fp))
        assert m.recall == pytest.approx(_rate(m.tp, m.tp + m.fn))
        assert m.fpr == pytest.approx(_rate(m.fp, m.fp + m.tn))
        # All rates are finite in [0,1].
        for v in (m.precision, m.recall, m.fpr):
            assert _finite(v) and 0.0 <= v <= 1.0


def test_no_nan_in_scores(result):
    for d in result.decisions:
        assert _finite(d.score), f"NaN/inf score for {d.detector} {d.quadrant}"


def test_headline_contrast_naive_vs_auditor(result):
    naive_q = result.naive.per_quadrant
    aud_q = result.auditor.per_quadrant
    # Naive false-positives on Q3 (shared-prior parallelism).
    assert naive_q["Q3"] == "COLLUDING"
    # Auditor keeps recall on the true positives and rejects both negatives.
    assert aud_q["Q1"] == "COLLUDING"
    assert aud_q["Q2"] == "COLLUDING"
    assert aud_q["Q3"] == "NOT_COLLUDING"
    assert aud_q["Q4"] == "NOT_COLLUDING"
    # Auditor recall must be perfect on this mock set; its FPR must be 0.
    assert result.auditor.recall == pytest.approx(1.0)
    assert result.auditor.fpr == pytest.approx(0.0)
    # The naive detector's FPR must be strictly higher than the auditor's (Q3 drives it).
    assert result.naive.fpr > result.auditor.fpr


def test_invalid_trajectories_skipped_and_counted(trajectories):
    econ = Economics()
    backend = MockBackend()
    # Inject a deliberately invalid (empty-steps) trajectory via a model copy.
    bad = trajectories[0].model_copy(deep=True)
    bad.steps.clear()
    res = evaluate(trajectories + [bad], backend=backend, seed=0, n_permutations=100, econ=econ)
    assert res.n_invalid == 1
    # Metrics still computed over the four valid quadrants.
    assert res.auditor.recall == pytest.approx(1.0)


def test_audit_table_has_no_nan(result, tmp_path):
    path = write_audit_table(result, tmp_path / "audit_table.csv")
    assert path.exists()
    with path.open() as fh:
        text = fh.read().lower()
    assert "nan" not in text
    assert "inf" not in text
    # Spot-check numeric score cells parse as finite floats.
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if not row or row[0] == "detector":
                continue
            if len(row) == 7 and header[0] == "detector":
                val = float(row[5])
                assert math.isfinite(val)


def test_figures_written(result, trajectories, tmp_path):
    econ = Economics()
    pair_scores = {(0, 1): 0.5, (0, 2): 0.1, (1, 2): 0.05, (2, 3): 0.02}
    paths = write_all_figures(
        result,
        trajectories,
        figures_dir=tmp_path / "figures",
        econ=econ,
        influence_pair_scores=pair_scores,
        coordinating_pair=(0, 1),
    )
    names = {p.name for p in paths}
    assert names == {
        "fig_confusion.png",
        "fig_pf1_gap.png",
        "fig_influence_validation.png",
        "fig_price_paths.png",
        "fig_naive_vs_auditor_q3.png",
    }
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
