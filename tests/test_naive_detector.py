"""AC-7: naive association baseline (Detector A).

Asserts the detector emits both scores + a flag; the F2 NaN guard maps a zero-variance
(flat) series to correlation 0; Q3 drives a false positive on the mock (the headline
contrast with the auditor); tau_naive is calibrated on Q1 vs Q4 ONLY (never Q3); and the
detector never reads ground_truth_label.
"""

from __future__ import annotations

import numpy as np

from pff.detectors.naive import (
    NaiveDetector,
    _pearson_safe,
    calibrate_tau_naive,
    naive_correlation,
)
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.quadrants.generator import generate_quadrant

SEEDS = [1, 2, 3, 4, 5]
N_ROUNDS = 25


def _quadrants():
    be = MockBackend()
    econ = Economics()
    trajs = {
        q: [generate_quadrant(q, be, seed=s, n_rounds=N_ROUNDS, econ=econ) for s in SEEDS]
        for q in ("Q1", "Q2", "Q3", "Q4")
    }
    return be, trajs


def test_emits_both_scores_and_flag():
    be, trajs = _quadrants()
    tau = calibrate_tau_naive(trajs["Q1"], trajs["Q4"], be)
    det = NaiveDetector(be, tau.tau)
    r = det.score(trajs["Q1"][0])
    assert isinstance(r.naive_corr, float)
    assert isinstance(r.naive_judge, float)
    assert 0.0 <= r.naive_judge <= 1.0
    assert r.decision in ("COLLUDING", "NOT_COLLUDING")


def test_f2_nan_guard_on_zero_variance_series():
    # Direct guard: a flat (zero-variance) series -> correlation 0, never NaN.
    flat = np.full(10, 5.0)
    varying = np.arange(10, dtype=float)
    assert _pearson_safe(flat, varying) == 0.0
    assert _pearson_safe(flat, flat) == 0.0
    assert not np.isnan(_pearson_safe(flat, varying))

    # Build a trajectory whose back-half prices are perfectly flat and confirm the
    # detector's correlation is a finite 0 (so the confusion table never emits NaN).
    be, trajs = _quadrants()
    traj = trajs["Q4"][0].model_copy(deep=True)
    flat_price = traj.steps[0].actions["A"].price
    for s in traj.steps:
        for aid in ("A", "B"):
            s.actions[aid].grid_index = traj.meta.economics.nash_idx
            s.actions[aid].price = flat_price
    corr = naive_correlation(traj)
    assert corr == 0.0
    assert np.isfinite(corr)


def test_q3_drives_false_positive_on_mock():
    be, trajs = _quadrants()
    tau = calibrate_tau_naive(trajs["Q1"], trajs["Q4"], be)
    det = NaiveDetector(be, tau.tau)
    # The naive detector flags Q3 as COLLUDING (its intended false positive): shared-prior
    # parallelism inflates both the correlation and the judge score.
    q3_decisions = [det.score(t).decision for t in trajs["Q3"]]
    assert any(d == "COLLUDING" for d in q3_decisions)
    assert all(d == "COLLUDING" for d in q3_decisions)


def test_tau_calibrated_without_q3():
    be, trajs = _quadrants()
    tau = calibrate_tau_naive(trajs["Q1"], trajs["Q4"], be)
    # Procedure logged and references only Q1/Q4; Q3 explicitly excluded.
    assert "Q3 excluded" in tau.procedure
    # Calibration consumed exactly the Q1 and Q4 sets (one score per seed each).
    assert len(tau.q1_scores) == len(SEEDS)
    assert len(tau.q4_scores) == len(SEEDS)
    assert isinstance(tau.tau, float)


def test_naive_detector_does_not_read_label():
    be, trajs = _quadrants()
    tau = calibrate_tau_naive(trajs["Q1"], trajs["Q4"], be)
    det = NaiveDetector(be, tau.tau)
    # Flip the label; the score/decision must be unchanged (label-blind).
    traj = trajs["Q3"][0].model_copy(deep=True)
    before = det.score(traj)
    traj.meta.ground_truth_label = "NOT_COLLUDING"  # opposite of a false positive
    after = det.score(traj)
    assert before.decision == after.decision
    assert before.naive_corr == after.naive_corr
    assert before.naive_judge == after.naive_judge
