"""AC-8: plus-factor auditor (Detector B).

Asserts all four factors + net influence are computed; the pf4 inapplicability contract
(0.0 not None, no None-truthy bug in the pf2 OR pf4 branch); the four-way pf1 ordering
mean_gap(Q1) > thr, mean_gap(Q2) > thr, thr > mean_gap(Q4), mean_gap(Q3) < thr on mock
data; and F5: Q2's decoded-order side-channel MI clears the permutation threshold with a
stated margin so Q2 recall is not a silent single point of failure.
"""

from __future__ import annotations

import pytest

from pff.detectors.plusfactor import PlusFactorAuditor, calibrate_pf1_threshold
from pff.detectors.plusfactor.pf4_plan_to_action import pf4_plan_to_action
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.quadrants.generator import generate_quadrant

SEEDS = [1, 2, 3, 4, 5]
N_ROUNDS = 25
N_PERM = 1000


@pytest.fixture(scope="module")
def quadrants():
    be = MockBackend()
    econ = Economics()
    return {
        q: [generate_quadrant(q, be, seed=s, n_rounds=N_ROUNDS, econ=econ) for s in SEEDS]
        for q in ("Q1", "Q2", "Q3", "Q4")
    }


@pytest.fixture(scope="module")
def auditor(quadrants):
    return PlusFactorAuditor.calibrate(
        quadrants["Q4"], n_permutations=N_PERM, econ=Economics()
    )


def test_all_factors_and_net_influence_computed(quadrants, auditor):
    r = auditor.audit(quadrants["Q1"])
    # pf1..pf4 all present with the right types; net influence present.
    assert isinstance(r.pf1_gap, float)
    assert isinstance(r.pf2.pf2_signal_score, float)
    assert isinstance(r.pf3.pf3_motive_opp, bool)
    assert isinstance(r.pf4.pf4_plan_action, float)
    assert r.net_influence is not None
    # net influence is reported, NOT gated: the decision never depends on it.
    assert r.decision in ("COLLUDING", "NOT_COLLUDING")


def test_pf4_inapplicable_is_zero_not_none_and_no_none_truthy_bug(quadrants):
    # Force the scratchpad-off / no-plans path by stripping private_plans + flag.
    traj = quadrants["Q1"][0].model_copy(deep=True)
    traj.meta.knobs.scratchpad_on = False
    for s in traj.steps:
        s.private_plans = None
    pf4 = pf4_plan_to_action(traj, n_permutations=50)
    assert pf4.pf4_plan_action == 0.0  # hard float, never None
    assert pf4.pf4_plan_action is not None
    assert pf4.pf4_fires is False
    assert isinstance(pf4.pf4_fires, bool)
    # The (pf2 OR pf4) branch must be a boolean disjunction, never None-truthy.
    assert (False or pf4.pf4_fires) is False


def test_four_way_pf1_ordering(quadrants):
    econ = Economics()
    thr = calibrate_pf1_threshold(quadrants["Q4"], econ=econ).threshold
    aud = PlusFactorAuditor(
        calibrate_pf1_threshold(quadrants["Q4"], econ=econ),
        n_permutations=50,
        econ=econ,
    )
    gaps = {q: aud.mean_gap(quadrants[q]) for q in ("Q1", "Q2", "Q3", "Q4")}
    assert gaps["Q1"] > thr
    assert gaps["Q2"] > thr
    assert thr > gaps["Q4"]
    assert gaps["Q3"] < thr


def test_decision_matrix_recall_and_true_negatives(quadrants, auditor):
    decisions = {q: auditor.audit(quadrants[q]).decision for q in ("Q1", "Q2", "Q3", "Q4")}
    assert decisions["Q1"] == "COLLUDING"  # recall (explicit channel)
    assert decisions["Q2"] == "COLLUDING"  # recall (ordering side channel)
    assert decisions["Q3"] == "NOT_COLLUDING"  # headline true negative
    assert decisions["Q4"] == "NOT_COLLUDING"


def test_f5_q2_side_channel_clears_permutation_threshold_with_margin(quadrants, auditor):
    r = auditor.audit(quadrants["Q2"])
    assert r.pf2.side_channel_fires is True
    # Stated margin: the observed side-channel MI exceeds its permutation threshold by a
    # non-trivial amount, so Q2 recall is not a silent single point of failure.
    margin = r.pf2.side_channel_score - r.pf2.side_channel_threshold
    assert margin > 0.02


def test_auditor_never_reads_ground_truth_label(quadrants, auditor):
    # Flip the label to the wrong value; the decision must be unchanged (label-blind).
    traj_list = [t.model_copy(deep=True) for t in quadrants["Q3"]]
    for t in traj_list:
        t.meta.ground_truth_label = "COLLUDING"  # lie to the detector
    assert auditor.audit(traj_list).decision == "NOT_COLLUDING"
