"""AC-9 / FR-A: counterfactual replay engine + plus factor #1 (CORRECTNESS RISK A).

The replay engine must RECOMPUTE both V_obs and V_myopic against a rival pinned at the
fixed competitive price p_nash via the payoff function -- never read the logged payoff.
These tests lock in the directional-gap fix (review Finding 1):

* a supracompetitive observed action -> gap > 0, and the replayed V_obs DIFFERS from the
  logged payoff (proving recompute, not echo);
* an observed action equal to the one-shot best response against p_nash -> gap == 0;
* an observed action BELOW best response -> directional gap == 0 even though the
  two-sided quantity would be large.
"""

from __future__ import annotations

import pytest

from pff.detectors.plusfactor.pf1_self_interest import pf1_round_gap
from pff.detectors.replay import ReplayError, replay_step
from pff.envs.economics import Economics
from pff.logging.schema import Action, PromptRecord, StepRecord

ECON = Economics()


def _step(agent_idx: int, rival_idx: int, *, with_inputs: bool = True) -> StepRecord:
    """Build a one-round StepRecord where agent 'A' played ``agent_idx`` and rival 'B'
    played ``rival_idx``; the logged payoff is earned against the ACTUAL rival."""
    p_a = float(ECON.grid[agent_idx])
    p_b = float(ECON.grid[rival_idx])
    payoff_a = ECON.payoff(p_a, p_b)
    payoff_b = ECON.payoff(p_b, p_a)
    payoff_inputs = {}
    if with_inputs:
        payoff_inputs = {
            "prices": {"A": p_a, "B": p_b},
            "quantities": {"A": ECON.demand(p_a, p_b), "B": ECON.demand(p_b, p_a)},
            "c": ECON.c,
            "mu": ECON.mu,
            "a_i": ECON.a_i,
            "a_0": ECON.a_0,
        }
    pr = PromptRecord(system="", user="", rendered_observation="")
    return StepRecord(
        round_index=0,
        actions={
            "A": Action(grid_index=agent_idx, price=p_a),
            "B": Action(grid_index=rival_idx, price=p_b),
        },
        action_order=["A", "B"],
        payoffs={"A": payoff_a, "B": payoff_b},
        payoff_inputs=payoff_inputs,
        logical_clock={"A": 0, "B": 1},
        prompts={"A": pr, "B": pr},
    )


def test_both_terms_recomputed_against_fixed_nash_rival():
    # Observed supracompetitive action (monopoly idx 9) against an ACTUAL rival also at
    # monopoly -> the logged payoff is high. The counterfactual rival is pinned at Nash.
    step = _step(ECON.mono_idx, ECON.mono_idx)
    r = replay_step(step, "A", ECON)
    # V_obs recomputed vs the fixed Nash rival, NOT the logged (mono-vs-mono) payoff.
    expected_v_obs = ECON.payoff(ECON.p_mono, ECON.p_nash)
    assert r.v_obs == pytest.approx(expected_v_obs)
    # V_myopic is the grid argmax best response to the fixed Nash rival.
    expected_v_myopic = max(ECON.payoff(float(p), ECON.p_nash) for p in ECON.grid)
    assert r.v_myopic == pytest.approx(expected_v_myopic)
    assert r.br_idx == ECON.nash_idx  # symmetric-NE: BR to Nash is Nash


def test_replayed_v_obs_differs_from_logged_payoff():
    # Logged payoff is earned against the actual (monopoly) rival; the replayed V_obs is
    # against the competitive (Nash) rival -> they must differ (recompute, not echo).
    step = _step(ECON.mono_idx, ECON.mono_idx)
    r = replay_step(step, "A", ECON)
    assert abs(r.v_obs - r.logged_payoff) > 1e-6


def test_supracompetitive_action_yields_positive_directional_gap():
    step = _step(ECON.mono_idx, ECON.mono_idx)
    r = replay_step(step, "A", ECON)
    gap = pf1_round_gap(r, ECON.pf1_norm())
    assert gap > 0.0


def test_best_response_action_yields_zero_gap():
    # Observed action equals the one-shot best response to a Nash rival (== Nash).
    step = _step(ECON.nash_idx, ECON.mono_idx)
    r = replay_step(step, "A", ECON)
    assert r.observed_idx == r.br_idx
    gap = pf1_round_gap(r, ECON.pf1_norm())
    assert gap == pytest.approx(0.0, abs=1e-9)


def test_below_best_response_directional_gap_is_zero():
    # Observed action BELOW best response (idx 0). The two-sided quantity is large, but
    # the DIRECTIONAL gap must be exactly 0 (locks in review Finding 1).
    assert 0 < ECON.nash_idx  # idx 0 is strictly below the best response
    step = _step(0, ECON.mono_idx)
    r = replay_step(step, "A", ECON)
    two_sided = r.v_myopic - r.v_obs
    assert two_sided > 0.0  # two-sided quantity is large on the low side
    gap = pf1_round_gap(r, ECON.pf1_norm())
    assert gap == 0.0  # directional mask zeroes it


def test_missing_payoff_inputs_raises_replay_error():
    step = _step(ECON.mono_idx, ECON.mono_idx, with_inputs=False)
    with pytest.raises(ReplayError):
        replay_step(step, "A", ECON)
