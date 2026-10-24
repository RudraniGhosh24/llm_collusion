"""Counterfactual replay engine (design 8.1 / FR-9 / FR-A). CORRECTNESS RISK A.

This engine implements the ``do``-intervention behind plus factor #1: it reconstructs a
decision context from the *logged* ``payoff_inputs`` + economics, substitutes the rival
with a non-cooperating :class:`CompetitivePolicy` pinned to ``p_nash`` every round, and
**recomputes** the acting agent's counterfactual payoffs by evaluating the environment
payoff function over the price grid.

The logged realized payoff is NEVER read as the counterfactual: ``V_obs`` is recomputed
as ``pi_i(p_i_observed, p_nash)`` (the observed price against a *competitive* rival, not
against the actual logged rival), and ``V_myopic`` is the grid argmax best response to
that same competitive rival. Because the logged payoff was earned against the *actual*
rival, the replayed ``V_obs`` differs from it whenever the actual rival did not price at
Nash -- that difference is the proof the engine recomputes rather than echoes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.base import Observation
from ..agents.scripted import CompetitivePolicy
from ..envs.economics import Economics
from ..logging.schema import StepRecord


class ReplayError(ValueError):
    """Raised when a step lacks the ``payoff_inputs`` needed to replay a decision."""


@dataclass(frozen=True)
class ReplayResult:
    """One round's counterfactual replay for a single acting agent.

    ``v_obs`` / ``v_myopic`` are recomputed against the fixed competitive rival; they
    are NOT the logged realized payoff. ``logged_payoff`` is carried only so callers /
    tests can assert the recompute differs from the echo.
    """

    agent_id: str
    round_index: int
    observed_idx: int
    br_idx: int
    v_obs: float
    v_myopic: float
    logged_payoff: float
    rival_nash_idx: int


def _observed_index(step: StepRecord, agent_id: str, econ: Economics) -> int:
    """Reconstruct the acting agent's observed grid index from ``payoff_inputs``.

    We deliberately read the observed *price* out of ``payoff_inputs`` (the replay-
    sufficient log field) and map it back onto the grid, rather than trusting the
    convenience ``actions`` mirror, so the replay path depends only on the economic
    inputs the schema promises (design 4.1).
    """
    prices = step.payoff_inputs.get("prices") if step.payoff_inputs else None
    if not prices or agent_id not in prices:
        raise ReplayError(
            f"step {step.round_index} missing payoff_inputs.prices for agent {agent_id!r}"
        )
    p_obs = float(prices[agent_id])
    # Snap to the nearest grid index (prices are logged as exact grid values).
    diffs = [abs(float(p) - p_obs) for p in econ.grid]
    return int(min(range(len(econ.grid)), key=lambda k: diffs[k]))


def replay_step(
    step: StepRecord, agent_id: str, econ: Economics
) -> ReplayResult:
    """Replay a single round for ``agent_id`` against a rival pinned at ``p_nash``.

    Steps (design 8.1):
      1. Reconstruct the observed grid price from ``payoff_inputs`` (raise
         :class:`ReplayError` if missing).
      2. Substitute the rival with :class:`CompetitivePolicy` -> fixed ``p_nash``.
      3. Recompute ``V_obs = pi_i(p_observed, p_nash)`` and
         ``V_myopic = max_p pi_i(p, p_nash)`` via the payoff function. The logged
         realized payoff is never used as either term.
    """
    observed_idx = _observed_index(step, agent_id, econ)

    # (2) The counterfactual rival: a fixed competitive best-responder (p_nash always).
    rival = CompetitivePolicy(agent_id="__cf_rival__", nash_idx=econ.nash_idx)
    rival_idx = rival.decide(
        Observation(
            agent_id="__cf_rival__",
            round_index=step.round_index,
            own_history=[],
            rival_history=[],
            channel_messages=[],
            grid_size=len(econ.grid),
            nash_idx=econ.nash_idx,
            mono_idx=econ.mono_idx,
            extra={},
        )
    ).grid_index
    p_rival = float(econ.grid[rival_idx])

    # (3) Recompute both terms via the payoff function -- never read the logged payoff.
    p_obs = float(econ.grid[observed_idx])
    v_obs = econ.payoff(p_obs, p_rival)
    grid_payoffs = [econ.payoff(float(p), p_rival) for p in econ.grid]
    v_myopic = max(grid_payoffs)
    # Argmax best response to the fixed competitive rival; ties -> lowest index.
    br_idx = int(min(range(len(econ.grid)), key=lambda k: (-grid_payoffs[k], k)))

    logged = float(step.payoffs.get(agent_id, 0.0)) if step.payoffs else 0.0
    return ReplayResult(
        agent_id=agent_id,
        round_index=step.round_index,
        observed_idx=observed_idx,
        br_idx=br_idx,
        v_obs=v_obs,
        v_myopic=v_myopic,
        logged_payoff=logged,
        rival_nash_idx=rival_idx,
    )
