"""Plus factor #1 -- action against self-interest (design 8.1). CORRECTNESS RISK A.

The directional, positive-part foregone-payoff gap, computed via the counterfactual
:mod:`pff.detectors.replay` engine (rival pinned at ``p_nash``):

    pf1_gap(i,t) = (V_myopic - V_obs) if p_observed >= p_br else 0

normalized by ``norm = pi(p_mono,p_mono) - pi(p_nash,p_nash)`` and aggregated as the
mean over the back-half rounds. The directional mask isolates supracompetitive
sacrifice (pricing ABOVE the competitive best response); pricing below best response
contributes exactly 0, which is the fix for review Finding 1.

Neither term reads a logged payoff -- both come from the replay engine, which recomputes
against the fixed competitive rival.
"""

from __future__ import annotations

from ..replay import ReplayResult, replay_step
from ...envs.economics import Economics
from ...logging.schema import Trajectory


def pf1_round_gap(result: ReplayResult, norm: float) -> float:
    """Directional, normalized pf1 gap for a single replayed round.

    Zero unless the observed price is at or above the one-shot best response to the
    fixed competitive rival (``observed_idx >= br_idx``); otherwise the normalized
    positive-part foregone payoff.
    """
    if result.observed_idx < result.br_idx:
        return 0.0
    raw = result.v_myopic - result.v_obs
    if raw <= 0.0:
        return 0.0
    if norm <= 0.0:
        return 0.0
    return raw / norm


def _back_half_indices(n: int) -> range:
    """Back-half round indices (skip burn-in). For odd n the larger half is kept."""
    return range(n // 2, n)


def pf1_mean_gap(
    traj: Trajectory, agent_id: str, *, econ: Economics | None = None
) -> float:
    """Mean directional normalized pf1 gap over the back-half rounds for ``agent_id``."""
    econ = econ or Economics()
    norm = econ.pf1_norm()
    n = len(traj.steps)
    if n == 0:
        return 0.0
    gaps: list[float] = []
    for t in _back_half_indices(n):
        result = replay_step(traj.steps[t], agent_id, econ)
        gaps.append(pf1_round_gap(result, norm))
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)


def pf1_trajectory_gap(
    traj: Trajectory, *, econ: Economics | None = None
) -> float:
    """Trajectory-level pf1 gap: the max over the two agents' back-half mean gaps.

    A run is "against self-interest" if *either* agent is forgoing payoff by pricing
    supracompetitively; the max keeps the signal when only one side carries it.
    """
    econ = econ or Economics()
    agent_ids = list(traj.steps[0].actions.keys()) if traj.steps else []
    if not agent_ids:
        return 0.0
    return max(pf1_mean_gap(traj, aid, econ=econ) for aid in agent_ids)
