"""Non-LLM scripted policies (design 8.1 / 3).

``CompetitivePolicy`` is the pf1 counterfactual rival: it returns the pinned Nash grid
price EVERY round regardless of history. The others are standard reference policies.
"""

from __future__ import annotations

from .base import Decision, Observation


class CompetitivePolicy:
    """Returns the pinned Nash grid index every round, independent of history.

    This is the non-cooperating, non-reciprocating rival used by the pf1 replay engine.
    """

    def __init__(self, agent_id: str, nash_idx: int) -> None:
        self.agent_id = agent_id
        self.nash_idx = nash_idx

    def decide(self, obs: Observation) -> Decision:
        return Decision(grid_index=self.nash_idx, plan="compete_at_nash")


class MonopolyPolicy:
    """Always prices at the joint-monopoly grid index."""

    def __init__(self, agent_id: str, mono_idx: int) -> None:
        self.agent_id = agent_id
        self.mono_idx = mono_idx

    def decide(self, obs: Observation) -> Decision:
        return Decision(grid_index=self.mono_idx, plan="hold_monopoly")


class TitForTatPolicy:
    """Starts at monopoly, then copies the rival's previous grid index."""

    def __init__(self, agent_id: str, mono_idx: int, nash_idx: int) -> None:
        self.agent_id = agent_id
        self.mono_idx = mono_idx
        self.nash_idx = nash_idx

    def decide(self, obs: Observation) -> Decision:
        if not obs.rival_history:
            return Decision(grid_index=self.mono_idx, plan="start_cooperative")
        return Decision(grid_index=obs.rival_history[-1], plan="mirror_rival")


class IndependentBestResponder:
    """One-shot grid best response to the rival's previous price.

    Requires a precomputed best-response table (``extra['br_table']``) so the policy is
    env-agnostic. Ties already resolved to lowest index in the table.
    """

    def __init__(self, agent_id: str, nash_idx: int) -> None:
        self.agent_id = agent_id
        self.nash_idx = nash_idx

    def decide(self, obs: Observation) -> Decision:
        if not obs.rival_history:
            return Decision(grid_index=self.nash_idx, plan="open_at_nash")
        br_table = obs.extra.get("br_table")
        if br_table is None:
            return Decision(grid_index=self.nash_idx, plan="no_br_table")
        idx = int(br_table[obs.rival_history[-1]])
        return Decision(grid_index=idx, plan="best_respond")
