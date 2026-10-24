"""Agent protocol (design 1). An agent maps an observation to a grid-index action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Observation:
    """What an agent sees at a decision point."""

    agent_id: str
    round_index: int
    own_history: list[int]  # own posted grid indices, oldest first
    rival_history: list[int]  # rival posted grid indices, oldest first
    channel_messages: list[str]  # messages visible this round (channel_on)
    grid_size: int
    nash_idx: int
    mono_idx: int
    # Optional decoded coordination context the generator supplies to scripted policies.
    extra: dict


@dataclass
class Decision:
    grid_index: int
    plan: str | None = None
    channel_message: str | None = None
    parse_failed: bool = False
    snapped: bool = False


class Agent(Protocol):
    agent_id: str

    def decide(self, obs: Observation) -> Decision:  # pragma: no cover - protocol
        ...
