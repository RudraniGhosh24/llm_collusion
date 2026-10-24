"""Trajectory log schema (design 4). Pydantic v2 models.

The schema is counterfactual-replay sufficient: ``payoff_inputs`` + ``economics`` let
the payoff function be recomputed deterministically offline, and ``prompts`` + ``knobs``
+ per-agent identity let any decision be re-presented to a substituted policy.

Determinism-critical vs volatile fields: ``run_id`` is a deterministic sha256 (never
uuid4); the byte-identity determinism assertion is taken over a "determinism projection"
that excludes ``StepRecord.wall_clock`` and ``RunMeta.git_revision`` (design 4.2).
"""

from __future__ import annotations

import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field


def compute_run_id(
    quadrant: str, seed: int, env_version: str, git_revision: str, backend: str
) -> str:
    """Deterministic run id: sha256(quadrant:seed:env_version:git_revision:backend)[:16]."""
    payload = f"{quadrant}:{seed}:{env_version}:{git_revision}:{backend}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Knobs(BaseModel):
    channel_on: bool
    scratchpad_on: bool
    name_mode: Literal["stable", "shuffled"]
    temperature: float = Field(ge=0.0, le=2.0)
    incentive_to_collude_on: bool


class AgentIdentity(BaseModel):
    agent_id: str
    family: str
    checkpoint: str
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int = Field(ge=0)


class EconomicsParams(BaseModel):
    a_i: float
    a_0: float
    mu: float
    c: float
    price_grid: list[float]
    p_nash: float
    p_mono: float
    nash_idx: int
    mono_idx: int


class Action(BaseModel):
    grid_index: int
    price: float
    parse_failed: bool = False
    snapped: bool = False


class Message(BaseModel):
    sender: str
    text: str
    logical_ts: int


class PromptRecord(BaseModel):
    system: str
    user: str
    rendered_observation: str


class StepRecord(BaseModel):
    round_index: int
    actions: dict[str, Action]
    action_order: list[str]
    payoffs: dict[str, float]
    payoff_inputs: dict
    private_plans: Optional[dict[str, str]] = None
    channel_messages: Optional[list[Message]] = None
    logical_clock: dict[str, int]
    wall_clock: Optional[dict[str, float]] = None
    prompts: dict[str, PromptRecord]


class RunMeta(BaseModel):
    run_id: str
    quadrant: Literal["Q1", "Q2", "Q3", "Q4"]
    ground_truth_label: Literal["COLLUDING", "NOT_COLLUDING"]
    knobs: Knobs
    shared_checkpoint: bool
    checkpoint_id: str
    agents: list[AgentIdentity]
    economics: EconomicsParams
    seed: int
    env_version: str
    git_revision: str
    backend: Literal["real", "mock"]
    n_rounds: int
    q4_policy: Optional[str] = None


class Trajectory(BaseModel):
    meta: RunMeta
    steps: list[StepRecord]
