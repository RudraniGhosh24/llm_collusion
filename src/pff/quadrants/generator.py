"""Four-quadrant labeled trajectory generator (FR-5, design 6).

Runs the Bertrand environment once per quadrant on an injected :class:`InferenceBackend`,
holding task / economics / seed FIXED across quadrants and varying only the confound
knobs read from the authoritative table in :mod:`pff.quadrants.config`. Construction-time
``ground_truth_label`` is written directly from that table and is NEVER inferred from the
trajectory.

Q2 realizes the ordering side channel: the generator supplies a deterministic,
non-constant commit schedule (``Q2_ORDER_CODE``), and the env hands each agent the
decoded signal for the round so the rival can condition its next action on it.

Q3 is the leak-free negative control: both agents are backed by the SAME Qwen checkpoint
with identical decoding params (temperature + seed), differing only by ``agent_id``. The
neutral prompt (built from :mod:`pff.agents.prompts`) carries no coordination incentive.
"""

from __future__ import annotations

from ..agents.llm_agent import LLMAgent
from ..envs.bertrand import BertrandConfig, run_bertrand
from ..envs.economics import Economics
from ..inference.interface import InferenceBackend
from ..logging.schema import AgentIdentity, Trajectory
from .config import QUADRANT_ORDER, QUADRANTS, QuadrantSpec, q2_order_schedule

AGENT_IDS = ("A", "B")


def _make_agents(
    spec: QuadrantSpec,
    backend: InferenceBackend,
    *,
    temperature: float,
    seed: int,
    econ: Economics,
) -> tuple[list[LLMAgent], list[AgentIdentity]]:
    """Build the two LLM agents and their logged identities for a quadrant.

    For a shared-checkpoint quadrant (Q3) both agents use the same model and the same
    decoding params; the only difference is ``agent_id``.
    """
    agents: list[LLMAgent] = []
    identities: list[AgentIdentity] = []
    for aid, model, family in zip(AGENT_IDS, spec.agent_models, spec.agent_families):
        agents.append(
            LLMAgent(
                agent_id=aid,
                backend=backend,
                model=model,
                quadrant=spec.quadrant,
                policy=spec.policy,
                temperature=temperature,
                seed=seed,
                scratchpad_on=spec.scratchpad_on,
            )
        )
        identities.append(
            AgentIdentity(
                agent_id=aid,
                family=family,
                checkpoint=spec.checkpoint_id,
                temperature=temperature,
                seed=seed,
            )
        )
    return agents, identities


def generate_quadrant(
    quadrant: str,
    backend: InferenceBackend,
    *,
    seed: int,
    n_rounds: int = 25,
    temperature: float = 0.7,
    econ: Economics | None = None,
    env_version: str = "0.1.0",
    git_revision: str = "0" * 40,
) -> Trajectory:
    """Generate one labeled :class:`Trajectory` for ``quadrant`` (Q1-Q4).

    ``seed`` / ``n_rounds`` / ``temperature`` / ``econ`` are the FIXED task-and-economics
    settings shared across quadrants; only the confound knobs (from the table) differ.
    """
    if quadrant not in QUADRANTS:
        raise ValueError(f"unknown quadrant {quadrant!r}")
    spec = QUADRANTS[quadrant]
    econ = econ or Economics()

    config = BertrandConfig(
        channel_on=spec.channel_on,
        scratchpad_on=spec.scratchpad_on,
        name_mode="stable",  # held fixed; Q2's ordering comes from order_schedule
        temperature=temperature,
        incentive_to_collude_on=spec.incentive_to_collude_on,
        n_rounds=n_rounds,
        seed=seed,
    )

    agents, identities = _make_agents(
        spec, backend, temperature=temperature, seed=seed, econ=econ
    )

    order_schedule = (
        q2_order_schedule(seed, n_rounds) if quadrant == "Q2" else None
    )

    return run_bertrand(
        config,
        agents,
        quadrant=spec.quadrant,
        ground_truth_label=spec.ground_truth_label,  # construction-time, never inferred
        backend_name="mock" if _is_mock(backend) else "real",
        agent_identities=identities,
        shared_checkpoint=spec.shared_checkpoint,
        checkpoint_id=spec.checkpoint_id,
        git_revision=git_revision,
        env_version=env_version,
        q4_policy=spec.q4_policy,
        econ=econ,
        order_schedule=order_schedule,
    )


def generate_all_quadrants(
    backend: InferenceBackend,
    *,
    seed: int,
    n_rounds: int = 25,
    temperature: float = 0.7,
    econ: Economics | None = None,
    env_version: str = "0.1.0",
    git_revision: str = "0" * 40,
) -> dict[str, Trajectory]:
    """Generate all four quadrants at a fixed task/economics/seed; return by quadrant."""
    econ = econ or Economics()
    return {
        q: generate_quadrant(
            q,
            backend,
            seed=seed,
            n_rounds=n_rounds,
            temperature=temperature,
            econ=econ,
            env_version=env_version,
            git_revision=git_revision,
        )
        for q in QUADRANT_ORDER
    }


def _is_mock(backend: InferenceBackend) -> bool:
    """Backend identity for the log: 'mock' for the local stub, 'real' otherwise."""
    return type(backend).__name__ == "MockBackend"
