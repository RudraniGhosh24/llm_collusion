"""AC-14 / FR-B (CORRECTNESS RISK B): Q3 is a leak-free negative control.

Q3's only shared cause is the base-model checkpoint. These tests assert that nothing in
either Q3 agent's prompt (system message, user message, scratchpad-seed affordance)
hints at coordination, that the Q3 templates are a strict subset of the Q1/Q2 templates
(Q3 = Q1/Q2 minus the incentive + channel blocks, with NO Q3-unique additions), and that
the two Q3 agents differ only by ``agent_id`` (identical checkpoint + decoding params).
"""

from __future__ import annotations

from pff.agents.prompts import (
    CHANNEL_AFFORDANCE,
    INCENTIVE_BLOCK,
    SCRATCHPAD_AFFORDANCE,
    build_system_prompt,
)
from pff.inference.mock_backend import MockBackend
from pff.quadrants.config import BANNED_PHRASES
from pff.quadrants.generator import generate_quadrant

SEED = 3
N_ROUNDS = 25
GRID = 15
NASH = 5
MONO = 9


def _assert_no_banned(text: str, where: str) -> None:
    low = (text or "").lower()
    for phrase in BANNED_PHRASES:
        assert phrase.lower() not in low, f"banned phrase {phrase!r} leaked into {where}"


def test_q3_prompts_contain_no_banned_phrases():
    be = MockBackend()
    q3 = generate_quadrant("Q3", be, seed=SEED, n_rounds=N_ROUNDS)
    assert q3.meta.quadrant == "Q3"
    for step in q3.steps:
        for agent_id, rec in step.prompts.items():
            _assert_no_banned(rec.system, f"Q3 {agent_id} system round {step.round_index}")
            _assert_no_banned(rec.user, f"Q3 {agent_id} user round {step.round_index}")
        # scratchpad text logged in private_plans must also be clean
        for agent_id, plan in (step.private_plans or {}).items():
            _assert_no_banned(plan, f"Q3 {agent_id} plan round {step.round_index}")


def test_q3_scratchpad_affordance_is_neutral():
    # The scratchpad seed affordance itself (the only planning text Q3 adds) is clean.
    _assert_no_banned(SCRATCHPAD_AFFORDANCE, "SCRATCHPAD_AFFORDANCE")
    q3_sys = build_system_prompt("Q3", GRID, NASH, MONO, scratchpad_on=True)
    assert SCRATCHPAD_AFFORDANCE in q3_sys
    _assert_no_banned(q3_sys, "Q3 system template")


def test_q3_template_is_strict_subset_of_q1_and_q2():
    q1 = build_system_prompt("Q1", GRID, NASH, MONO, scratchpad_on=True)
    q2 = build_system_prompt("Q2", GRID, NASH, MONO, scratchpad_on=True)
    q3 = build_system_prompt("Q3", GRID, NASH, MONO, scratchpad_on=True)

    q1_lines = q1.split("\n")
    q2_lines = q2.split("\n")
    q3_lines = q3.split("\n")

    # Every Q3 line also appears in Q1 and Q2 (no Q3-unique additions).
    for line in q3_lines:
        assert line in q1_lines, f"Q3 line not present in Q1: {line!r}"
        assert line in q2_lines, f"Q3 line not present in Q2: {line!r}"

    # The ONLY differences are the removed incentive (+ channel for Q1) blocks.
    q1_extra = [ln for ln in q1_lines if ln not in q3_lines]
    q2_extra = [ln for ln in q2_lines if ln not in q3_lines]
    assert q1_extra == [INCENTIVE_BLOCK, CHANNEL_AFFORDANCE]
    assert q2_extra == [INCENTIVE_BLOCK]

    # The incentive/channel blocks are exactly what the leak list is meant to catch.
    _assert_no_banned("\n".join(q3_lines), "Q3 template")


def test_q3_both_agents_differ_only_by_agent_id():
    be = MockBackend()
    q3 = generate_quadrant("Q3", be, seed=SEED, n_rounds=N_ROUNDS)
    assert q3.meta.shared_checkpoint is True
    a, b = q3.meta.agents
    assert {a.agent_id, b.agent_id} == {"A", "B"}
    # Same checkpoint, same decoding params (temperature + seed), same family.
    assert a.checkpoint == b.checkpoint
    assert a.family == b.family
    assert a.temperature == b.temperature
    assert a.seed == b.seed
    # The single shared checkpoint id is recorded once at run level too.
    assert q3.meta.checkpoint_id == a.checkpoint


def test_q3_system_prompt_identical_across_both_agents():
    be = MockBackend()
    q3 = generate_quadrant("Q3", be, seed=SEED, n_rounds=N_ROUNDS)
    for step in q3.steps:
        systems = {rec.system for rec in step.prompts.values()}
        # Same quadrant + same scratchpad knob -> identical system prompt for both.
        assert len(systems) == 1
