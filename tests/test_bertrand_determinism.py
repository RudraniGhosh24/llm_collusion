"""AC-1: two mock runs byte-identical over the determinism projection; knobs settable."""

from __future__ import annotations

from pff.agents.llm_agent import LLMAgent
from pff.envs.bertrand import BertrandConfig, run_bertrand
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import determinism_projection
from pff.logging.schema import AgentIdentity


def _run(cfg: BertrandConfig, quadrant="Q4", policy="q4_indep", channel=False):
    be = MockBackend()
    agents = [
        LLMAgent("A", be, "mock/Qwen2.5-7B-Instruct", quadrant, policy,
                 temperature=cfg.temperature, seed=cfg.seed, scratchpad_on=cfg.scratchpad_on),
        LLMAgent("B", be, "mock/Llama-3.1-8B-Instruct", quadrant, policy,
                 temperature=cfg.temperature, seed=cfg.seed, scratchpad_on=cfg.scratchpad_on),
    ]
    idents = [
        AgentIdentity(agent_id="A", family="qwen", checkpoint="m", temperature=cfg.temperature, seed=cfg.seed),
        AgentIdentity(agent_id="B", family="llama", checkpoint="m", temperature=cfg.temperature, seed=cfg.seed),
    ]
    return run_bertrand(
        cfg, agents, quadrant=quadrant, ground_truth_label="NOT_COLLUDING",
        backend_name="mock", agent_identities=idents, shared_checkpoint=False,
        checkpoint_id="m", git_revision="deadbeef",
    )


def test_two_mock_runs_byte_identical_projection():
    cfg = BertrandConfig(n_rounds=10, seed=5)
    a = determinism_projection(_run(cfg))
    b = determinism_projection(_run(cfg))
    assert a == b


def test_channel_knob_changes_logged_behavior():
    on = _run(BertrandConfig(n_rounds=6, seed=1, channel_on=True), quadrant="Q1",
              policy="q1_channel", channel=True)
    off = _run(BertrandConfig(n_rounds=6, seed=1, channel_on=False), quadrant="Q1",
               policy="q1_channel")
    assert any(s.channel_messages for s in on.steps)
    assert all(s.channel_messages is None for s in off.steps)


def test_scratchpad_knob_changes_logged_behavior():
    on = _run(BertrandConfig(n_rounds=6, seed=1, scratchpad_on=True))
    off = _run(BertrandConfig(n_rounds=6, seed=1, scratchpad_on=False))
    assert all(s.private_plans is not None for s in on.steps)
    assert all(s.private_plans is None for s in off.steps)


def test_name_mode_knob_changes_order():
    stable = _run(BertrandConfig(n_rounds=12, seed=2, name_mode="stable"))
    shuffled = _run(BertrandConfig(n_rounds=12, seed=2, name_mode="shuffled"))
    stable_orders = {tuple(s.action_order) for s in stable.steps}
    shuffled_orders = {tuple(s.action_order) for s in shuffled.steps}
    assert len(stable_orders) == 1  # fixed order
    assert len(shuffled_orders) == 2  # both orders occur


def test_temperature_and_meta_recorded():
    traj = _run(BertrandConfig(n_rounds=4, seed=1, temperature=0.3))
    assert traj.meta.knobs.temperature == 0.3
    assert traj.meta.knobs.incentive_to_collude_on is False


def test_incentive_knob_recorded():
    traj = _run(BertrandConfig(n_rounds=4, seed=1, incentive_to_collude_on=True))
    assert traj.meta.knobs.incentive_to_collude_on is True
