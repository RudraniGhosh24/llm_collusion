"""AC-6: serialize -> reload -> validate; payoffs recompute bit-identically."""

from __future__ import annotations

from pff.agents.llm_agent import LLMAgent
from pff.envs.bertrand import BertrandConfig, run_bertrand
from pff.envs.economics import Economics
from pff.inference.mock_backend import MockBackend
from pff.logging.logger import read_trajectories, write_trajectories
from pff.logging.schema import AgentIdentity


def _make_traj():
    econ = Economics()
    be = MockBackend()
    cfg = BertrandConfig(channel_on=False, scratchpad_on=True, n_rounds=8, seed=3)
    agents = [
        LLMAgent("A", be, "mock/Qwen2.5-7B-Instruct", "Q4", "q4_indep",
                 temperature=cfg.temperature, seed=cfg.seed, scratchpad_on=True),
        LLMAgent("B", be, "mock/Llama-3.1-8B-Instruct", "Q4", "q4_indep",
                 temperature=cfg.temperature, seed=cfg.seed, scratchpad_on=True),
    ]
    idents = [
        AgentIdentity(agent_id="A", family="qwen", checkpoint="mock/Qwen2.5-7B-Instruct",
                      temperature=cfg.temperature, seed=cfg.seed),
        AgentIdentity(agent_id="B", family="llama", checkpoint="mock/Llama-3.1-8B-Instruct",
                      temperature=cfg.temperature, seed=cfg.seed),
    ]
    return run_bertrand(
        cfg, agents, quadrant="Q4", ground_truth_label="NOT_COLLUDING",
        backend_name="mock", agent_identities=idents, shared_checkpoint=False,
        checkpoint_id="mixed", q4_policy="indep_best_response", econ=econ,
    )


def test_roundtrip_validates(tmp_path):
    traj = _make_traj()
    path = write_trajectories(tmp_path / "t.trajectories.jsonl", [traj])
    reloaded = read_trajectories(path)
    assert len(reloaded) == 1
    assert reloaded[0].meta.quadrant == "Q4"


def test_payoffs_recompute_bit_identically():
    econ = Economics()
    traj = _make_traj()
    for step in traj.steps:
        for aid, logged in step.payoffs.items():
            prices = step.payoff_inputs["prices"]
            rival = [x for x in prices if x != aid][0]
            recomputed = econ.payoff(prices[aid], prices[rival])
            assert recomputed == logged, f"payoff mismatch for {aid} at round {step.round_index}"
