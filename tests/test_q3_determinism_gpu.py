"""Q3 same-checkpoint determinism contract (F1 / AC-14 real-backend, design 5.4).

Two parts:

* **$0 unit coverage** of the seller-identity-NEUTRALIZING helper (always runs): proves the
  round-0 Q3 observations are NOT byte-identical raw (seller label differs) but ARE identical
  under neutralization, so the symmetric decision point is real and the GPU assertion cannot
  pass vacuously.

* **GPU-gated assertion** (skipped unless a real Modal vLLM endpoint is wired via
  ``PFF_MODAL_VLLM_URL``): records BOTH Q3 agents' raw completions against the SAME served
  Qwen checkpoint with the Q3 determinism contract (greedy ``temperature=0``, fixed seed,
  serialized single-request inference), compares them on the NEUTRALIZED observation, forces
  the round-0 symmetric decision point, and asserts the symmetric-trigger count >= 1 and
  byte-equality of the completions at every symmetric point (non-vacuous).

GPU ACCESS NOTE: Modal GPU functions on this account require a payment method
(``InvalidError: Please add a payment method to use <TIER> GPU functions.``), so the gated
part is skipped in the committed $0 run. Wire ``PFF_MODAL_VLLM_URL`` (and deploy
``modal_app/vllm_server.py`` with ``PFF_VLLM_Q3_DETERMINISM=1``) to exercise it.
"""

from __future__ import annotations

import os

import pytest

from pff.agents.llm_agent import LLMAgent
from pff.quadrants.config import QUADRANTS
from pff.quadrants.q3_determinism import (
    neutralize_observation_text,
    q3_observation_is_symmetric,
)

MODAL_URL = os.environ.get("PFF_MODAL_VLLM_URL")
GPU_REASON = (
    "real Modal vLLM endpoint not wired (set PFF_MODAL_VLLM_URL); Modal GPU requires a "
    "payment method on this account"
)


def _q3_round0_messages():
    """Build the round-0 user/system messages for both Q3 agents (stable mode)."""
    from pff.agents.base import Observation

    spec = QUADRANTS["Q3"]
    msgs = {}
    for aid, model in zip(("A", "B"), spec.agent_models):
        agent = LLMAgent(
            agent_id=aid,
            backend=None,  # not called here; we only build messages
            model=model,
            quadrant="Q3",
            policy=spec.policy,
            temperature=0.0,
            seed=0,
            scratchpad_on=spec.scratchpad_on,
        )
        obs = Observation(
            agent_id=aid,
            round_index=0,
            own_history=[],
            rival_history=[],
            channel_messages=[],
            grid_size=15,
            nash_idx=5,
            mono_idx=9,
            extra={"br_table": list(range(15))},
        )
        system, user = agent.build_messages(obs)
        msgs[aid] = (system, user)
    return msgs


def test_q3_round0_is_symmetric_only_after_neutralization():
    """F1: raw round-0 Q3 observations differ by seller id; neutralized they are equal."""
    msgs = _q3_round0_messages()
    user_a = msgs["A"][1]
    user_b = msgs["B"][1]
    # The two Q3 system prompts are identical (neutral role, no seller label).
    assert msgs["A"][0] == msgs["B"][0]
    # Raw user messages differ (the control block carries agent_id A vs B).
    assert user_a != user_b
    # After neutralization they are byte-identical -> a real symmetric decision point.
    assert neutralize_observation_text(user_a) == neutralize_observation_text(user_b)
    assert q3_observation_is_symmetric(user_a, user_b)


def test_neutralizer_is_idempotent_and_targets_only_identity():
    msgs = _q3_round0_messages()
    user_a = msgs["A"][1]
    once = neutralize_observation_text(user_a)
    assert neutralize_observation_text(once) == once  # idempotent
    # Neutralization changes the raw text (it carried an agent_id), but only the id token.
    assert once != user_a
    assert "<AGENT>" in once


@pytest.mark.skipif(MODAL_URL is None, reason=GPU_REASON)
def test_q3_same_checkpoint_determinism_real_backend():
    """GPU-gated: both Q3 agents' raw completions are byte-equal on symmetric states.

    Serialized, greedy (temperature=0), fixed-seed requests against the SAME served Qwen
    checkpoint. Forces the round-0 symmetric point and asserts trigger count >= 1 so the
    check is non-vacuous (design 5.4 / finding F1).
    """
    from pff.inference.modal_client import ModalVLLMBackend

    backend = ModalVLLMBackend(base_url=MODAL_URL)
    spec = QUADRANTS["Q3"]
    # Same served checkpoint for BOTH agents (the Q3 contract).
    served = spec.agent_models[0]

    msgs = _q3_round0_messages()
    symmetric_triggers = 0
    completions: dict[str, str] = {}
    for aid in ("A", "B"):  # SERIALIZED, never concurrent (design 5.4)
        system, user = msgs[aid]
        comp = backend.chat(
            served,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,  # greedy: removes sampling RNG from the path
            seed=0,
            max_tokens=16,
        )
        completions[aid] = comp.text

    # Round 0 is a guaranteed symmetric point under neutralization.
    if q3_observation_is_symmetric(msgs["A"][1], msgs["B"][1]):
        symmetric_triggers += 1
        assert completions["A"] == completions["B"], (
            "same-checkpoint Q3 agents produced different completions on a symmetric "
            f"observation: {completions['A']!r} != {completions['B']!r}"
        )

    assert symmetric_triggers >= 1, "determinism assertion was vacuous (no symmetric point)"
