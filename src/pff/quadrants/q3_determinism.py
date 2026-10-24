"""Q3 same-checkpoint determinism contract (deferred finding F1, design 5.4).

Q3's scientific purpose is a negative control in which the ONLY shared cause between the
two agents is the base-model checkpoint. The determinism claim ("same checkpoint twice is
deterministic") is checked by comparing both Q3 agents' RAW completions whenever they are
shown a byte-identical observation.

The subtlety F1 fixes: in ``stable`` name mode the round-0 observations of agents "A" and
"B" are NOT byte-identical, because each agent's prompt carries its own seller-identity
label (``agent_id`` in the ``<<PFF_CONTROL>>`` control block, and the ``rendered_observation``
repr). If we compared the raw observations we would wrongly conclude "no symmetric state
ever occurs" and the equality assertion would pass vacuously. The fix is a
**seller-identity-NEUTRALIZING** projection that strips/normalizes exactly those identity
fields, leaving everything the checkpoint actually conditions on (empty price history, grid,
neutral role). Two Q3 agents at round 0 are byte-identical UNDER THIS NEUTRALIZED VIEW, which
is the correct symmetric decision point to test determinism at.

This module is pure/GPU-free; the GPU-gated assertion lives in
``tests/test_q3_determinism_gpu.py`` and is skipped unless a real backend is wired.
"""

from __future__ import annotations

import json
import re

# The control block the agent embeds in its user message (design 5.1 / llm_agent).
_CONTROL_RE = re.compile(r"<<PFF_CONTROL (\{.*?\})>>", re.DOTALL)

# Seller-identity fields that distinguish the two Q3 agents on an OTHERWISE symmetric
# state. These are neutralized (not compared) so the determinism test is non-vacuous.
SELLER_IDENTITY_KEYS = ("agent_id",)
_NEUTRAL_ID = "<AGENT>"


def _neutralize_control_block(payload: str) -> str:
    """Return the control-block JSON with seller-identity keys normalized."""
    ctrl = json.loads(payload)
    for key in SELLER_IDENTITY_KEYS:
        if key in ctrl:
            ctrl[key] = _NEUTRAL_ID
    return "<<PFF_CONTROL " + json.dumps(ctrl, sort_keys=True) + ">>"


def neutralize_observation_text(text: str) -> str:
    """Strip seller-identity from any rendered observation / prompt text.

    Normalizes the ``agent_id`` inside an embedded ``<<PFF_CONTROL {...}>>`` block and any
    ``agent_id='A'`` / ``agent_id="B"`` token in a dataclass ``repr`` so two Q3 agents on a
    symmetric state project to byte-identical text. Idempotent and dependency-free.
    """
    # Normalize the structured control block (JSON) if present.
    def _sub_ctrl(m: re.Match[str]) -> str:
        return _neutralize_control_block(m.group(1))

    out = _CONTROL_RE.sub(_sub_ctrl, text)
    # Normalize the Observation repr's agent_id='...' token (rendered_observation).
    out = re.sub(r"agent_id=('[^']*'|\"[^\"]*\")", f"agent_id='{_NEUTRAL_ID}'", out)
    return out


def q3_observation_is_symmetric(obs_a_text: str, obs_b_text: str) -> bool:
    """True iff two Q3 observations are byte-identical after identity neutralization.

    This is the operational definition of a "symmetric decision point": the two agents
    face the same market state and differ only by seller identity. Round 0 (empty price
    history, ``stable`` mode) is GUARANTEED to satisfy this, so a run always has >= 1
    symmetric point to assert determinism at (design 5.4, finding F1).
    """
    return neutralize_observation_text(obs_a_text) == neutralize_observation_text(
        obs_b_text
    )
