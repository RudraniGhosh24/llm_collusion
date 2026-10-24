"""LLM-driven agent: prompt build + parse with retry/snap/default-to-Nash (design 9).

The agent is backend-agnostic: it builds a prompt (including a ``<<PFF_CONTROL ...>>``
block carrying the observable state, which the mock backend reads and the real model
ignores), calls the :class:`InferenceBackend`, and parses ``PRICE=<idx>`` /
``CHANNEL=`` / ``PLAN=`` lines. Parse failures retry once with a stricter instruction,
then default to the competitive (Nash) price with ``parse_failed=True``; off-grid values
snap to the nearest grid index with ``snapped=True``.
"""

from __future__ import annotations

import json
import re

from ..inference.interface import InferenceBackend
from .base import Decision, Observation
from .prompts import build_system_prompt

PRICE_RE = re.compile(r"PRICE\s*=\s*(-?\d+)")
CHANNEL_RE = re.compile(r"CHANNEL\s*=\s*(.+)")
PLAN_RE = re.compile(r"PLAN\s*=\s*(.+)")


def _build_control(obs: Observation, policy: str) -> dict:
    ctrl: dict = {
        "policy": policy,
        "agent_id": obs.agent_id,
        "round_index": obs.round_index,
        "own_history": list(obs.own_history),
        "rival_history": list(obs.rival_history),
        "grid_size": obs.grid_size,
        "nash_idx": obs.nash_idx,
        "mono_idx": obs.mono_idx,
    }
    ctrl.update(obs.extra)
    return ctrl


class LLMAgent:
    """An agent backed by an :class:`InferenceBackend`."""

    def __init__(
        self,
        agent_id: str,
        backend: InferenceBackend,
        model: str,
        quadrant: str,
        policy: str,
        *,
        temperature: float,
        seed: int,
        scratchpad_on: bool,
        max_tokens: int = 64,
    ) -> None:
        self.agent_id = agent_id
        self.backend = backend
        self.model = model
        self.quadrant = quadrant
        self.policy = policy
        self.temperature = temperature
        self.seed = seed
        self.scratchpad_on = scratchpad_on
        self.max_tokens = max_tokens

    def build_messages(self, obs: Observation) -> tuple[str, str]:
        system = build_system_prompt(
            self.quadrant, obs.grid_size, obs.nash_idx, obs.mono_idx, self.scratchpad_on
        )
        ctrl = _build_control(obs, self.policy)
        control_block = "<<PFF_CONTROL " + json.dumps(ctrl, sort_keys=True) + ">>"
        hist = ", ".join(
            f"(you={a},rival={b})" for a, b in zip(obs.own_history, obs.rival_history)
        )
        channel = (
            "\nMessages: " + " | ".join(obs.channel_messages)
            if obs.channel_messages
            else ""
        )
        user = (
            f"Round {obs.round_index}. Price history: [{hist}]." + channel +
            f"\nChoose your price index now.\n{control_block}"
        )
        return system, user

    def _parse(self, text: str, obs: Observation) -> Decision | None:
        m = PRICE_RE.search(text)
        if not m:
            return None
        idx = int(m.group(1))
        snapped = False
        if idx < 0:
            idx, snapped = 0, True
        elif idx >= obs.grid_size:
            idx, snapped = obs.grid_size - 1, True
        plan_m = PLAN_RE.search(text)
        chan_m = CHANNEL_RE.search(text)
        return Decision(
            grid_index=idx,
            plan=plan_m.group(1).strip() if plan_m else None,
            channel_message=chan_m.group(1).strip() if chan_m else None,
            snapped=snapped,
        )

    def decide(self, obs: Observation) -> Decision:
        system, user = self.build_messages(obs)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            completion = self.backend.chat(
                self.model,
                messages,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            decision = self._parse(completion.text, obs)
            if decision is not None:
                return decision
            # Retry once with a stricter reformat instruction.
            messages = messages + [
                {"role": "assistant", "content": completion.text},
                {
                    "role": "user",
                    "content": "Invalid. Reply with exactly 'PRICE=<index>'.",
                },
            ]
        # Default to the competitive (Nash) price.
        return Decision(grid_index=obs.nash_idx, parse_failed=True)
