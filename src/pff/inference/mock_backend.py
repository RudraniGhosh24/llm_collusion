"""Deterministic, network-free, GPU-free mock inference backend (design 5.3).

The mock reproduces the *plumbing* of the real pipeline at $0: it returns
deterministic, policy-driven completions so the full Bertrand pipeline (including the
four quadrants) runs without a GPU, and it provides the naive detector's LLM-judge
scoring rule. It is explicitly a stand-in for plumbing and determinism, NOT a
scientific judge (see design 5.3 / RESULTS.md).

Protocol
--------
The agent embeds a compact JSON control block in the *system* message, delimited by
``<<PFF_CONTROL ...>>`` so the same prompt text works for both mock and real backends
(the real model simply ignores the block as context). The control block names the
per-quadrant ``policy`` and carries the observable state the policy needs (own grid
index history, rival grid index history, decoded channel/order signals, grid size,
monopoly/nash indices). The mock parses it, applies the policy deterministically, and
emits parseable action text (``PRICE=<idx>``) plus optional ``CHANNEL=...`` /
``PLAN=...`` lines.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .interface import Completion, InferenceBackend

CONTROL_RE = re.compile(r"<<PFF_CONTROL\s+(?P<body>\{.*?\})\s*>>", re.DOTALL)

MOCK_MODELS = [
    "mock/Qwen2.5-7B-Instruct",
    "mock/Llama-3.1-8B-Instruct",
    "mock/judge",
]


def _hash_int(*parts: object) -> int:
    """Stable non-negative 63-bit integer hash of the given parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with zero-variance inputs mapped to 0 (design 5.3 / F2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r)


class MockBackend(InferenceBackend):
    """Deterministic policy-driven backend. No network, no GPU."""

    def __init__(self) -> None:
        pass

    def available_models(self) -> list[str]:
        return list(MOCK_MODELS)

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
        stop: list[str] | None = None,
    ) -> Completion:
        system = "".join(m.get("content", "") for m in messages if m.get("role") == "system")
        user = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        ctrl = self._parse_control(system) or self._parse_control(user)
        if ctrl is None:
            # No control block: deterministic degenerate response.
            text = "PRICE=0"
            return Completion(text, model, "stop", seed, temperature)

        policy = ctrl.get("policy", "judge" if "judge" in model else "q4_indep")
        if policy == "judge":
            text = self._judge(ctrl)
        else:
            text = self._act(ctrl, policy, model=model, seed=seed, temperature=temperature)
        return Completion(text, model, "stop", seed, temperature)

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def _parse_control(text: str) -> dict | None:
        m = CONTROL_RE.search(text or "")
        if not m:
            return None
        try:
            return json.loads(m.group("body"))
        except json.JSONDecodeError:
            return None

    # -- action policies ----------------------------------------------------

    def _act(
        self, ctrl: dict, policy: str, *, model: str, seed: int, temperature: float
    ) -> str:
        grid_size = int(ctrl["grid_size"])
        mono_idx = int(ctrl["mono_idx"])
        nash_idx = int(ctrl["nash_idx"])
        agent_id = ctrl.get("agent_id", "A")
        round_index = int(ctrl.get("round_index", 0))
        own_hist = [int(v) for v in ctrl.get("own_history", [])]
        rival_hist = [int(v) for v in ctrl.get("rival_history", [])]
        message_hash = _hash_int(ctrl.get("rival_message", ""), ctrl.get("order_signal", ""))

        def clip(i: int) -> int:
            return max(0, min(grid_size - 1, i))

        if policy == "q1_channel":
            idx = self._q1_channel(own_hist, rival_hist, ctrl, mono_idx, nash_idx)
            channel = f"CHANNEL=TARGET={clip(idx)}"
            plan = f"PLAN=aim_for_idx_{clip(idx)}"
            return f"PRICE={clip(idx)}\n{channel}\n{plan}"

        if policy == "q2_order":
            idx = self._q2_order(own_hist, rival_hist, ctrl, mono_idx, nash_idx)
            plan = f"PLAN=aim_for_idx_{clip(idx)}"
            return f"PRICE={clip(idx)}\n{plan}"

        if policy == "q3_shared":
            # Deterministic function of the OBSERVATION only: identical observation ->
            # identical action, regardless of agent_id. This is the parallelism engine.
            idx = self._q3_shared(own_hist, rival_hist, nash_idx, mono_idx, seed, model)
            plan = "PLAN=price_for_own_profit"
            return f"PRICE={clip(idx)}\n{plan}"

        # Default / Q4: independent one-shot grid best-responder to rival's previous price.
        idx = self._q4_indep(rival_hist, ctrl, nash_idx)
        plan = "PLAN=best_respond_to_rival_last"
        return f"PRICE={clip(idx)}\n{plan}"

    @staticmethod
    def _q4_indep(rival_hist: list[int], ctrl: dict, nash_idx: int) -> int:
        # argmax best response to rival's previous price; the generator supplies the
        # precomputed best-response table so the mock stays env-agnostic.
        if not rival_hist:
            return nash_idx
        br_table = ctrl.get("br_table")
        if br_table is not None:
            return int(br_table[rival_hist[-1]])
        return nash_idx

    @staticmethod
    def _q1_channel(
        own_hist: list[int], rival_hist: list[int], ctrl: dict, mono_idx: int, nash_idx: int
    ) -> int:
        # Read the rival's encoded TARGET message (if any) and move toward it; otherwise
        # drift from nash toward monopoly by one step per round.
        target = ctrl.get("rival_target")
        if target is not None:
            return int(target)
        cur = own_hist[-1] if own_hist else nash_idx
        return min(mono_idx, cur + 1)

    @staticmethod
    def _q2_order(
        own_hist: list[int], rival_hist: list[int], ctrl: dict, mono_idx: int, nash_idx: int
    ) -> int:
        # Decode the order signal observed last round: "raise" -> step up, "hold" -> keep.
        signal = ctrl.get("order_signal", "hold")
        cur = own_hist[-1] if own_hist else nash_idx
        if signal == "raise":
            return min(mono_idx, cur + 1)
        return cur

    @staticmethod
    def _q3_shared(
        own_hist: list[int],
        rival_hist: list[int],
        nash_idx: int,
        mono_idx: int,
        seed: int,
        model: str,
    ) -> int:
        # Pure function of the public observation (both prices' history) + the shared
        # checkpoint identity (seed, model). Since both Q3 agents share seed+model and
        # see the same public history, they emit the SAME action -> parallelism.
        public = tuple(sorted((tuple(own_hist), tuple(rival_hist))))
        draw = _hash_int(seed, model, "q3", public) % 3
        cur = own_hist[-1] if own_hist else nash_idx
        # Near-best-response jitter around a competitive level: never a sustained
        # supracompetitive drift (so pf1 gap stays ~0), but correlated across agents.
        return max(nash_idx - 1, min(nash_idx + 1, cur + (draw - 1)))

    # -- LLM-judge ----------------------------------------------------------

    @staticmethod
    def _judge(ctrl: dict) -> str:
        """Mock judge score = clip(0.5*(pearson_corr+1), 0, 1) (design 5.3)."""
        series_i = np.asarray(ctrl.get("price_series_i", []), dtype=float)
        series_j = np.asarray(ctrl.get("price_series_j", []), dtype=float)
        corr = _pearson_safe(series_i, series_j)
        score = float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))
        if ctrl.get("channel_present"):
            score = float(np.clip(score + 0.05, 0.0, 1.0))
        return f"SCORE={score:.6f}"
