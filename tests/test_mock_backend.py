"""AC-4: deterministic mock backend, interface parity, per-quadrant policies, judge."""

from __future__ import annotations

import json

from pff.inference.mock_backend import MockBackend


def _ctrl(**kw) -> list[dict]:
    body = json.dumps(kw)
    system = f"system text <<PFF_CONTROL {body}>>"
    return [{"role": "system", "content": system}, {"role": "user", "content": "go"}]


def test_determinism_identical_inputs_identical_output():
    be = MockBackend()
    msgs = _ctrl(policy="q4_indep", grid_size=15, mono_idx=9, nash_idx=5,
                 own_history=[5], rival_history=[9], br_table=list(range(15)))
    c1 = be.chat("mock/Qwen2.5-7B-Instruct", msgs, temperature=0.7, seed=1, max_tokens=16)
    c2 = be.chat("mock/Qwen2.5-7B-Instruct", msgs, temperature=0.7, seed=1, max_tokens=16)
    assert c1.text == c2.text
    assert c1.seed == 1 and c1.temperature == 0.7


def test_available_models():
    assert MockBackend().available_models()


def test_action_text_is_parseable():
    be = MockBackend()
    msgs = _ctrl(policy="q1_channel", grid_size=15, mono_idx=9, nash_idx=5,
                 own_history=[5], rival_history=[5])
    out = be.chat("mock/Qwen2.5-7B-Instruct", msgs, temperature=0.7, seed=0, max_tokens=16)
    assert "PRICE=" in out.text
    assert "CHANNEL=" in out.text  # Q1 emits an encoded channel message


def test_q3_shared_prior_same_observation_same_action():
    be = MockBackend()
    # Two agents, same model/seed, same observation -> identical action (parallelism).
    common = dict(policy="q3_shared", grid_size=15, mono_idx=9, nash_idx=5,
                  own_history=[5, 6], rival_history=[5, 6])
    a = be.chat("mock/Qwen2.5-7B-Instruct", _ctrl(agent_id="A", **common),
                temperature=0.0, seed=7, max_tokens=16)
    b = be.chat("mock/Qwen2.5-7B-Instruct", _ctrl(agent_id="B", **common),
                temperature=0.0, seed=7, max_tokens=16)
    assert a.text.split("\n")[0] == b.text.split("\n")[0]


def test_judge_monotone_in_correlation():
    be = MockBackend()
    # Perfectly correlated price series -> high score; flat -> ~0.5; anti -> low.
    hi = be.chat("mock/judge", _ctrl(policy="judge",
                 price_series_i=[1, 2, 3, 4], price_series_j=[1, 2, 3, 4]),
                 temperature=0.0, seed=0, max_tokens=16)
    anti = be.chat("mock/judge", _ctrl(policy="judge",
                   price_series_i=[1, 2, 3, 4], price_series_j=[4, 3, 2, 1]),
                   temperature=0.0, seed=0, max_tokens=16)
    hi_score = float(hi.text.split("=")[1])
    anti_score = float(anti.text.split("=")[1])
    assert hi_score > anti_score
    assert 0.0 <= anti_score <= hi_score <= 1.0


def test_judge_zero_variance_is_nan_safe():
    be = MockBackend()
    flat = be.chat("mock/judge", _ctrl(policy="judge",
                   price_series_i=[3, 3, 3, 3], price_series_j=[3, 3, 3, 3]),
                   temperature=0.0, seed=0, max_tokens=16)
    score = float(flat.text.split("=")[1])
    # corr defined as 0 -> score = 0.5, never NaN.
    assert abs(score - 0.5) < 1e-9
