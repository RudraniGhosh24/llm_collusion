"""AC-12 (logic portion): economics benchmarks recomputed, not hard-coded.

The scientific supracompetitive-pricing claim with real LLMs is GPU-gated and lives
elsewhere; this module asserts only the economics logic, which runs on the mock at $0.
"""

from __future__ import annotations

from pff.envs.economics import (
    Economics,
    find_symmetric_equilibria,
    make_price_grid,
    monopoly_index,
    select_nash_index,
)


def test_exactly_two_symmetric_equilibria():
    grid = make_price_grid()
    eq = find_symmetric_equilibria(grid)
    assert len(eq) == 2, f"expected exactly two symmetric equilibria, got {eq}"
    assert eq == [4, 5]


def test_nash_tiebreak_selects_highest_profit_index_5():
    grid = make_price_grid()
    eq = find_symmetric_equilibria(grid)
    nash = select_nash_index(eq, grid)
    assert nash == 5, "pinned tie-break must select the highest-profit equilibrium"


def test_monopoly_index_is_9_and_interior():
    grid = make_price_grid()
    mono = monopoly_index(grid)
    assert mono == 9
    assert 0 < mono < len(grid) - 1  # interior


def test_economics_recomputes_not_literal():
    econ = Economics()
    # Recompute independently and compare to the object's computed indices.
    grid = make_price_grid()
    eq = find_symmetric_equilibria(grid)
    assert econ.equilibria == tuple(eq)
    assert econ.nash_idx == select_nash_index(eq, grid)
    assert econ.mono_idx == monopoly_index(grid)
    # Benchmark normalizer is the symmetric monopoly-vs-Nash payoff gap.
    assert abs(econ.pf1_norm() - 0.08789) < 1e-4
    # Best response to a rival at Nash is Nash itself (symmetric NE property).
    assert econ.best_response_index(econ.nash_idx) == econ.nash_idx
