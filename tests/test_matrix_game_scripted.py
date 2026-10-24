"""AC-2: scripted abstract game, deterministic, exposes a seed-chosen coordinating pair."""

from __future__ import annotations

import numpy as np

from pff.envs.matrix_game import MatrixGameConfig, run_matrix_game


def test_scripted_run_is_deterministic():
    cfg = MatrixGameConfig(n_agents=4, n_rounds=40, seed=11)
    a = run_matrix_game(cfg)
    b = run_matrix_game(cfg)
    assert np.array_equal(a.actions, b.actions)
    assert np.array_equal(a.payoffs, b.payoffs)
    assert a.coordinating_pair == b.coordinating_pair


def test_coordinating_pair_exposed_and_valid():
    r = run_matrix_game(MatrixGameConfig(n_agents=4, seed=0))
    i, j = r.coordinating_pair
    assert 0 <= i < j < 4


def test_actions_are_binary():
    r = run_matrix_game(MatrixGameConfig(n_agents=4, n_rounds=30, seed=3))
    assert set(np.unique(r.actions)).issubset({0, 1})


def test_coordinating_pair_is_coupled():
    # After round 0, each coupled agent mirrors the partner's previous action except on
    # exploration; so agreement with the partner's lagged action should be high.
    cfg = MatrixGameConfig(n_agents=4, n_rounds=80, seed=4, couple_explore_prob=0.15)
    r = run_matrix_game(cfg)
    i, j = r.coordinating_pair
    si, sj = r.action_series(i), r.action_series(j)
    # i_next vs j_prev agreement
    agree = np.mean(si[1:] == sj[:-1])
    assert agree > 0.7
