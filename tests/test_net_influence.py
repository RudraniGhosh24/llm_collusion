"""AC-10: seeded pair clears the permutation threshold strictly above every other pair."""

from __future__ import annotations

import itertools

from pff.detectors.plusfactor.net_influence import net_pairwise_influence
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game


def _collect(n_agents=4, seeds=(0, 1, 2, 3, 4)):
    base = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=seeds[0]))
    coord = base.coordinating_pair
    runs = [run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=s)) for s in seeds]
    runs = [r for r in runs if r.coordinating_pair == coord]
    return coord, runs, n_agents


def test_seeded_pair_ranks_strictly_above_all_others():
    coord, runs, n = _collect()
    scores = {}
    fired = {}
    for (i, j) in itertools.combinations(range(n), 2):
        res = net_pairwise_influence(
            [r.back_half_series(i) for r in runs],
            [r.back_half_series(j) for r in runs],
            cardinality=CARDINALITY, seed=999,
        )
        scores[(i, j)] = res.net_pairwise
        fired[(i, j)] = res.fires
    coord_score = scores[coord]
    for pair, score in scores.items():
        if pair == coord:
            continue
        assert coord_score > score, f"{coord} not strictly above {pair}"
    assert fired[coord], "coordinating pair must clear the permutation threshold"


def test_low_power_flag_on_tiny_sample():
    res = net_pairwise_influence(
        [[0, 1, 0]], [[1, 0, 1]], cardinality=CARDINALITY, seed=0, n_permutations=50
    )
    assert res.low_power is True


def test_non_coordinating_pair_does_not_fire():
    coord, runs, n = _collect()
    # Find a pair disjoint from the coordinating pair.
    others = [p for p in itertools.combinations(range(n), 2)
              if not set(p) & set(coord)]
    assert others, "expected at least one fully non-coordinating pair"
    res = net_pairwise_influence(
        [r.back_half_series(others[0][0]) for r in runs],
        [r.back_half_series(others[0][1]) for r in runs],
        cardinality=CARDINALITY, seed=999,
    )
    assert res.fires is False
