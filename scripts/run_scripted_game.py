"""Phase 0 / GATE 1 (design 11): scripted abstract-game run at $0, no LLM, no GPU.

Runs the N-agent iterated matrix game in scripted mode over several seeds, pools the
back-half action series per pair, computes net pairwise influence, prints the ranking,
and exits 0 only if the seeded coordinating pair ranks strictly above every
non-coordinating pair (and clears the permutation significance threshold).
"""

from __future__ import annotations

import itertools
import sys

from pff.detectors.plusfactor.net_influence import net_pairwise_influence
from pff.envs.matrix_game import CARDINALITY, MatrixGameConfig, run_matrix_game


def main() -> int:
    seeds = [0, 1, 2, 3, 4]
    n_agents = 4

    # All seeds share the same coordinating pair so pooling is coherent; we fix the pair
    # by using the same base seed for pair selection while varying the play RNG via the
    # per-agent streams (which already incorporate the seed). To guarantee a stable pair
    # across pooled seeds, select the pair from seed 0 and run each seed with it.
    base = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=0))
    coord = base.coordinating_pair

    # Collect per-seed back-half series, keeping only runs with the same coord pair.
    results = []
    for s in seeds:
        r = run_matrix_game(MatrixGameConfig(n_agents=n_agents, seed=s))
        if r.coordinating_pair == coord:
            results.append(r)
    if not results:
        print("No runs matched the base coordinating pair; aborting.", file=sys.stderr)
        return 1

    pairs = list(itertools.combinations(range(n_agents), 2))
    scores: dict[tuple[int, int], float] = {}
    fired: dict[tuple[int, int], bool] = {}
    for (i, j) in pairs:
        series_i = [r.back_half_series(i) for r in results]
        series_j = [r.back_half_series(j) for r in results]
        res = net_pairwise_influence(
            series_i, series_j, cardinality=CARDINALITY, seed=12345
        )
        scores[(i, j)] = res.net_pairwise
        fired[(i, j)] = res.fires

    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    print(f"Seeded coordinating pair (ground truth): {coord}")
    print("Net pairwise influence ranking (desc):")
    for (pair, score) in ranking:
        tag = "  <-- coordinating" if pair == coord else ""
        fire = "FIRES" if fired[pair] else "null "
        print(f"  pair {pair}: net_influence={score:.5f}  [{fire}]{tag}")

    coord_score = scores[coord]
    others = [v for k, v in scores.items() if k != coord]
    strictly_above = all(coord_score > o for o in others)
    coord_fires = fired[coord]

    if strictly_above and coord_fires:
        print("\nGATE 1 PASS: seeded pair ranks strictly above all non-colluding pairs "
              "and clears the permutation threshold.")
        return 0
    print("\nGATE 1 FAIL: seeded pair did not rank strictly above all others "
          f"(strictly_above={strictly_above}, coord_fires={coord_fires}).",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
