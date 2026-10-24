"""Abstract discrete-game environment (FR-2, design 3).

An N-agent iterated Prisoner's-Dilemma-style game used to validate the net-influence
statistic against known ground truth before trusting it on the LLM market.

- Actions: {0=cooperate (C), 1=defect (D)} (C = high-price analog).
- Payoffs: pairwise-summed symmetric PD bimatrix across opponents.
- Exactly one seed-chosen unordered coordinating pair (i*, j*) runs COUPLED reciprocal
  policies (each conditions on the other's previous action, tit-for-tat); all other
  agents play independent policies (action depends only on own seed/history, never on a
  rival). Ground truth is {(i*, j*)}.
- Scripted ($0) mode: every policy is a pure Python function, no backend call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Symmetric 2-player PD payoff: payoff[my_action][their_action].
# 0 = cooperate, 1 = defect. Standard PD ordering T > R > P > S.
_PD = np.array(
    [
        [3.0, 0.0],  # I cooperate: (both C -> R=3), (I C, they D -> S=0)
        [5.0, 1.0],  # I defect:   (I D, they C -> T=5), (both D -> P=1)
    ]
)

CARDINALITY = 2  # actions C/D


@dataclass
class MatrixGameConfig:
    n_agents: int = 4
    n_rounds: int = 60
    seed: int = 0
    # Independent agents' cooperation probability each round (i.i.d., no coupling).
    indep_coop_prob: float = 0.5
    # Probability the coupled pair "explores" (flips) instead of mirroring the partner.
    # Nonzero exploration gives the coupled series variation so the coupling is
    # statistically measurable; the mirror still makes each agent's next action track
    # the PARTNER's previous action (the detectable conditional dependence).
    couple_explore_prob: float = 0.15


@dataclass
class MatrixGameResult:
    actions: np.ndarray  # shape (n_rounds, n_agents), values in {0,1}
    payoffs: np.ndarray  # shape (n_rounds, n_agents)
    coordinating_pair: tuple[int, int]
    config: MatrixGameConfig = field(repr=False)

    def action_series(self, agent: int) -> np.ndarray:
        return self.actions[:, agent]

    def back_half_series(self, agent: int) -> np.ndarray:
        half = self.actions.shape[0] // 2
        return self.actions[half:, agent]


def choose_coordinating_pair(n_agents: int, rng: np.random.Generator) -> tuple[int, int]:
    """Deterministically choose one unordered coordinating pair from the seed."""
    if n_agents < 2:
        raise ValueError("need at least 2 agents")
    i, j = rng.choice(n_agents, size=2, replace=False)
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)


def run_matrix_game(config: MatrixGameConfig) -> MatrixGameResult:
    """Run the scripted ($0) iterated game and return actions/payoffs + ground truth."""
    rng = np.random.default_rng(config.seed)
    n = config.n_agents
    i_star, j_star = choose_coordinating_pair(n, rng)

    # Per-agent independent RNG streams (deterministic, no global state).
    streams = [np.random.default_rng(config.seed * 1000 + a + 1) for a in range(n)]

    actions = np.zeros((config.n_rounds, n), dtype=int)
    prev = np.zeros(n, dtype=int)  # previous-round actions; round 0 seeded below

    for t in range(config.n_rounds):
        cur = np.zeros(n, dtype=int)
        for a in range(n):
            if a in (i_star, j_star):
                partner = j_star if a == i_star else i_star
                if t == 0:
                    # Seed round-0 actions from the agent's own stream so the coupled
                    # series is not frozen at a constant from the start.
                    cur[a] = int(streams[a].random() > 0.5)
                else:
                    # Coupled tit-for-tat: mirror the PARTNER's previous action, with
                    # occasional exploration so the series carries variation.
                    mirror = prev[partner]
                    if streams[a].random() < config.couple_explore_prob:
                        cur[a] = 1 - mirror
                    else:
                        cur[a] = mirror
            else:
                # Independent: i.i.d. draw, no dependence on any rival.
                cur[a] = int(streams[a].random() > config.indep_coop_prob)
        actions[t] = cur
        prev = cur

    # Payoffs: pairwise-summed across opponents.
    payoffs = np.zeros((config.n_rounds, n))
    for t in range(config.n_rounds):
        for a in range(n):
            tot = 0.0
            for b in range(n):
                if b == a:
                    continue
                tot += _PD[actions[t, a], actions[t, b]]
            payoffs[t, a] = tot

    return MatrixGameResult(
        actions=actions,
        payoffs=payoffs,
        coordinating_pair=(i_star, j_star),
        config=config,
    )
