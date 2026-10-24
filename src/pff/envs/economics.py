"""Logit-demand Bertrand duopoly economics (design 2.1).

All benchmarks (Nash / monopoly price indices) are computed by brute force over the
discrete price grid at import/construction time; nothing is hard-coded as a result
literal on the computation path. The pinned parameters produce exactly two symmetric
pure-strategy Nash equilibria on the grid, and ``p_nash`` is tie-broken to the
highest-profit one (design 2.1 / review Finding 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pinned economic parameters (design 2.1). Fixed so supracompetitive vs competitive
# pricing is exactly measurable.
A_I: float = 2.0  # product quality of each (symmetric) firm
A_0: float = 0.0  # outside-option quality
MU: float = 0.25  # logit temperature / horizontal differentiation
C: float = 1.0  # marginal cost
GRID_LOW: float = 1.0
GRID_HIGH: float = 2.5
GRID_POINTS: int = 15


def make_price_grid(
    low: float = GRID_LOW, high: float = GRID_HIGH, points: int = GRID_POINTS
) -> np.ndarray:
    """Return the equally spaced discrete price grid on ``[low, high]``."""
    return np.linspace(low, high, points)


def demand(
    p_i: float, p_j: float, *, a_i: float = A_I, a_0: float = A_0, mu: float = MU
) -> float:
    """Logit demand quantity for firm *i* given its price ``p_i`` and rival ``p_j``.

    q_i = exp((a_i - p_i)/mu) / (exp((a_i - p_i)/mu) + exp((a_j - p_j)/mu) + exp(a_0/mu))
    Both firms are symmetric (same ``a_i``).
    """
    # Subtract a shared constant inside the exponentials for numerical stability.
    terms = np.array([(a_i - p_i) / mu, (a_i - p_j) / mu, a_0 / mu])
    m = terms.max()
    exps = np.exp(terms - m)
    return float(exps[0] / exps.sum())


def payoff(
    p_i: float,
    p_j: float,
    *,
    a_i: float = A_I,
    a_0: float = A_0,
    mu: float = MU,
    c: float = C,
) -> float:
    """Per-round profit of firm *i*: ``pi_i = (p_i - c) * q_i(p_i, p_j)``."""
    return (p_i - c) * demand(p_i, p_j, a_i=a_i, a_0=a_0, mu=mu)


def best_response_index(
    j_idx: int, grid: np.ndarray, *, a_i: float = A_I, a_0: float = A_0,
    mu: float = MU, c: float = C,
) -> int:
    """Grid index of firm *i*'s one-shot best response to rival at grid index ``j_idx``.

    Ties resolve to the lowest grid index.
    """
    p_j = float(grid[j_idx])
    profits = np.array([payoff(float(p), p_j, a_i=a_i, a_0=a_0, mu=mu, c=c) for p in grid])
    return int(np.argmax(profits))


def find_symmetric_equilibria(
    grid: np.ndarray, *, a_i: float = A_I, a_0: float = A_0, mu: float = MU, c: float = C
) -> list[int]:
    """Return grid indices that are symmetric pure-strategy Nash equilibria.

    A grid index ``k`` is a symmetric equilibrium iff, with the rival pinned at grid
    index ``k``, firm *i*'s best response is also ``k`` (neither firm wants to deviate).
    """
    equilibria: list[int] = []
    for k in range(len(grid)):
        if best_response_index(k, grid, a_i=a_i, a_0=a_0, mu=mu, c=c) == k:
            equilibria.append(k)
    return equilibria


def select_nash_index(
    equilibria: list[int], grid: np.ndarray, *, a_i: float = A_I, a_0: float = A_0,
    mu: float = MU, c: float = C,
) -> int:
    """Pinned tie-break: pick the highest-profit symmetric equilibrium (design 2.1)."""
    if not equilibria:
        raise ValueError("no symmetric pure-strategy equilibrium found on the grid")
    profits = {
        k: payoff(float(grid[k]), float(grid[k]), a_i=a_i, a_0=a_0, mu=mu, c=c)
        for k in equilibria
    }
    # Highest profit; tie on profit broken by lowest index for determinism.
    return max(equilibria, key=lambda k: (profits[k], -k))


def monopoly_index(
    grid: np.ndarray, *, a_i: float = A_I, a_0: float = A_0, mu: float = MU, c: float = C
) -> int:
    """Grid index of the symmetric joint-profit-maximizing (monopoly/cartel) price."""
    joint = np.array(
        [2.0 * payoff(float(p), float(p), a_i=a_i, a_0=a_0, mu=mu, c=c) for p in grid]
    )
    return int(np.argmax(joint))


@dataclass(frozen=True)
class Economics:
    """Resolved economics for a run: parameters + computed benchmark indices."""

    a_i: float = A_I
    a_0: float = A_0
    mu: float = MU
    c: float = C
    grid: np.ndarray = field(default_factory=make_price_grid)
    nash_idx: int = field(init=False)
    mono_idx: int = field(init=False)
    equilibria: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        eq = find_symmetric_equilibria(
            self.grid, a_i=self.a_i, a_0=self.a_0, mu=self.mu, c=self.c
        )
        nash = select_nash_index(
            eq, self.grid, a_i=self.a_i, a_0=self.a_0, mu=self.mu, c=self.c
        )
        mono = monopoly_index(
            self.grid, a_i=self.a_i, a_0=self.a_0, mu=self.mu, c=self.c
        )
        object.__setattr__(self, "equilibria", tuple(eq))
        object.__setattr__(self, "nash_idx", nash)
        object.__setattr__(self, "mono_idx", mono)

    @property
    def p_nash(self) -> float:
        return float(self.grid[self.nash_idx])

    @property
    def p_mono(self) -> float:
        return float(self.grid[self.mono_idx])

    def payoff(self, p_i: float, p_j: float) -> float:
        return payoff(p_i, p_j, a_i=self.a_i, a_0=self.a_0, mu=self.mu, c=self.c)

    def demand(self, p_i: float, p_j: float) -> float:
        return demand(p_i, p_j, a_i=self.a_i, a_0=self.a_0, mu=self.mu)

    def best_response_index(self, j_idx: int) -> int:
        return best_response_index(
            j_idx, self.grid, a_i=self.a_i, a_0=self.a_0, mu=self.mu, c=self.c
        )

    def delta(self, p_bar: float) -> float:
        """Price-space collusion index Delta = (p_bar - p_nash)/(p_mono - p_nash)."""
        return (p_bar - self.p_nash) / (self.p_mono - self.p_nash)

    def pf1_norm(self) -> float:
        """pf1 normalizer: pi(p_mono,p_mono) - pi(p_nash,p_nash) (design 8.1)."""
        return self.payoff(self.p_mono, self.p_mono) - self.payoff(
            self.p_nash, self.p_nash
        )
