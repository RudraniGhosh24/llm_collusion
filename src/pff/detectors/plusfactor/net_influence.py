"""Net pairwise influence (FR-10, design 8.5). Bonjour et al. 2022 ``verify citation``.

The information-theoretic realization of plus factor #1. Actions are discrete grid /
game indices, so the estimator is a discrete plug-in conditional-mutual-information
(CMI) estimator with Miller-Madow bias correction and Laplace (add-alpha) smoothing.
The k-NN (Kraskov) continuous estimator is explicitly NOT used.

Estimated quantity (lag-1 conditioning):

    I(A_i^{t-1} ; A_j^{t} | A_j^{t-1})

pooled across all back-half rounds of all seeds of a quadrant / game. "Fires" means the
observed CMI exceeds the 95th percentile of a 1000-permutation null (shuffling
``A_i^{past}`` against ``A_j^{next}``). A ``low_power`` flag is raised when fewer than
``min_samples`` pooled transitions are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_SAMPLES = 40
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_ALPHA = 0.5  # Laplace / add-alpha smoothing
SIGNIFICANCE_PERCENTILE = 95.0


def _entropy_mm(counts: np.ndarray, alpha: float) -> float:
    """Miller-Madow-corrected plug-in entropy (nats) of a smoothed count table."""
    counts = counts.astype(float) + alpha
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    nz = probs[probs > 0]
    h = -np.sum(nz * np.log(nz))
    # Miller-Madow correction: + (K - 1) / (2 N), K = number of non-zero-support bins.
    k = int(np.count_nonzero(counts > 0))
    h += (k - 1) / (2.0 * total)
    return float(h)


def _joint_counts(samples: np.ndarray, cards: tuple[int, ...]) -> np.ndarray:
    """Build a joint count table over columns of ``samples`` with given cardinalities."""
    table = np.zeros(cards, dtype=float)
    for row in samples:
        table[tuple(int(v) for v in row)] += 1.0
    return table


def _cmi(
    a_past: np.ndarray,
    a_next: np.ndarray,
    cond: np.ndarray,
    cardinality: int,
    alpha: float,
) -> float:
    """Discrete plug-in CMI I(a_past ; a_next | cond) with Miller-Madow correction.

    All three variables share the same cardinality (grid/action index count).
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).
    """
    k = cardinality
    xz = _joint_counts(np.stack([a_past, cond], axis=1), (k, k))
    yz = _joint_counts(np.stack([a_next, cond], axis=1), (k, k))
    xyz = _joint_counts(np.stack([a_past, a_next, cond], axis=1), (k, k, k))
    z = _joint_counts(cond.reshape(-1, 1), (k,))
    h_xz = _entropy_mm(xz.ravel(), alpha)
    h_yz = _entropy_mm(yz.ravel(), alpha)
    h_xyz = _entropy_mm(xyz.ravel(), alpha)
    h_z = _entropy_mm(z.ravel(), alpha)
    return max(0.0, h_xz + h_yz - h_xyz - h_z)


@dataclass
class InfluenceResult:
    directed_i_to_j: float
    directed_j_to_i: float
    net_pairwise: float
    p_value_i_to_j: float
    p_value_j_to_i: float
    threshold_i_to_j: float
    threshold_j_to_i: float
    fires: bool
    low_power: bool
    n_samples: int


def _build_transitions(
    series_i: list[np.ndarray], series_j: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool lag-1 transitions across multiple seed series.

    Each element of ``series_i``/``series_j`` is one seed's 1-D array of action indices
    for agent i / agent j over rounds. Returns pooled arrays:
    (i_past, j_next, j_past, i_next).
    """
    i_past, j_next, j_past, i_next = [], [], [], []
    for xi, xj in zip(series_i, series_j):
        xi = np.asarray(xi, dtype=int)
        xj = np.asarray(xj, dtype=int)
        n = min(len(xi), len(xj))
        for t in range(1, n):
            i_past.append(xi[t - 1])
            j_next.append(xj[t])
            j_past.append(xj[t - 1])
            i_next.append(xi[t])
    return (
        np.asarray(i_past, dtype=int),
        np.asarray(j_next, dtype=int),
        np.asarray(j_past, dtype=int),
        np.asarray(i_next, dtype=int),
    )


def _directed_influence(
    src_past: np.ndarray,
    dst_next: np.ndarray,
    dst_past: np.ndarray,
    cardinality: int,
    rng: np.random.Generator,
    alpha: float,
    n_perm: int,
) -> tuple[float, float, float]:
    """Return (cmi, p_value, threshold) for src_past -> dst_next | dst_past."""
    observed = _cmi(src_past, dst_next, dst_past, cardinality, alpha)
    null = np.empty(n_perm, dtype=float)
    idx = np.arange(len(src_past))
    for p in range(n_perm):
        perm = rng.permutation(idx)
        null[p] = _cmi(src_past[perm], dst_next, dst_past, cardinality, alpha)
    threshold = float(np.percentile(null, SIGNIFICANCE_PERCENTILE))
    # p-value: fraction of null >= observed (with +1 smoothing).
    p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return observed, p_value, threshold


def net_pairwise_influence(
    series_i: list[np.ndarray],
    series_j: list[np.ndarray],
    *,
    cardinality: int,
    seed: int = 0,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
) -> InfluenceResult:
    """Estimate net pairwise influence between agents i and j.

    ``series_i``/``series_j`` are lists of per-seed back-half action-index arrays.
    """
    i_past, j_next, j_past, i_next = _build_transitions(series_i, series_j)
    n = len(i_past)
    low_power = n < min_samples
    rng = np.random.default_rng(seed)

    if n == 0:
        return InfluenceResult(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, False, True, 0)

    cmi_ij, p_ij, thr_ij = _directed_influence(
        i_past, j_next, j_past, cardinality, rng, alpha, n_permutations
    )
    cmi_ji, p_ji, thr_ji = _directed_influence(
        j_past, i_next, i_past, cardinality, rng, alpha, n_permutations
    )
    fires = (cmi_ij > thr_ij) or (cmi_ji > thr_ji)
    return InfluenceResult(
        directed_i_to_j=cmi_ij,
        directed_j_to_i=cmi_ji,
        net_pairwise=cmi_ij + cmi_ji,
        p_value_i_to_j=p_ij,
        p_value_j_to_i=p_ji,
        threshold_i_to_j=thr_ij,
        threshold_j_to_i=thr_ji,
        fires=fires,
        low_power=low_power,
        n_samples=n,
    )
