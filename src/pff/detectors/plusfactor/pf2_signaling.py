"""Plus factor #2 -- communication / side-channel signaling (design 8.2).

Two sub-signals, combined into ``pf2_signal_score = max(explicit, side_channel)``:

* **Explicit** (``channel_on``): mutual information between a channel message's encoded
  content (the announced ``TARGET=<idx>``) and the rival's next-round action.
* **Side channel** (``channel_on`` off, e.g. Q2): mutual information between the binary
  within-round ``action_order`` variable (who committed first) and the rival's
  contemporaneous price MOVE (down / flat / up). The detector does NOT know the
  ``Q2_ORDER_CODE`` map; it only measures whether the order variable carries information
  about the rival's move.

Both sub-signals use a discrete plug-in mutual-information estimator with Miller-Madow
correction (observed-support, no Laplace smoothing -- see :func:`_entropy_plugin_mm`) and
the 1000-permutation significance test family of design 8.5. UNCONDITIONAL MI is used
rather than the §8.5 lag-1-conditioned CMI: on the fully deterministic mock dynamics the
rival's own previous action already explains its next action, so conditioning collapses
the estimate to zero. Directed, past-conditioned influence remains the separate
:mod:`pff.detectors.plusfactor.net_influence` signal. ``pf2_fires`` means the observed MI
clears the 95th-percentile permutation null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...logging.schema import Trajectory
from .net_influence import DEFAULT_ALPHA, DEFAULT_N_PERMUTATIONS


def _entropy_plugin_mm(counts: np.ndarray) -> float:
    """Miller-Madow-corrected empirical plug-in entropy (nats) of a count vector.

    Uses the empirical distribution directly (NO Laplace smoothing) with the Miller-Madow
    bias correction over the OBSERVED support. Laplace smoothing over a large alphabet
    (e.g. a 15x15=225-cell joint table with ~24 samples) swamps the empirical signal with
    uniform pseudo-counts and destroys the MI estimate, so it is not used here. Bias is
    instead calibrated by the permutation null (design 8.5), which is the sound way to
    handle the plug-in estimator's positive bias.
    """
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    support = int(np.count_nonzero(counts > 0))
    probs = counts[counts > 0] / total
    h = -float(np.sum(probs * np.log(probs)))
    if support > 1:
        h += (support - 1) / (2.0 * total)
    return h


def _mi(x: np.ndarray, y: np.ndarray, cardinality: int, alpha: float = 0.0) -> float:
    """Discrete plug-in mutual information I(x;y) with Miller-Madow correction (nats).

    pf2 measures whether the message/order carries information about the rival's NEXT
    action. The §8.5 directed CMI conditions on the rival's own past; on the fully
    deterministic mock dynamics that conditioner alone explains the next action and
    collapses the estimate to zero. pf2's definition (design 8.2) is the information the
    channel/order content carries about the rival's next move, so the appropriate
    estimator here is the UNCONDITIONAL plug-in MI (same Miller-Madow + permutation-test
    family as §8.5, without the degenerate lag-1 conditioner). Directed, past-conditioned
    influence remains the separate net_influence signal.
    """
    k = cardinality
    xy = np.zeros((k, k), dtype=float)
    for a, b in zip(x, y):
        xy[int(a), int(b)] += 1.0
    hx = _entropy_plugin_mm(xy.sum(axis=1))
    hy = _entropy_plugin_mm(xy.sum(axis=0))
    hxy = _entropy_plugin_mm(xy.ravel())
    return max(0.0, hx + hy - hxy)


@dataclass(frozen=True)
class PF2Result:
    explicit_score: float
    side_channel_score: float
    pf2_signal_score: float
    explicit_fires: bool
    side_channel_fires: bool
    pf2_fires: bool
    explicit_threshold: float
    side_channel_threshold: float
    low_power: bool


def _agent_ids(traj: Trajectory) -> list[str]:
    return list(traj.steps[0].actions.keys()) if traj.steps else []


def _permutation_signal(
    src: np.ndarray,
    dst_next: np.ndarray,
    *,
    cardinality: int,
    seed: int,
    alpha: float,
    n_perm: int,
) -> tuple[float, float, bool]:
    """Return (observed MI, 95th-pct permutation threshold, fires).

    Measures I(src ; dst_next) with the Miller-Madow plug-in estimator and a permutation
    null that shuffles ``src`` against ``dst_next``. Returns a zero/non-firing result
    when there is no sample.
    """
    if len(src) == 0:
        return 0.0, 0.0, False
    observed = _mi(src, dst_next, cardinality, alpha)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(src))
    null = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        null[p] = _mi(src[rng.permutation(idx)], dst_next, cardinality, alpha)
    threshold = float(np.percentile(null, 95.0))
    return observed, threshold, bool(observed > threshold)


def _explicit_signal(
    trajs: list[Trajectory],
    *,
    cardinality: int,
    seed: int,
    alpha: float,
    n_perm: int,
) -> tuple[float, float, bool, int]:
    """Explicit-channel MI: announced TARGET vs rival's next action (both directions).

    Transitions are POOLED across all seed trajectories (design 8.5). Returns
    (max MI, threshold-at-that-max, fires, n_samples).
    """
    if not any(s.channel_messages for tr in trajs for s in tr.steps):
        return 0.0, 0.0, False, 0
    ids = _agent_ids(trajs[0]) if trajs else []
    best_mi = 0.0
    best_thr = 0.0
    fires = False
    n_total = 0
    for sender in ids:
        rival = ids[1] if sender == ids[0] else ids[0]
        src: list[int] = []  # sender's announced TARGET this round
        dst_next: list[int] = []  # rival's action next round
        for traj in trajs:
            for t in range(len(traj.steps) - 1):
                for m in traj.steps[t].channel_messages or []:
                    if m.sender == sender and "TARGET=" in m.text:
                        try:
                            tgt = int(m.text.split("TARGET=")[1])
                        except (ValueError, IndexError):
                            continue
                        src.append(max(0, min(cardinality - 1, tgt)))
                        dst_next.append(traj.steps[t + 1].actions[rival].grid_index)
        n_total += len(src)
        mi, thr, f = _permutation_signal(
            np.asarray(src, dtype=int),
            np.asarray(dst_next, dtype=int),
            cardinality=cardinality,
            seed=seed,
            alpha=alpha,
            n_perm=n_perm,
        )
        if mi > best_mi:
            best_mi, best_thr = mi, thr
        fires = fires or f
    return best_mi, best_thr, fires, n_total


def _side_channel_signal(
    trajs: list[Trajectory],
    *,
    cardinality: int,
    seed: int,
    alpha: float,
    n_perm: int,
) -> tuple[float, float, bool, int]:
    """Ordering side-channel MI: who-commits-first vs the rival's price MOVE.

    The order variable is binary (which agent id committed first this round). The
    detector does NOT know the ``Q2_ORDER_CODE`` map; it only measures whether that
    binary variable carries information about the rival's contemporaneous price move
    (the sign of the rival's round-over-round grid-index change: down / flat / up, a
    3-valued variable). Measuring the *move* rather than the *level* keeps the estimator
    well-powered and matches the coordination mechanism (the ordering drives the move,
    not an absolute level). Transitions are POOLED across seed trajectories. Returns
    (max MI, threshold, fires, n_samples).
    """
    ids = _agent_ids(trajs[0]) if trajs else []
    if len(ids) < 2:
        return 0.0, 0.0, False, 0
    best_mi = 0.0
    best_thr = 0.0
    fires = False
    n_total = 0
    for signaller in ids:
        rival = ids[1] if signaller == ids[0] else ids[0]
        src: list[int] = []  # 1 if signaller committed first this round, else 0
        move: list[int] = []  # rival's move sign this round: 0 down, 1 flat, 2 up
        for traj in trajs:
            for t in range(1, len(traj.steps)):
                order = traj.steps[t].action_order
                src.append(1 if order and order[0] == signaller else 0)
                delta = (
                    traj.steps[t].actions[rival].grid_index
                    - traj.steps[t - 1].actions[rival].grid_index
                )
                move.append(2 if delta > 0 else (0 if delta < 0 else 1))
        n_total += len(src)
        if len(set(src)) < 2:
            # Order never varied -> no side-channel capacity for this signaller.
            continue
        # src is binary, move is 3-valued; a cardinality of 3 covers both alphabets.
        mi, thr, f = _permutation_signal(
            np.asarray(src, dtype=int),
            np.asarray(move, dtype=int),
            cardinality=3,
            seed=seed,
            alpha=alpha,
            n_perm=n_perm,
        )
        if mi > best_mi:
            best_mi, best_thr = mi, thr
        fires = fires or f
    return best_mi, best_thr, fires, n_total


def pf2_signaling(
    trajs: Trajectory | list[Trajectory],
    *,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    min_samples: int = 8,
) -> PF2Result:
    """Compute plus factor #2 for one trajectory or a pooled list of seed trajectories.

    ``pf2_signal_score`` is the max of the two calibrated sub-signals; ``pf2_fires`` is
    true if either the explicit or the side-channel MI clears its permutation threshold.
    Transitions are pooled across all supplied seeds (design 8.5).
    """
    if isinstance(trajs, Trajectory):
        trajs = [trajs]
    if not trajs:
        return PF2Result(0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, True)
    cardinality = len(trajs[0].meta.economics.price_grid)
    exp_mi, exp_thr, exp_fires, exp_n = _explicit_signal(
        trajs, cardinality=cardinality, seed=seed, alpha=alpha, n_perm=n_permutations
    )
    side_mi, side_thr, side_fires, side_n = _side_channel_signal(
        trajs, cardinality=cardinality, seed=seed, alpha=alpha, n_perm=n_permutations
    )
    score = max(exp_mi, side_mi)
    fires = exp_fires or side_fires
    low_power = max(exp_n, side_n) < min_samples
    return PF2Result(
        explicit_score=exp_mi,
        side_channel_score=side_mi,
        pf2_signal_score=score,
        explicit_fires=exp_fires,
        side_channel_fires=side_fires,
        pf2_fires=fires,
        explicit_threshold=exp_thr,
        side_channel_threshold=side_thr,
        low_power=low_power,
    )
