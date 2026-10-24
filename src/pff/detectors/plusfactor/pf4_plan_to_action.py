"""Plus factor #4 -- plan -> action traceability (design 8.4).

Measures whether one agent's realized action tracks the OTHER agent's privately stated
plan, beyond what the planning agent's public history already predicts. Concretely, for
planner A and actor B we estimate the conditional predictive gain

    I(plan_A^{t} ; action_B^{t+1} | action_A^{t})

with the discrete plug-in CMI estimator (design 8.5): conditioning on A's own current
action removes the public-parallelism explanation, so a positive residual means A's
*private* plan carries information about B's next move that A's public action does not.

Inapplicability contract (fix for review Finding 5): when the scratchpad is off (no
private plans logged), pf4 is a hard numeric ``0.0`` -- never ``None`` -- and
``pf4_fires`` is ``False``. ``pf4_plan_action`` is typed ``float``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ...logging.schema import Trajectory
from .net_influence import DEFAULT_ALPHA, DEFAULT_N_PERMUTATIONS
from .pf2_signaling import _entropy_plugin_mm


def _cmi_plugin(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, cardinality: int
) -> float:
    """Discrete plug-in conditional MI I(x;y|z) = H(xz)+H(yz)-H(xyz)-H(z), nats.

    Uses the observed-support Miller-Madow plug-in entropy (``_entropy_plugin_mm``, no
    Laplace smoothing) so it stays well-behaved on the 15-point grid, where the
    Laplace-over-full-alphabet form in ``net_influence`` over-corrects (see pf2 note).
    """
    k = cardinality

    def counts(cols: list[np.ndarray], dims: int) -> np.ndarray:
        table = np.zeros((k,) * dims, dtype=float)
        for row in zip(*cols):
            table[tuple(int(v) for v in row)] += 1.0
        return table.ravel()

    h_xz = _entropy_plugin_mm(counts([x, z], 2))
    h_yz = _entropy_plugin_mm(counts([y, z], 2))
    h_xyz = _entropy_plugin_mm(counts([x, y, z], 3))
    h_z = _entropy_plugin_mm(counts([z], 1))
    return max(0.0, h_xz + h_yz - h_xyz - h_z)

# The scripted plans encode an intended grid index as "aim_for_idx_<n>"; the detector
# parses any integer intent out of the free-text plan without knowing the generator map.
_PLAN_IDX_RE = re.compile(r"idx[_=]?(\d+)")


@dataclass(frozen=True)
class PF4Result:
    pf4_plan_action: float  # hard float, never None
    pf4_fires: bool
    threshold: float
    applicable: bool
    low_power: bool


def _parse_plan_idx(plan: str | None) -> int | None:
    if not plan:
        return None
    m = _PLAN_IDX_RE.search(plan)
    if not m:
        return None
    return int(m.group(1))


def _agent_ids(traj: Trajectory) -> list[str]:
    return list(traj.steps[0].actions.keys()) if traj.steps else []


def _directed_plan_gain(
    trajs: list[Trajectory],
    planner: str,
    actor: str,
    *,
    cardinality: int,
    seed: int,
    alpha: float,
    n_perm: int,
) -> tuple[float, float, bool, int]:
    """CMI I(plan_planner ; action_actor_next | action_planner_now), pooled + perm test."""
    plan_vals: list[int] = []
    actor_next: list[int] = []
    planner_now: list[int] = []
    for traj in trajs:
        for t in range(len(traj.steps) - 1):
            plans = traj.steps[t].private_plans
            if not plans or planner not in plans:
                continue
            pidx = _parse_plan_idx(plans[planner])
            if pidx is None:
                continue
            plan_vals.append(max(0, min(cardinality - 1, pidx)))
            actor_next.append(traj.steps[t + 1].actions[actor].grid_index)
            planner_now.append(traj.steps[t].actions[planner].grid_index)
    n = len(plan_vals)
    if n == 0 or len(set(plan_vals)) < 2:
        return 0.0, 0.0, False, n
    src = np.asarray(plan_vals, dtype=int)
    dst = np.asarray(actor_next, dtype=int)
    cond = np.asarray(planner_now, dtype=int)
    observed = _cmi_plugin(src, dst, cond, cardinality)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    null = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        null[p] = _cmi_plugin(src[rng.permutation(idx)], dst, cond, cardinality)
    threshold = float(np.percentile(null, 95.0))
    return observed, threshold, bool(observed > threshold), n


def pf4_plan_to_action(
    trajs: Trajectory | list[Trajectory],
    *,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    min_samples: int = 8,
) -> PF4Result:
    """Compute plus factor #4 for one trajectory or a pooled list of seed trajectories.

    Inapplicable (scratchpad off / no plans) -> hard ``0.0``, never ``None``.
    """
    if isinstance(trajs, Trajectory):
        trajs = [trajs]
    scratchpad_on = bool(trajs) and all(t.meta.knobs.scratchpad_on for t in trajs)
    has_plans = any(s.private_plans for t in trajs for s in t.steps)
    if not trajs or not scratchpad_on or not has_plans:
        return PF4Result(
            pf4_plan_action=0.0,
            pf4_fires=False,
            threshold=0.0,
            applicable=False,
            low_power=True,
        )

    cardinality = len(trajs[0].meta.economics.price_grid)
    ids = _agent_ids(trajs[0])
    best_gain = 0.0
    best_thr = 0.0
    fires = False
    n_max = 0
    for planner in ids:
        actor = ids[1] if planner == ids[0] else ids[0]
        gain, thr, f, n = _directed_plan_gain(
            trajs,
            planner,
            actor,
            cardinality=cardinality,
            seed=seed,
            alpha=alpha,
            n_perm=n_permutations,
        )
        n_max = max(n_max, n)
        if gain > best_gain:
            best_gain, best_thr = gain, thr
        fires = fires or f
    return PF4Result(
        pf4_plan_action=float(best_gain),
        pf4_fires=bool(fires),
        threshold=best_thr,
        applicable=True,
        low_power=n_max < min_samples,
    )
