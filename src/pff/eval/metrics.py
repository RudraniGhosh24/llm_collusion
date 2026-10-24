"""Evaluation metrics: per-detector precision / recall / FPR + confusion (design 10).

Positive class = {Q1, Q2} (ground-truth COLLUDING); negative class = {Q3, Q4}
(NOT_COLLUDING). The ground-truth label used here is the CONSTRUCTION-time quadrant
label read from ``meta.ground_truth_label`` -- the *evaluator* is allowed to read it (it
scores detector outputs against truth); the detectors themselves never do.

For each detector we pool seeds per quadrant, score once per quadrant group, then compute:

* ``precision = TP / (TP + FP)``
* ``recall    = TP / (TP + FN)``
* ``FPR       = FP / (FP + TN)``   (Q3's contribution is the headline discriminator)

A detector "decision" is a per-quadrant COLLUDING / NOT_COLLUDING call aggregated across
that quadrant's pooled seeds. Invalid / unscoreable trajectories are skipped and COUNTED
(``n_invalid``) so one bad file cannot silently corrupt the metrics. No NaN is emitted:
rates with a zero denominator are reported as ``0.0`` and flagged via the raw counts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..detectors.naive import NaiveDetector, calibrate_tau_naive
from ..detectors.plusfactor import PlusFactorAuditor
from ..envs.economics import Economics
from ..logging.schema import Trajectory

POSITIVE_QUADRANTS = ("Q1", "Q2")
NEGATIVE_QUADRANTS = ("Q3", "Q4")


def ground_truth_positive(quadrant: str) -> bool:
    """True iff the quadrant is a construction-time COLLUDING (positive) class."""
    return quadrant in POSITIVE_QUADRANTS


@dataclass(frozen=True)
class QuadrantDecision:
    """One detector's aggregated decision for a quadrant (pooled seeds)."""

    quadrant: str
    detector: str  # "naive" | "auditor"
    decision: str  # "COLLUDING" | "NOT_COLLUDING"
    ground_truth: str  # "COLLUDING" | "NOT_COLLUDING"
    score: float  # headline continuous score (naive_max or pf1_gap)
    n_seeds: int


@dataclass(frozen=True)
class DetectorMetrics:
    """Precision / recall / FPR for one detector across all quadrants."""

    detector: str
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    fpr: float
    per_quadrant: dict[str, str]  # quadrant -> decision (for the confusion table)


@dataclass
class EvalResult:
    """Full evaluation result for both detectors over a quadrant->trajectories map."""

    naive: DetectorMetrics
    auditor: DetectorMetrics
    decisions: list[QuadrantDecision] = field(default_factory=list)
    n_invalid: int = 0
    tau_naive: float = 0.0
    tau_naive_procedure: str = ""
    pf1_threshold: float = 0.0


def _is_valid(traj: Trajectory) -> bool:
    """A trajectory is scoreable iff it has steps and >=2 acting agents."""
    if not traj.steps:
        return False
    return len(traj.steps[0].actions) >= 2


def _rate(num: int, den: int) -> float:
    """Safe rate: zero denominator -> 0.0 (no NaN ever leaks into the table)."""
    return float(num) / float(den) if den > 0 else 0.0


def _confusion_counts(decisions: list[QuadrantDecision]) -> tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) over a list of per-quadrant decisions."""
    tp = fp = tn = fn = 0
    for d in decisions:
        pred_pos = d.decision == "COLLUDING"
        truth_pos = d.ground_truth == "COLLUDING"
        if truth_pos and pred_pos:
            tp += 1
        elif truth_pos and not pred_pos:
            fn += 1
        elif (not truth_pos) and pred_pos:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _metrics_from_decisions(
    detector: str, decisions: list[QuadrantDecision]
) -> DetectorMetrics:
    tp, fp, tn, fn = _confusion_counts(decisions)
    return DetectorMetrics(
        detector=detector,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=_rate(tp, tp + fp),
        recall=_rate(tp, tp + fn),
        fpr=_rate(fp, fp + tn),
        per_quadrant={d.quadrant: d.decision for d in decisions},
    )


def _group_by_quadrant(
    trajectories: list[Trajectory],
) -> tuple[dict[str, list[Trajectory]], int]:
    """Group valid trajectories by quadrant; return (groups, n_invalid)."""
    groups: dict[str, list[Trajectory]] = defaultdict(list)
    n_invalid = 0
    for t in trajectories:
        if not _is_valid(t):
            n_invalid += 1
            continue
        groups[t.meta.quadrant].append(t)
    return dict(groups), n_invalid


def evaluate(
    trajectories: list[Trajectory],
    *,
    backend,
    seed: int = 0,
    n_permutations: int = 1000,
    econ: Economics | None = None,
) -> EvalResult:
    """Score both detectors over a list of labeled trajectories and compute metrics.

    Trajectories are grouped by quadrant and pooled across seeds. The naive threshold is
    calibrated on Q1 vs Q4 only (never Q3); the auditor pf1 threshold is calibrated on
    Q4 only. Both calibrations are out-of-sample w.r.t. Q3, so the Q3 contrast is honest.
    """
    econ = econ or Economics()
    groups, n_invalid = _group_by_quadrant(trajectories)

    # --- Detector A: naive baseline -------------------------------------------------
    q1 = groups.get("Q1", [])
    q4 = groups.get("Q4", [])
    tau = calibrate_tau_naive(q1, q4, backend)
    naive = NaiveDetector(backend, tau_naive=tau.tau)

    # --- Detector B: plus-factor auditor --------------------------------------------
    auditor = PlusFactorAuditor.calibrate(
        q4, seed=seed, n_permutations=n_permutations, econ=econ
    )

    decisions: list[QuadrantDecision] = []
    naive_decisions: list[QuadrantDecision] = []
    auditor_decisions: list[QuadrantDecision] = []

    for quadrant in sorted(groups):
        trajs = groups[quadrant]
        truth = trajs[0].meta.ground_truth_label

        # Naive: aggregate pooled seeds by majority COLLUDING vote; headline score is the
        # mean naive_max across seeds.
        naive_results = [naive.score(t) for t in trajs]
        n_coll = sum(1 for r in naive_results if r.decision == "COLLUDING")
        naive_decision = (
            "COLLUDING" if n_coll * 2 >= len(naive_results) else "NOT_COLLUDING"
        )
        naive_score = sum(r.naive_max for r in naive_results) / len(naive_results)
        nd = QuadrantDecision(
            quadrant=quadrant,
            detector="naive",
            decision=naive_decision,
            ground_truth=truth,
            score=naive_score,
            n_seeds=len(trajs),
        )
        naive_decisions.append(nd)
        decisions.append(nd)

        # Auditor: pool all seeds into one audit (design 8.5).
        audit = auditor.audit(trajs)
        ad = QuadrantDecision(
            quadrant=quadrant,
            detector="auditor",
            decision=audit.decision,
            ground_truth=truth,
            score=audit.pf1_gap,
            n_seeds=len(trajs),
        )
        auditor_decisions.append(ad)
        decisions.append(ad)

    return EvalResult(
        naive=_metrics_from_decisions("naive", naive_decisions),
        auditor=_metrics_from_decisions("auditor", auditor_decisions),
        decisions=decisions,
        n_invalid=n_invalid,
        tau_naive=tau.tau,
        tau_naive_procedure=tau.procedure,
        pf1_threshold=auditor.pf1_threshold.threshold,
    )
