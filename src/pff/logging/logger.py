"""JSONL trajectory writer + validate-on-read + determinism projection (design 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Trajectory


def write_trajectories(path: str | Path, trajectories: list[Trajectory]) -> Path:
    """Write one JSON ``Trajectory`` per line to ``path`` (``*.trajectories.jsonl``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            fh.write(traj.model_dump_json())
            fh.write("\n")
    return path


def read_trajectories(path: str | Path) -> list[Trajectory]:
    """Read and validate every line of a ``*.trajectories.jsonl`` file."""
    path = Path(path)
    out: list[Trajectory] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(Trajectory.model_validate_json(line))
    return out


def determinism_projection(traj: Trajectory) -> dict[str, Any]:
    """Return a dict of the determinism-critical fields only.

    Excludes the two explicitly volatile fields: ``StepRecord.wall_clock`` and
    ``RunMeta.git_revision`` (design 4.2). Everything the science depends on
    (actions, payoffs, payoff_inputs, plans, messages, order, logical clock, prompts,
    knobs, labels) is retained.
    """
    data = traj.model_dump(mode="json")
    data["meta"].pop("git_revision", None)
    # run_id depends on git_revision, so exclude it from the projection too.
    data["meta"].pop("run_id", None)
    for step in data["steps"]:
        step.pop("wall_clock", None)
    return data
