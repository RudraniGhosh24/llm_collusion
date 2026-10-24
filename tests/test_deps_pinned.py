"""NFR-3: assert no unpinned specifiers in pyproject.toml [project].dependencies."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
LOCK = Path(__file__).resolve().parents[1] / "requirements.lock"


def _load_deps() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["dependencies"]


def test_all_dependencies_use_exact_pins():
    deps = _load_deps()
    assert deps, "expected at least one dependency"
    for spec in deps:
        assert "==" in spec, f"dependency not exactly pinned: {spec!r}"
        # No loose/compatible-release specifiers allowed.
        for bad in (">=", "<=", "~=", ">", "<", "!="):
            assert bad not in spec, f"disallowed specifier {bad!r} in {spec!r}"
        # A bare name (no specifier) is also disallowed.
        assert re.search(r"==\s*\d", spec), f"missing exact version in {spec!r}"


def test_lock_file_exists_and_agrees_with_pins():
    assert LOCK.exists(), "requirements.lock must exist"
    lock_text = LOCK.read_text(encoding="utf-8").lower()
    deps = _load_deps()
    for spec in deps:
        name = re.split(r"==", spec)[0].strip().lower()
        assert f"{name}==" in lock_text, f"{name} missing from requirements.lock"
