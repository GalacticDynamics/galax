"""Every package's tests must be reachable from the configured testpaths.

A dropped path fails nothing -- the tests simply stop running -- which is the
same silence that made a missing entry point dangerous in phase 1.
"""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).parents[2]


def _testpaths() -> list[str]:
    """Return the `testpaths` configured in the root pyproject."""
    with (ROOT / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["tool"]["pytest"]["ini_options"]["testpaths"]


def test_every_package_tests_dir_is_covered() -> None:
    """A packages/*/tests tree must sit under some configured testpath."""
    paths = [ROOT / p for p in _testpaths()]
    uncovered = [
        str(d.relative_to(ROOT))
        for d in sorted(ROOT.glob("packages/*/tests"))
        if not any(d.is_relative_to(p) for p in paths)
    ]
    assert not uncovered, f"not reachable from testpaths: {uncovered}"
