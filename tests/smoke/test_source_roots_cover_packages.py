"""Every package's source root must be configured for coverage and for mypy.

Both settings enumerate one entry per distribution, and a missing entry is
silent in the worst way: coverage simply stops measuring the portion, which
reads as a coverage *drop* attributable to nothing in the diff, and mypy
resolves every cross-module type to `Any`, which surfaces as
``Class cannot subclass "X" (has type "Any")`` rather than as a missing path.

This is the same silence `test_testpaths_cover_packages.py` guards for tests.
"""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).parents[2]


def _config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _package_src_roots() -> list[str]:
    """Return every `packages/*/src` directory, as a posix-style path."""
    return [d.relative_to(ROOT).as_posix() for d in sorted(ROOT.glob("packages/*/src"))]


def test_coverage_measures_every_package() -> None:
    """A portion missing from `run.source` is silently unmeasured."""
    configured = set(_config()["tool"]["coverage"]["run"]["source"])
    missing = [p for p in _package_src_roots() if p not in configured]
    assert not missing, f"missing from coverage run.source: {missing}"


def test_mypy_path_includes_every_package() -> None:
    """A portion missing from `mypy_path` loses every cross-module type."""
    configured = set(_config()["tool"]["mypy"]["mypy_path"])
    missing = [p for p in _package_src_roots() if p not in configured]
    assert not missing, f"missing from mypy_path: {missing}"
