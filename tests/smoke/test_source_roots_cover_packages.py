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
    """Return the `src` root of every distribution under `packages/`.

    Keyed on `pyproject.toml`, not on the `src` directory existing. hatch's
    version-file hook writes a gitignored `_version.py` into each portion, and
    one left behind by a branch switch is enough to conjure a
    `packages/<dist>/src` for a distribution the tree does not have -- which
    failed this guard against a branch that was fine. Packaging metadata cannot
    be left behind that way, and it is the better definition anyway: a
    `packages/*` entry is a distribution because it declares itself one.
    """
    return [
        (p.parent / "src").relative_to(ROOT).as_posix()
        for p in sorted(ROOT.glob("packages/*/pyproject.toml"))
    ]


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


def test_no_configured_source_root_is_stale() -> None:
    """The converse: a configured path that is no longer a distribution.

    The two tests above only check presence, so a path left behind after a
    portion moves goes unnoticed. That is not harmless -- coverage errors on a
    source path that does not exist, and a dead `mypy_path` entry is silently
    useless. Both lists carried `"src"` for the whole of the move that deleted
    it, and these tests passed throughout.
    """
    expected = set(_package_src_roots())
    cfg = _config()
    for setting, configured in (
        ("coverage run.source", cfg["tool"]["coverage"]["run"]["source"]),
        ("mypy_path", cfg["tool"]["mypy"]["mypy_path"]),
    ):
        stale = [p for p in configured if p not in expected]
        assert not stale, f"{setting} names paths that are not distributions: {stale}"
