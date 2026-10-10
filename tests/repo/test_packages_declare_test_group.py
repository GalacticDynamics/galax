"""Every distribution must declare the test dependencies it needs on its own.

A portion whose `pyproject.toml` has no `test` group is still covered by the
whole-repo run, which syncs the root group and collects all of `packages/` --
so nothing fails, and the gap shows up only when someone first tries to test
that portion alone and finds they cannot. Declaring the group per portion is
what makes `uv sync --package <dist> --group test` a working command, and this
guards the declaration against the next portion added.

The floor is identical across the portions because the root configuration and
the workflow's own invocation force it, not anything a portion imports.
`--arraydiff` and `env` come from `[tool.pytest.ini_options]`; the root
`conftest.py` imports `sybil` and `optional_dependencies` at module level, so
every run rooted here loads them whatever it collects; and `pytest-cov` and
`pytest-xdist` are what `--cov` and `-n logical` in `ci.yml` need, which no
pytest configuration states. Each absent member fails loudly --
`--strict-config` on an unknown `env` key, an unrecognised `--arraydiff` or
`--cov`, an `ImportError` out of `conftest.py` -- but only for whoever runs the
portion alone, which is the run this test exists to keep possible.

This is the same silence `test_source_roots_cover_packages.py` guards for
coverage and mypy. It is a stopgap: once CI syncs and tests each portion on its
own (#941), that job fails on any missing member -- including the per-portion
extras, like `pytest-mpl`, that this floor cannot see -- and this can go.
"""

import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).parents[2]

FLOOR = frozenset(
    {
        "optional-dependencies",
        "pytest",
        "pytest-arraydiff",
        "pytest-cov",
        "pytest-env",
        "pytest-xdist",
        "sybil",
    }
)

PACKAGES = sorted(ROOT.glob("packages/*/pyproject.toml"))


def _names(reqs: list[str]) -> set[str]:
    """Return the PEP 503-normalised distribution names in `reqs`."""
    names = (re.split(r"[^A-Za-z0-9._-]", r, maxsplit=1)[0] for r in reqs)
    return {re.sub(r"[-_.]+", "-", n).lower() for n in names}


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.parent.name)
def test_package_declares_test_group(path: pathlib.Path) -> None:
    """A portion with no `test` group cannot be tested on its own."""
    with path.open("rb") as f:
        config = tomllib.load(f)
    groups = config.get("dependency-groups", {})
    assert "test" in groups, f"{path.parent.name} declares no `test` group"
    missing = FLOOR - _names(groups["test"])
    assert not missing, f"{path.parent.name} `test` group is missing: {sorted(missing)}"


def test_every_package_is_covered() -> None:
    """The parametrisation is only as good as the glob that built it."""
    assert PACKAGES, "found no `packages/*/pyproject.toml` to check"
