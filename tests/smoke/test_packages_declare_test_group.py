"""Every distribution must declare the test dependencies it needs on its own.

A portion whose `pyproject.toml` has no `test` group is still covered by the
whole-repo run, which syncs the root group and collects `packages/` entire --
so nothing fails, and the gap shows up only when someone first tries to test
that portion alone and finds they cannot. Declaring the group per portion is
what makes `uv sync --package <dist> --group test` a working command, and this
guards the declaration against the next portion added.

The floor is forced by the root configuration rather than by any portion's
imports, which is why it is identical across all of them: `--arraydiff` and
`env` come from `[tool.pytest.ini_options]`, and the root `conftest.py` imports
`sybil` and `optional_dependencies` at module level, so every run rooted here
loads them whatever it collects. Each absent floor member fails loudly --
`--strict-config` on an unknown `env` key, an unrecognised `--arraydiff`, an
`ImportError` out of `conftest.py` -- but only for whoever runs the portion
alone, which is the run this test exists to keep possible.

This is the same silence `test_source_roots_cover_packages.py` guards for
coverage and mypy.
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


def _names(group: list[str | dict]) -> set[str]:
    """Return the normalised distribution names in a dependency group.

    Requirement strings carry extras, version bounds and markers; a name runs
    until the first character that cannot appear in one. Entries that are not
    strings are `include-group` tables, which name a group rather than a
    distribution.
    """
    return {
        re.sub(r"[-_.]+", "-", m[0]).lower()
        for req in group
        if isinstance(req, str)
        for m in [re.match(r"[A-Za-z0-9._-]+", req)]
        if m
    }


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.parent.name)
def test_package_declares_test_group(path: pathlib.Path) -> None:
    """A portion with no `test` group cannot be tested on its own."""
    with path.open("rb") as f:
        config = tomllib.load(f)
    group = config.get("dependency-groups", {}).get("test")
    assert group is not None, f"{path.parent.name} declares no `test` group"
    missing = FLOOR - _names(group)
    assert not missing, f"{path.parent.name} `test` group is missing: {sorted(missing)}"


def test_every_package_is_covered() -> None:
    """The parametrisation is only as good as the glob that built it."""
    assert PACKAGES, "found no `packages/*/pyproject.toml` to check"
