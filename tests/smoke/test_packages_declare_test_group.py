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


_NAME = re.compile(r"[A-Za-z0-9._-]+")


def _normalise(name: str) -> str:
    """Normalise a distribution or group name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(
    groups: dict[str, list], name: str, _seen: frozenset[str] = frozenset()
) -> set[str]:
    """Return the normalised distribution names a group requires.

    Requirement strings carry extras, version bounds and markers; a name runs
    until the first character that cannot appear in one. An entry that is a
    table is an `include-group`, naming another group in the same file, whose
    members count as this group's -- the root `pyproject.toml` factors `dev`
    and `test-all` that way, so a portion may reasonably do the same. Treating
    those entries as nothing would report a group that includes the whole floor
    as missing all of it.
    """
    key = _normalise(name)
    if key in _seen:  # a cycle, which PEP 735 forbids; stop rather than recurse
        return set()
    out: set[str] = set()
    for req in groups.get(name, []):
        if isinstance(req, str):
            if m := _NAME.match(req):
                out.add(_normalise(m[0]))
        elif isinstance(req, dict) and (included := req.get("include-group")):
            out |= _names(groups, included, _seen | {key})
    return out


def test_include_group_members_count_as_the_group_s_own() -> None:
    """A group factored through `include-group` is not an empty group."""
    groups = {
        "floor": ["pytest>=8.3", "sybil>=9.0.0"],
        "test": [{"include-group": "floor"}, "pytest-mpl>=0.17.0"],
        "self": [{"include-group": "self"}],
    }
    assert _names(groups, "test") == {"pytest", "sybil", "pytest-mpl"}
    assert _names(groups, "self") == set()  # a cycle terminates


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.parent.name)
def test_package_declares_test_group(path: pathlib.Path) -> None:
    """A portion with no `test` group cannot be tested on its own."""
    with path.open("rb") as f:
        config = tomllib.load(f)
    groups = config.get("dependency-groups", {})
    assert "test" in groups, f"{path.parent.name} declares no `test` group"
    missing = FLOOR - _names(groups, "test")
    assert not missing, f"{path.parent.name} `test` group is missing: {sorted(missing)}"


def test_every_package_is_covered() -> None:
    """The parametrisation is only as good as the glob that built it."""
    assert PACKAGES, "found no `packages/*/pyproject.toml` to check"
