"""Interop plugins must actually register, and not collide with each other.

A missing or broken entry point produces *silence*: the conversion simply does
not exist, and a test that never exercised it still passes. These assertions are
deliberately behavioural.
"""

import subprocess
import sys
from importlib.metadata import entry_points

import pytest

GROUPS = (
    "galax.coordinates.interop",
    "galax.potential.interop",
    "galax.dynamics.interop",
)


def test_no_group_is_empty() -> None:
    """Every group must be populated.

    If the root dropped its entry points and no package picked them up, the
    groups go quietly empty.
    """
    for group in GROUPS:
        assert list(entry_points(group=group)), f"{group} has no entry points"


def test_every_entry_point_loads() -> None:
    """`load()` imports the target module; a typo'd path fails only here."""
    for group in GROUPS:
        for ep in entry_points(group=group):
            assert ep.load() is not None, f"{group}:{ep.name} failed to load"


# Run in a fresh interpreter. `plum`'s default dispatcher does not warn on
# redefinition (`warn_redefinition=False`), so a second plugin claiming a
# signature silently *replaces* the first. Turn the warning on and make it an
# error, but only for galax-sourced methods:
# third-party libraries (e.g. unxt) legitimately redefine each other's.
# Registration is lazy, so every function's `.methods` is read to force it.
_FRESH_PROCESS_PROBE = f"""
import warnings
from importlib.metadata import entry_points

import plum
from plum._resolver import MethodRedefinitionWarning

# `Dispatcher` is a frozen dataclass.
object.__setattr__(plum.dispatch, "warn_redefinition", True)
warnings.filterwarnings(
    "error",
    message=r"(?s).*/(src|site-packages)/galax/.*",
    category=MethodRedefinitionWarning,
)

for group in {GROUPS!r}:
    for ep in entry_points(group=group):
        ep.load()

import galax.coordinates, galax.dynamics, galax.potential
from galax.potential.utils import coord_dispatcher

for dispatcher in (plum.dispatch, coord_dispatcher):
    for function in dispatcher.functions.values():
        function.methods
"""


def test_fresh_process_registration_redefines_no_galax_method() -> None:
    """No two distributions may claim the same galax dispatch signature.

    This must run in a fresh interpreter. By the time pytest runs, `galax` has
    already imported every plugin, so `ep.load()` here would re-run no
    registration; a fresh process performs each one for the first time with
    redefinition warnings enabled.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _FRESH_PROCESS_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# (group, name, target module): every entry point declared by the four packages'
# own pyprojects. The dev environment syncs `--all-extras`, so all are present.
# Asserting the target as well as the name catches a copy-paste that registers
# e.g. matplotlib's dynamics entry point at its `.potential` module.
EXPECTED = [
    ("galax.coordinates.interop", "astropy", "galax.interop.astropy.coordinates"),
    ("galax.potential.interop", "astropy", "galax.interop.astropy.potential"),
    ("galax.dynamics.interop", "astropy", "galax.interop.astropy.dynamics"),
    ("galax.coordinates.interop", "gala", "galax.interop.gala.coordinates"),
    ("galax.potential.interop", "gala", "galax.interop.gala.potential"),
    ("galax.potential.interop", "galpy", "galax.interop.galpy.potential"),
    ("galax.potential.interop", "matplotlib", "galax.interop.matplotlib.potential"),
    ("galax.dynamics.interop", "matplotlib", "galax.interop.matplotlib.orbit"),
]


@pytest.mark.parametrize(("group", "name", "value"), EXPECTED)
def test_entry_point_is_registered(group: str, name: str, value: str) -> None:
    """Each interop package's entry points are present and aim at the right module.

    Without this a group stays non-empty on astropy alone, so dropping gala's,
    galpy's or matplotlib's entry point would pass every other test here. Whether
    astropy registers on a bare install, with no extras, is shown only by
    `scripts/check_install_shapes.sh`; this suite runs with every extra installed.
    """
    registered = {ep.name: ep.value for ep in entry_points(group=group)}
    assert registered.get(name) == value, f"{group}: {name} -> {registered.get(name)}"
