"""Interop plugins must actually register, and not collide with each other.

A missing or broken entry point produces *silence*: the conversion simply does
not exist, and a test that never exercised it still passes. These assertions are
deliberately behavioural.
"""

import subprocess
import sys
import warnings
from importlib.metadata import entry_points

import pytest

GROUPS = (
    "galax.coordinates.interop",
    "galax.potential.interop",
    "galax.dynamics.interop",
)


def test_no_group_is_empty() -> None:
    """After the split, every group must still be populated.

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


def test_loading_all_plugins_raises_no_redefinition_warning() -> None:
    """Review Focus 3: four distributions registering into shared dispatchers.

    Two packages claiming the same plum signature raises
    `MethodRedefinitionWarning`, which `filterwarnings = ["error"]` turns into a
    hard failure at an arbitrary later call. Load everything and look.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for group in GROUPS:
            for ep in entry_points(group=group):
                ep.load()


# Run in a fresh interpreter. `plum`'s default dispatcher does not warn on
# redefinition (`warn_redefinition=False`), so a second plugin claiming a
# signature silently *replaces* the first -- the worst kind of collision. Turn
# the warning on and make it an error, but only for galax-sourced methods:
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
    """The in-process check above can be vacuous; this one cannot.

    By the time pytest runs, `galax` has already imported every plugin, so
    `ep.load()` is a no-op that re-runs no registration. A fresh interpreter with
    redefinition warnings enabled performs each registration for the first time.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _FRESH_PROCESS_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("group", "name"),
    [
        ("galax.coordinates.interop", "astropy"),
        ("galax.potential.interop", "astropy"),
        ("galax.dynamics.interop", "astropy"),
    ],
)
def test_astropy_registers_on_a_default_install(group: str, name: str) -> None:
    """Astropy interop is required, so it is present without any extra."""
    assert name in {ep.name for ep in entry_points(group=group)}
