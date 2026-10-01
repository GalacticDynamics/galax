"""Each interop distribution owns its own optional-dependency check.

The shared four-member `galax.interop.optional_deps.OptDeps` could not survive
the split: `galax/interop/` is a namespace directory shared by four
distributions, and none of them may own a module sitting loose in it.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    ("module", "member"),
    [
        ("galax.interop.astropy.optional_deps", "ASTROPY"),
        ("galax.interop.gala.optional_deps", "GALA"),
        ("galax.interop.galpy.optional_deps", "GALPY"),
        ("galax.interop.matplotlib.optional_deps", "MATPLOTLIB"),
    ],
)
def test_each_package_declares_only_its_own_library(module: str, member: str) -> None:
    """`OptDeps` in each package has exactly one member, its own."""
    OptDeps = importlib.import_module(module).OptDeps
    assert [m.name for m in OptDeps] == [member]


@pytest.mark.parametrize(
    ("module", "member"),
    [
        ("galax.interop.astropy.optional_deps", "ASTROPY"),
        ("galax.interop.gala.optional_deps", "GALA"),
        ("galax.interop.galpy.optional_deps", "GALPY"),
        ("galax.interop.matplotlib.optional_deps", "MATPLOTLIB"),
    ],
)
def test_installed_is_a_bool_and_never_raises(module: str, member: str) -> None:
    """`.installed` answers for an absent or half-built library, not raises.

    Review Focus 4: `gala` present but `gala.dynamics` missing is the exact case
    `chain_checks` guards. Probing must degrade to `False`, never propagate.
    """
    OptDeps = importlib.import_module(module).OptDeps
    assert isinstance(getattr(OptDeps, member).installed, bool)


def test_gala_package_owns_gsl_enabled() -> None:
    """`GSL_ENABLED` is gala-specific and lives with gala."""
    from galax.interop.gala.optional_deps import GSL_ENABLED

    assert isinstance(GSL_ENABLED, bool)


def test_shared_module_is_gone() -> None:
    """The shared module is deleted, not stubbed.

    A signposting stub would itself be a loose module in `galax/interop/` --
    exactly what `tests/smoke/test_namespace_hygiene.py` forbids, since that
    directory is shared by five distributions. The signpost for users lives in
    `RELEASING.md` and the release notes instead.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("galax.interop.optional_deps")
