"""Only the gala interop distribution needs an optional-dependency module.

The shared four-member `galax.interop.optional_deps.OptDeps` could not survive
the split: `galax/interop/` is a namespace directory shared by five
distributions, and none of them may own a module sitting loose in it.

Dissolving it per library initially produced four modules, one each. Three were
dead on arrival: every interop distribution *requires* the library it wraps, so
`OptDeps.<LIB>.installed` is a constant `True` inside it, and astropy's, galpy's
and matplotlib's modules had no consumer but their own tests. Only gala's does
real work -- version gating and `GSL_ENABLED` -- so only gala's remains.
"""

import importlib

import pytest
from packaging.version import Version


def test_gala_reports_the_two_facts_a_pin_cannot_express() -> None:
    """Whether gala was built against GSL, and which version it is.

    Not *whether* gala is installed: `gala>=1.10` is a required dependency of
    that distribution, so that question has a constant answer.
    """
    from galax.interop.gala.optional_deps import GALA_VERSION, GSL_ENABLED

    assert isinstance(GSL_ENABLED, bool)
    assert Version("1.10") <= GALA_VERSION


@pytest.mark.parametrize("lib", ["astropy", "galpy", "matplotlib"])
def test_the_other_packages_have_no_optional_deps_module(lib: str) -> None:
    """They require the library they wrap, so the probe would be a constant.

    Asserted rather than merely omitted: re-adding one for symmetry is the
    tempting mistake, and it is how the three dead modules got there the first
    time.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"galax.interop.{lib}.optional_deps")


def test_shared_module_is_gone() -> None:
    """The shared module is deleted, not stubbed.

    A signposting stub would itself be a loose module in `galax/interop/` --
    exactly what `tests/smoke/test_namespace_hygiene.py` forbids. The signpost
    for users lives in `RELEASING.md` and the release notes instead.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("galax.interop.optional_deps")
