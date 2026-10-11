"""Only the gala interop distribution needs an optional-dependency module.

Every interop distribution *requires* the library it wraps, so an "is it
installed" probe inside one is a constant `True`. Only gala's module answers
something a dependency pin cannot -- the GSL build flag and the exact version.

`galax/interop/` is a namespace directory shared by five distributions, so none
of them may own a module sitting loose in it.
"""

import importlib

import pytest
from packaging.version import Version


def test_gala_reports_the_two_facts_a_pin_cannot_express() -> None:
    """Whether gala was built against GSL, and which version it is.

    Not *whether* gala is installed: `gala>=1.10` is a required dependency of
    *that distribution*, so inside it the question has a constant answer.

    This file lives in the shared `tests/unit/` tree, which is always
    collected -- `conftest.py`'s `collect_ignore_glob` drops
    `packages/galax.interop.gala/*` when the distribution is absent, but not
    this. The `checks` CI job installs only the required astropy member, so
    the import has to be skippable. A function-scope `from ... import` would
    collect cleanly and then fail at run time.
    """
    optional_deps = pytest.importorskip(
        "galax.interop.gala.optional_deps",
        reason="requires the galax.interop.gala distribution",
    )

    assert isinstance(optional_deps.GSL_ENABLED, bool)
    assert Version("1.10") <= optional_deps.GALA_VERSION


@pytest.mark.parametrize("lib", ["astropy", "galpy", "matplotlib"])
def test_the_other_packages_have_no_optional_deps_module(lib: str) -> None:
    """They require the library they wrap, so the probe would be a constant.

    Asserted rather than merely omitted: re-adding one for symmetry is the
    tempting mistake.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"galax.interop.{lib}.optional_deps")


def test_shared_module_is_gone() -> None:
    """The shared module is deleted, not stubbed.

    A signposting stub would itself be a loose module in `galax/interop/` --
    exactly what `tests/repo/test_namespace_hygiene.py` forbids. The signpost
    for users lives in `RELEASING.md` and the release notes instead.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("galax.interop.optional_deps")
