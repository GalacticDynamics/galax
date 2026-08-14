"""Test that `galax` is a PEP 420 namespace package."""

from importlib.metadata import version

import galax


def test_version() -> None:
    """The distribution still reports a version, via metadata."""
    assert version("galax")


def test_is_namespace_package() -> None:
    """`galax` is a namespace package.

    Asserted through the spec, which is how :pep:`451` defines it: a namespace
    package has no origin but does have submodule search locations. Checking
    `type(galax.__path__).__name__ == "_NamespacePath"` would test a CPython
    internal instead, and `__file__ is None` only tests a consequence.
    """
    spec = galax.__spec__
    assert spec is not None
    assert spec.origin is None
    assert spec.submodule_search_locations is not None


def test_no_package_attributes() -> None:
    """A namespace package carries no version or `__all__` of its own.

    `galax.__version__` was removed with `galax/__init__.py`; use
    `importlib.metadata.version("galax")` instead.
    """
    for attr in ("__version__", "__version_tuple__", "__all__"):
        assert not hasattr(galax, attr)
