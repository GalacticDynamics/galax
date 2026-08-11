"""Test that `galax` is a PEP 420 namespace package."""

from importlib.metadata import version

import galax


def test_version() -> None:
    """The distribution still reports a version, via metadata."""
    assert version("galax")


def test_is_namespace_package() -> None:
    """`galax` is a namespace: no `__init__.py`, so no `__file__`."""
    assert getattr(galax, "__file__", None) is None


def test_no_package_attributes() -> None:
    """A namespace package carries no version or `__all__` of its own.

    `galax.__version__` was removed with `galax/__init__.py`; use
    `importlib.metadata.version("galax")` instead.
    """
    for attr in ("__version__", "__version_tuple__", "__all__"):
        assert not hasattr(galax, attr)
