"""Test that `galax` is a PEP 420 namespace package."""

from importlib.metadata import version

import pytest

import galax


def test_version() -> None:
    """The distribution still reports a version, via metadata."""
    assert version("galax")


def test_is_namespace_package() -> None:
    """`galax` is a namespace: no `__init__.py`, so no `__file__`."""
    assert getattr(galax, "__file__", None) is None
    assert type(galax.__path__).__name__ == "_NamespacePath"


def test_no_package_attributes() -> None:
    """A namespace package carries no version or `__all__` of its own.

    `galax.__version__` was removed with `galax/__init__.py`; use
    `importlib.metadata.version("galax")` instead.
    """
    for attr in ("__version__", "__version_tuple__", "__all__"):
        assert not hasattr(galax, attr)


@pytest.mark.parametrize("name", ["coordinates", "potential", "dynamics"])
def test_portions_are_importable(name: str) -> None:
    """Each portion imports on its own, without a parent `__init__`."""
    mod = __import__(f"galax.{name}", fromlist=[name])
    assert mod.__name__ == f"galax.{name}"
