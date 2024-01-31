"""Test the :mod:`galax.utils.dataclasses` module."""

import dataclasses

import equinox as eqx

from galax.utils.dataclasses import field

COMP_FIELDS: tuple[str, ...] = (
    "name",
    "type",
    "default",
    "default_factory",
    "init",
    "repr",
    "hash",
    "compare",
    "metadata",
)


def test_field() -> None:
    """Test the :func:`field` function."""
    # Basic test that it returns the output of `dataclasses.field`.
    f = field(dimensions="length")
    fcomp = dataclasses.field(metadata={"dimensions": "length"})
    for n in COMP_FIELDS:
        assert getattr(f, n) == getattr(fcomp, n), n

    # Test that it matches `equinox.field``.
    f = field(static=True, converter=lambda x: x, dimensions="length")
    fcomp = eqx.field(
        static=True,
        converter=f.metadata["converter"],
        metadata={"dimensions": "length"},
    )
    for n in COMP_FIELDS:
        assert getattr(f, n) == getattr(fcomp, n), n


def test_field_on_equinox_object() -> None:
    """Test that `field` works on Equinox objects."""

    class A(eqx.Module):
        x: int = field(converter=lambda x: int(x))

    obj = A(1.0)  # float -> int
    assert isinstance(obj.x, int)
