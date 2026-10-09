"""Tests for `galax.potential`'s translation transforms."""

import jax.numpy as jnp
import pytest

import unxt as u
import unxt.unitsystems as usys

import galax.potential as gp


@pytest.fixture
def path() -> tuple[u.Quantity, u.Quantity]:
    """Return a straight-line trajectory along x, in Myr and kpc."""
    t = u.Q(jnp.linspace(0.0, 1.0, 10), "Myr")
    xyz = u.Q(
        jnp.stack([t.ustrip("Myr"), jnp.zeros(t.shape), jnp.zeros(t.shape)], axis=-1),
        "kpc",
    )
    return t, xyz


def test_from_derives_a_unitsystem_when_units_is_omitted(path) -> None:
    """`units` defaults to one derived from the inputs, not a dimensionless one.

    The values are stripped against the derived system, so storing anything
    else leaves the parameter's numbers and its units disagreeing -- which
    construction does not notice and evaluation reports as a confusing
    `UnitConversionError` much later.
    """
    t, xyz = path
    param = gp.params.TimeDependentTranslationParameter.from_(t, xyz)

    assert not isinstance(param.units, usys.DimensionlessUnitSystem)
    assert param.units["time"] == u.unit("Myr")
    assert param.units["length"] == u.unit("kpc")


def test_omitting_units_evaluates_like_passing_them(path) -> None:
    """The default path must agree with the explicit one, not merely not raise."""
    t, xyz = path
    explicit = gp.params.TimeDependentTranslationParameter.from_(
        t, xyz, units=u.unitsystem("kpc", "Myr", "Msun", "rad")
    )
    derived = gp.params.TimeDependentTranslationParameter.from_(t, xyz)

    at = u.Q(0.5, "Myr")
    got, want = derived(at), explicit(at)
    assert got.unit == want.unit
    assert jnp.allclose(got.value, want.value, atol=1e-10)
