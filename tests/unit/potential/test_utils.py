"""Tests for `galax.potential._src.utils` package."""

from dataclasses import replace

from typing import Any

import jax
import pytest
from jax import Array

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.utils import gauss_legendre_nodes


class FieldUnitSystemMixin:
    """Mixin for testing the ``units`` field on a ``Potential``."""

    @pytest.fixture
    def fields_unitless(self, fields: dict[str, Any]) -> dict[str, Array]:
        """Fields with no units."""
        return {
            k: (v.value if isinstance(v, u.Quantity) else v) for k, v in fields.items()
        }

    # ===========================================

    def test_init_units_from_usys(self, pot: gp.AbstractPotential) -> None:
        """Test unit system from UnitSystem."""
        usys = u.unitsystem("km", "s", "Msun", "radian")
        assert replace(pot, units=usys).units == usys

    def test_init_units_from_tuple(self, pot: gp.AbstractPotential) -> None:
        """Test unit system from tuple."""
        units = ("km", "s", "Msun", "radian")
        assert replace(pot, units=units).units == u.unitsystem(*units)

    def test_init_units_from_name(
        self, pot_cls: type[gp.AbstractPotential], fields_unitless: dict[str, Array]
    ) -> None:
        """Test unit system from named string."""
        fields_unitless.pop("units")

        # TODO: sort this out
        # pot = pot_cls(**fields_unitless, units="dimensionless")
        # # assert pot.units == dimensionless

        pot = pot_cls(**fields_unitless, units="solarsystem")
        assert pot.units == u.unitsystems.solarsystem

        pot = pot_cls(**fields_unitless, units="galactic")
        assert pot.units == u.unitsystems.galactic

        with pytest.raises(ValueError, match="invalid_value"):
            pot_cls(**fields_unitless, units="invalid_value")


@pytest.mark.parametrize("order", [3, 8, 16])
def test_gauss_legendre_nodes_follow_the_caller_dtype(order: int) -> None:
    """The cache must not decide the dtype for later callers.

    REGRESSION: the `lru_cache` was keyed on ``(order, interval)`` and held
    the *JAX* arrays, so ``dtype=float`` resolved against whatever
    ``jax_enable_x64`` the first caller happened to have. Under x64 the entry
    was float64, and a float32 caller got float64 back -- while the same call
    at an uncached order correctly gave float32. Same function, same config,
    two dtypes, decided by call order.

    That is why float32 tests elsewhere passed alone and failed in a file:
    earlier tests primed the cache under the suite's forced x64, and the
    float64 nodes then tripped JAX's truncation warning, which
    ``filterwarnings = ["error"]`` turns into a failure.

    Each case primes the cache under x64 *first*, so a regression reproduces
    rather than passing by luck on a cold cache.
    """
    primed, _ = gauss_legendre_nodes(order, (-1.0, 1.0))
    assert primed.dtype == jnp.float64  # the suite forces x64

    with jax.enable_x64(False):  # noqa: FBT003
        cached, w = gauss_legendre_nodes(order, (-1.0, 1.0))
        assert cached.dtype == jnp.float32, "cache handed back the x64 dtype"
        assert w.dtype == jnp.float32

    # ...and the x64 caller is unaffected: same entry, not a recomputed one.
    again, _ = gauss_legendre_nodes(order, (-1.0, 1.0))
    assert again is primed
