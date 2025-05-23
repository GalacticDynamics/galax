from typing import Any

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
)


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["angle"]:
        return u.Quantity(0.9, "rad")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = u.Quantity(1.0, "rad")
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == u.Quantity(1.0, "rad")

    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""

        def cos_alpha(t: u.Quantity["time"]) -> u.Quantity["angle"]:
            return u.Quantity(10 * jnp.cos(t.ustrip("Myr")), "deg")

        fields["alpha"] = cos_alpha
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Quantity(0, "Myr")) == u.Quantity(10, "deg")


class TestLongMuraliBarPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
    AlphaParameterMixin,
):
    """Test the `galax.potential.LongMuraliBarPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.LongMuraliBarPotential]:
        return gp.LongMuraliBarPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_b: u.Quantity,
        field_c: u.Quantity,
        field_alpha: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "alpha": field_alpha,
            "a": field_a,
            "b": field_b,
            "c": field_c,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-0.9494695, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity([0.04017315, 0.08220449, 0.16854858], "kpc / Myr2")
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(2.02402357e08, "solMass / kpc3")
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.03722412, -0.01077521, -0.02078279],
                [-0.01077521, 0.02101076, -0.04320745],
                [-0.02078279, -0.04320745, -0.0467931],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.0334102, -0.01077521, -0.02078279],
                [-0.01077521, 0.01719683, -0.04320745],
                [-0.02078279, -0.04320745, -0.05060703],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
