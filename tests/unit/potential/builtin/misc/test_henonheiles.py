"""Test the `galax.potential.HenonHeilesPotential` class."""

from typing import Any, ClassVar

import pytest
from plum import convert

import quaxed.numpy as jnp
from unxt import AbstractUnitSystem, Quantity

import galax.potential as gp
import galax.typing as gt
from ...param.test_field import ParameterFieldMixin
from ...test_core import AbstractPotential_Test
from galax.potential import AbstractPotentialBase, HenonHeilesPotential


class ParameterCoeffMixin(ParameterFieldMixin):
    """Test the coeff parameter."""

    @pytest.fixture(scope="class")
    def field_coeff(self) -> Quantity["wavenumber"]:
        return Quantity(1.0, "1/kpc")

    # =====================================================

    def test_coeff_constant(self, pot_cls, fields):
        """Test the `coeff` parameter."""
        fields["coeff"] = 1 / Quantity(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.coeff(t=0) == 1 / Quantity(1.0, "kpc")

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_coeff_userfunc(self, pot_cls, fields):
        """Test the `coeff` parameter."""
        fields["coeff"] = lambda t: t * 1.2
        pot = pot_cls(**fields)
        assert pot.a1(t=0) == 2


# ===================================================================


class ParameterTimeScaleMixin(ParameterFieldMixin):
    """Test the timescale parameter."""

    @pytest.fixture(scope="class")
    def field_timescale(self) -> Quantity["time"]:
        return Quantity(1.0, "Myr")

    # =====================================================

    def test_timescale_constant(self, pot_cls, fields):
        """Test the `timescale` parameter."""
        fields["timescale"] = Quantity(1.0, "Myr")
        pot = pot_cls(**fields)
        assert pot.timescale(t=0) == Quantity(1.0, "Myr")

    def test_timescale_userfunc(self, pot_cls, fields):
        """Test the `timescale` parameter."""

        def func(t: Quantity["time"]) -> Quantity["time"]:
            return Quantity.constructor(t * 1.2, "Myr")

        fields["timescale"] = func
        pot = pot_cls(**fields)
        assert pot.timescale(t=1) == Quantity(1.2, "Myr")


#####################################################################


class TestHenonHeilesPotential(
    AbstractPotential_Test,
    # Parameters
    ParameterCoeffMixin,
    ParameterTimeScaleMixin,
):
    """Test the `galax.potential.HenonHeilesPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False
    # TODO: figure out how to convert this to Gala given the different parameters

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.HenonHeilesPotential]:
        return gp.HenonHeilesPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_coeff: Quantity,
        field_timescale: Quantity,
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "coeff": field_coeff,
            "timescale": field_timescale,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: HenonHeilesPotential, x: gt.Vec3) -> None:
        got = pot.potential(x, t=0)
        exp = Quantity(1.83333333, unit="kpc2 / Myr2")
        assert jnp.isclose(got, exp, atol=Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: HenonHeilesPotential, x: gt.Vec3) -> None:
        got = convert(pot.gradient(x, t=0), Quantity)
        exp = Quantity([5.0, -1, 0], "kpc / Myr2")
        assert jnp.allclose(got, exp, atol=Quantity(1e-8, exp.unit))

    def test_density(self, pot: HenonHeilesPotential, x: gt.Vec3) -> None:
        got = pot.density(x, t=0)
        exp = Quantity(3.53795414e10, "solMass / kpc3")
        assert jnp.isclose(got, exp, atol=Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: HenonHeilesPotential, x: gt.Vec3) -> None:
        got = pot.hessian(x, t=0)
        exp = Quantity([[5.0, 2.0, 0.0], [2.0, -3.0, 0.0], [0.0, 0.0, 0.0]], "1/Myr2")
        assert jnp.allclose(got, exp, atol=Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: AbstractPotentialBase, x: gt.Vec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        got = pot.tidal_tensor(x, t=0)
        exp = Quantity(
            [[4.33333333, 2.0, 0.0], [2.0, -3.66666667, 0.0], [0.0, 0.0, -0.66666667]],
            "1/Myr2",
        )
        assert jnp.allclose(got, exp, atol=Quantity(1e-8, exp.unit))
