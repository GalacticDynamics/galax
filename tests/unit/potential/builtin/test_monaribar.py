from typing import Any, ClassVar

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ..test_core import AbstractSinglePotential_Test


class TestMonariEtAl2016BarPotential(
    AbstractSinglePotential_Test,
):
    """Test the `galax.potential.MonariEtAl2016BarPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MonariEtAl2016BarPotential]:
        return gp.MonariEtAl2016BarPotential

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["dimensionless"]:
        return u.Quantity(0.01, "")

    @pytest.fixture(scope="class")
    def field_R0(self) -> u.Quantity["length"]:
        return u.Quantity(8.0, "kpc")

    @pytest.fixture(scope="class")
    def field_v0(self) -> u.Quantity["speed"]:
        return u.Quantity(220.0, "km/s")

    @pytest.fixture(scope="class")
    def field_Rb(self) -> u.Quantity["length"]:
        return u.Quantity(3.5, "kpc")

    @pytest.fixture(scope="class")
    def field_phi_b(self) -> u.Quantity["angle"]:
        return u.Quantity(25, "deg")

    @pytest.fixture(scope="class")
    def field_Omega(self) -> u.Quantity["frequency"]:
        return u.Quantity(52.2, "km/(s kpc)")

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_alpha: u.AbstractQuantity,
        field_R0: u.AbstractQuantity,
        field_v0: u.AbstractQuantity,
        field_Rb: u.AbstractQuantity,
        field_phi_b: u.AbstractQuantity,
        field_Omega: u.AbstractQuantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "alpha": field_alpha,
            "R0": field_R0,
            "v0": field_v0,
            "Rb": field_Rb,
            "phi_b": field_phi_b,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        got = pot.potential(x, t=0)
        exp = u.Quantity(-0.00013381, pot.units["specific energy"])
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_gradient(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        exp = u.Quantity(
            [-0.00046465, 0.00021799, 0.00014337], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_density(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Quantity(4.79482196e-10, "Msun / kpc3")
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        got = pot.hessian(x, t=0)
        exp = u.Quantity(
            [
                [2.38472072e-04, 9.40281053e-05, 4.77361985e-04],
                [9.40281053e-05, -7.12096378e-05, -2.74522943e-04],
                [4.77361985e-04, -2.74522943e-04, -1.67262435e-04],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        got = pot.tidal_tensor(x, t=0)
        exp = u.Quantity(
            [
                [2.38472072e-04, 9.40281053e-05, 4.77361985e-04],
                [9.40281053e-05, -7.12096378e-05, -2.74522943e-04],
                [4.77361985e-04, -2.74522943e-04, -1.67262435e-04],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(got, exp, atol=u.Quantity(1e-8, exp.unit))
