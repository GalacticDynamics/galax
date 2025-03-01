from typing import Any
from typing_extensions import override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMMixin, ParameterScaleRadiusMixin

###############################################################################


class TestNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterScaleRadiusMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.NFWPotential]:
        return gp.NFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(-1.87120528, pot.units["specific energy"])
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [0.06589185, 0.1317837, 0.19767556], pot.units["acceleration"]
        )
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Quantity(9.45944763e08, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Quantity(1e-8, exp.unit))

    def test_hessian(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [0.05559175, -0.02060021, -0.03090031],
                [-0.02060021, 0.02469144, -0.06180062],
                [-0.03090031, -0.06180062, -0.02680908],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_special_inits(self) -> None:
        """Test specialized initializers of the NFW potential."""
        pot = gp.NFWPotential.from_circular_velocity(
            v_c=u.Quantity(220.0, "km/s"), r_s=u.Quantity(15.0, "kpc"), units="galactic"
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.23399598
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_circular_velocity(
            v_c=u.Quantity(220.0, "km/s"),
            r_s=u.Quantity(15.0, "kpc"),
            r_ref=u.Quantity(20.0, "kpc"),
            units="galactic",
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.21843999
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_M200_c(
            M200=u.Quantity(1e12, "Msun"), c=15.0, units="galactic"
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.15451932
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_M200_c(
            M200=u.Quantity(1e12, "Msun"),
            c=15.0,
            rho_c=u.Quantity(1, "g / m3"),
            units="galactic",
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -10.73095438
        assert jnp.allclose(got, exp, atol=1e-8)

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [0.03776704, -0.02060021, -0.03090031],
                [-0.02060021, 0.00686674, -0.06180062],
                [-0.03090031, -0.06180062, -0.04463378],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )
