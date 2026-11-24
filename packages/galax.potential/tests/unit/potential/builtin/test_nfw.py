from typing import Any, override

import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin
from galax.potential import nfw

###############################################################################


def test_rho0_of_m_and_m_of_rho0_match_mpmath() -> None:
    """``M(<r_s) = m (ln 2 - 1/2)`` with ``M`` the mpmath integral of the density.

    The density is ``rho_s / (x (1 + x)^2)``; the integral does not use
    ``m = 4 pi rho_s r_s^3``. At r_s = 1 (the doctests) an error in the power
    of r_s would not show. Measured error <= 1.6e-16, a margin of 600 at
    rtol 1e-13.
    """
    rho_s, r_s = 3.7e8, 2.3
    rs = mp.mpf(r_s)
    mass = mp.quad(
        lambda s: 4 * mp.pi * s**2 * rho_s / (s / rs * (1 + s / rs) ** 2), [0, rs]
    )
    m = float(mass / (mp.log(2) - mp.mpf("0.5")))
    np.testing.assert_allclose(
        nfw.m_of_rho0({"rho0": rho_s, "r_s": r_s}), m, rtol=1e-13
    )
    np.testing.assert_allclose(nfw.rho0_of_m({"m": m, "r_s": r_s}), rho_s, rtol=1e-13)


class TestNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
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
        expect = u.Q(-1.87120528, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/nfw", atol=1e-8
    )
    def test_gradient(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(9.45944763e08, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/nfw", atol=1e-8
    )
    def test_hessian(self, pot: gp.NFWPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    def test_special_inits(self) -> None:
        """Test specialized initializers of the NFW potential."""
        pot = gp.NFWPotential.from_circular_velocity(
            v_c=u.Q(220.0, "km/s"), r_s=u.Q(15.0, "kpc"), units="galactic"
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.23399598
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_circular_velocity(
            v_c=u.Q(220, "km/s"),
            r_s=u.Q(15, "kpc"),
            r_ref=u.Q(20, "kpc"),
            units="galactic",
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.21843999
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_M200_c(
            M200=u.Q(1e12, "Msun"), c=15.0, units="galactic"
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -0.15451932
        assert jnp.allclose(got, exp, atol=1e-8)

        pot = gp.NFWPotential.from_M200_c(
            M200=u.Q(1e12, "Msun"), c=15.0, rho_c=u.Q(1, "g / m3"), units="galactic"
        )
        got = pot.potential(jnp.array([1.0, 2.0, 3.0]), 0.0)
        exp = -10.73095438
        assert jnp.allclose(got, exp, atol=1e-8)

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/nfw", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
