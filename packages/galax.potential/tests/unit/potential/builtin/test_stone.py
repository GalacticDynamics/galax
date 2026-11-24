from typing import Any

import mpmath as mp
import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterFieldMixin,
    ParameterMTotMixin,
    mp_spherical_potential,
)
from galax.potential._src.builtin import stoneostriker15


def test_rhoc_of_mtot_matches_mpmath() -> None:
    """``M(inf) = m_tot``: mpmath integral of S&O15 Eq. 1 with ``rhoc_of_mtot``.

    r_c != 1: at r_c = 1 (the class fixture) an error in the power of r_c would
    not show. Measured error 0 (rtol 1e-13).
    """
    for r_c, r_h in ((0.8, 5.0), (2.5, 0.7)):
        p = {"m_tot": 1e11, "r_c": r_c, "r_h": r_h}
        rho_c = mp.mpf(float(stoneostriker15.rhoc_of_mtot(p)))
        rc, rh = mp.mpf(r_c), mp.mpf(r_h)

        def dm(s, rho_c=rho_c, rc=rc, rh=rh):
            return (
                4 * mp.pi * s**2 * rho_c / ((1 + (s / rc) ** 2) * (1 + (s / rh) ** 2))
            )

        mass = mp.quad(dm, [0, rc, rh, mp.inf])
        np.testing.assert_allclose(float(mass), p["m_tot"], rtol=1e-13)


@pytest.mark.parametrize(("r_c", "r_h"), [(0.8, 5.0), (2.5, 0.7)])
@pytest.mark.parametrize("r", [0.01, 0.3, 1.0, 3.0, 30.0, 1000.0])
def test_potential_matches_mpmath(r_c: float, r_h: float, r: float) -> None:
    """Phi(r) from the quadrature of S&O15 Eq. 1, at 1e-11 relative.

    Both orders of r_c and r_h, and r far outside both. Measured error < 1e-13.
    """
    m, g = 1e11, 1.3
    rc, rh = mp.mpf(r_c), mp.mpf(r_h)
    ref = mp_spherical_potential(
        lambda s: 1 / ((1 + (s / rc) ** 2) * (1 + (s / rh) ** 2)),
        min(r_c, r_h),
        m,
        g,
        r,
    )
    got = stoneostriker15.potential({"G": g, "m_tot": m, "r_c": r_c, "r_h": r_h}, r)
    np.testing.assert_allclose(got, ref, rtol=1e-11)


class ParameterRCMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> u.Quantity["length"]:
        return u.Q(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = u.Q(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(1.0, "kpc")

    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""

        def cos_r_c(t: u.Quantity["time"]) -> ParametricQuantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_c"] = cos_r_c
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


class ParameterRHMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_h(self) -> u.Quantity["length"]:
        return u.Q(10.0, "kpc")

    # =====================================================

    def test_r_h_constant(self, pot_cls, fields):
        """Test the `r_h` parameter."""
        fields["r_h"] = u.Q(11.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Q(0, "Myr")) == u.Q(11.0, "kpc")

    def test_r_h_userfunc(self, pot_cls, fields):
        """Test the `r_h` parameter."""

        def cos_r_h(t: u.Quantity["time"]) -> ParametricQuantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_h"] = cos_r_h
        pot = pot_cls(**fields)
        assert pot.r_h(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


class TestStoneOstriker15Potential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRCMixin,
    ParameterRHMixin,
):
    """Test the `galax.potential.StoneOstriker15Potential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.StoneOstriker15Potential]:
        return gp.StoneOstriker15Potential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_c: u.Quantity,
        field_r_h: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_c": field_r_c,
            "r_h": field_r_h,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.51579523, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_gradient(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        expect = u.Q(3.25886848e08, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_hessian(self, pot: gp.StoneOstriker15Potential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/stone", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
