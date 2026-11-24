from typing import Any, ClassVar, override

import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..optional_deps import OptDeps
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterRSMixin,
    assert_gaussian_matches_galpy,
)
from galax.potential import gaussian

###############################################################################


def test_module_functions_match_mpmath() -> None:
    """``potential``, ``mass_enclosed`` and ``density`` against mpmath quadrature.

    The reference integrates the Gaussian density itself, with no closed form:
    ``M(<r) = int_0^r 4 pi s^2 rho ds`` and the Poisson integral
    ``Phi(r) = -G [M(<r) / r + 4 pi int_r^inf rho s ds]``, both at 30 digits.
    Nothing limits rtol 1e-12 by cancellation: the module sums a series of
    positive terms for ``mass_enclosed`` at ``r <= r_s`` and the direct
    difference cancels by at most a factor 3.5 beyond.
    """
    pot = gp.GaussianPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    p = pot.params(0.0)
    g, m_tot, r_s = (float(p[k]) for k in ("G", "m_tot", "r_s"))

    rho0 = mp.mpf(m_tot) / ((2 * mp.pi) ** mp.mpf("1.5") * mp.mpf(r_s) ** 3)

    def rho(s: mp.mpf) -> mp.mpf:
        return rho0 * mp.exp(-(s**2) / (2 * mp.mpf(r_s) ** 2))

    for x in (1e-4, 1e-2, 0.1, 0.3, 1.0, 2.0, 3.7, 8.0, 20.0):
        r = x * r_s
        rr = mp.mpf(r)
        mass = mp.quad(lambda s: 4 * mp.pi * s**2 * rho(s), [0, rr])
        outer = mp.quad(lambda s: 4 * mp.pi * s * rho(s), [rr, 10 * rr + 20 * r_s])
        phi = -mp.mpf(g) * (mass / rr + outer)
        kw = {"rtol": 1e-12, "err_msg": f"x={x}"}
        np.testing.assert_allclose(gaussian.potential(p, r), float(phi), **kw)
        np.testing.assert_allclose(gaussian.mass_enclosed(p, r), float(mass), **kw)
        np.testing.assert_allclose(gaussian.density(p, r), float(rho(rr)), **kw)
        # The closed form in the docstring agrees with the quadrature.
        closed = -mp.mpf(g) * m_tot * mp.erf(rr / (mp.sqrt(2) * r_s)) / rr
        assert abs(closed - phi) <= mp.mpf("1e-20") * abs(phi), x


def test_potential_at_origin_matches_mpmath() -> None:
    """``Phi(0) = -4 pi G int_0^inf rho s ds``, on the module and the class.

    The module substitutes the 0/0 limit ``-G m sqrt(2/pi) / r_s`` at r = 0
    exactly; the quadrature does not use it. Measured error 1.5e-16, a margin
    of 600 at rtol 1e-13.
    """
    pot = gp.GaussianPotential(
        m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
    )
    p = pot.params(0.0)
    g, m_tot, r_s = (mp.mpf(float(p[k])) for k in ("G", "m_tot", "r_s"))

    rho0 = m_tot / ((2 * mp.pi) ** mp.mpf("1.5") * r_s**3)
    outer = mp.quad(
        lambda s: 4 * mp.pi * s * rho0 * mp.exp(-(s**2) / (2 * r_s**2)), [0, mp.inf]
    )
    want = float(-g * outer)

    np.testing.assert_allclose(gaussian.potential(p, 0.0), want, rtol=1e-13)
    got = pot.potential(jnp.zeros(3), 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_rho0_of_m_and_m_of_rho0_match_mpmath() -> None:
    """``m_tot = int_0^inf 4 pi s^2 rho0 exp(-s^2 / 2 r_s^2) ds`` at r_s != 1.

    At r_s = 1 (the doctests) an error in the power of r_s would not show.
    Measured error <= 1.6e-16, a margin of 600 at rtol 1e-13.
    """
    rho0, r_s = 3.7e8, 2.3
    rs = mp.mpf(r_s)
    mass = mp.quad(
        lambda s: 4 * mp.pi * s**2 * rho0 * mp.exp(-(s**2) / (2 * rs**2)), [0, mp.inf]
    )
    np.testing.assert_allclose(
        gaussian.m_of_rho0({"rho0": rho0, "r_s": r_s}), float(mass), rtol=1e-13
    )
    np.testing.assert_allclose(
        gaussian.rho0_of_m({"m_tot": float(mass), "r_s": r_s}), rho0, rtol=1e-13
    )


class TestGaussianPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.GaussianPotential]:
        return gp.GaussianPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-1.20205545, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/gaussian", atol=1e-8
    )
    def test_gradient(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(5.78986720e07, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/gaussian", atol=1e-8
    )
    def test_hessian(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/gaussian", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy")
    def test_method_galpy(self, pot: gp.GaussianPotential, x: gt.QuSz3) -> None:
        """Test the equivalence of potential/density between galpy and galax."""
        assert_gaussian_matches_galpy(pot, x)
