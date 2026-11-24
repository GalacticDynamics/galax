from typing import Any, ClassVar

import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..optional_deps import OptDeps
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin, mp_spherical_potential
from galax.potential import burkert


def test_rho0() -> None:
    """``burkert.rho0`` is the central density: ``m / (pi C r_s^3)``."""
    pot = gp.BurkertPotential(
        m=u.Q(1e11, "Msun"), r_s=u.Q(2.5, "kpc"), units="galactic"
    )
    got = burkert.rho0(pot.m(0.0), pot.r_s(0.0)).ustrip("Msun / kpc3")
    # C = 3 ln 2 - pi / 2, from the docstring.
    c = 3 * np.log(2) - np.pi / 2
    expect = 1e11 / (np.pi * c * 2.5**3)
    np.testing.assert_allclose(got, expect, rtol=1e-12)
    np.testing.assert_allclose(got, burkert.density(pot.params(0.0), 0.0), rtol=1e-12)
    np.testing.assert_allclose(pot.rho0(0.0).ustrip("Msun / kpc3"), got, rtol=1e-12)


def test_from_central_density() -> None:
    """``from_central_density`` sets ``m = pi C rho_0 r_s^3`` and inverts ``rho0``."""
    pot = gp.BurkertPotential.from_central_density(
        u.Q(2.0e6, "Msun / kpc3"), u.Q(2.5, "kpc"), units="galactic"
    )
    c = 3 * np.log(2) - np.pi / 2
    np.testing.assert_allclose(
        pot.m(0.0).ustrip("Msun"), np.pi * c * 2.0e6 * 2.5**3, rtol=1e-12
    )
    np.testing.assert_allclose(pot.rho0(0.0).ustrip("Msun / kpc3"), 2.0e6, rtol=1e-12)


@pytest.mark.parametrize("r_s", [0.7, 3.0])
@pytest.mark.parametrize("x", [0.01, 0.3, 1.0, 3.0, 30.0, 1000.0])
def test_potential_matches_mpmath(r_s: float, x: float) -> None:
    """Phi(r) from the quadrature of the Burkert density, at 1e-11 relative.

    The shape is ``1 / ((1 + x)(1 + x^2))`` (Burkert 1995, eq. 2) and the
    normalisation comes from its own mass integral. Measured error < 1e-13.
    """
    m, g = 1e11, 1.3
    ref = mp_spherical_potential(
        lambda s: 1 / ((1 + s / mp.mpf(r_s)) * (1 + (s / mp.mpf(r_s)) ** 2)),
        r_s,
        m,
        g,
        x * r_s,
        r_norm=r_s,
    )
    got = burkert.potential({"G": g, "m": m, "r_s": r_s}, x * r_s)
    np.testing.assert_allclose(got, ref, rtol=1e-11)


class TestBurkertPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
):
    """Test the `galax.potential.BurkertPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = OptDeps.GALA.installed

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.BurkertPotential]:
        return gp.BurkertPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Q(-15.76623941, "kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/burkert", atol=1e-8
    )
    def test_gradient(self, pot: gp.BurkertPotential, x: gt.Sz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.BurkertPotential, x: gt.Sz3) -> None:
        expect = u.Q(8.79860325e09, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/burkert", atol=1e-8
    )
    def test_hessian(self, pot: gp.BurkertPotential, x: gt.Sz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/burkert", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.Sz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            # ("hessian", "hessian", 1e-8),  # TODO: get gala and galax to agree
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
