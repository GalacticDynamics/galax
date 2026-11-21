from typing import Any, override

import mpmath as mp
import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..optional_deps import OptDeps
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin
from galax.potential import leesutotriaxialnfw as ls


class ShapeA1ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a1(self) -> u.Quantity[""]:
        return u.Q(1.0, "")

    # =====================================================

    def test_a1_constant(self, pot_cls, fields):
        """Test the `a1` parameter."""
        fields["a1"] = u.Q(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a1(t=u.Q(0, "Myr")) == u.Q(1.0, "")

    def test_a1_userfunc(self, pot_cls, fields):
        """Test the `a1` parameter."""

        def cos_a1(t: u.Quantity["time"]) -> ParametricQuantity[""]:
            return u.Q(jnp.abs(jnp.cos(t.ustrip("Myr"))) + 1.2, "")

        fields["a1"] = cos_a1
        fields["a2"] = u.Q(1.1, "")
        fields["a3"] = u.Q(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a1(t=u.Q(0, "Myr")) == u.Q(2.2, "")


class ShapeA2ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a2(self) -> u.Quantity[""]:
        return u.Q(1.0, "")

    # =====================================================

    def test_a2_constant(self, pot_cls, fields):
        """Test the `a2` parameter."""
        fields["a2"] = u.Q(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a2(t=u.Q(0, "Myr")) == u.Q(1.0, "")

    def test_a2_userfunc(self, pot_cls, fields):
        """Test the `a2` parameter."""

        def cos_a2(t: u.Quantity["time"]) -> ParametricQuantity[""]:
            return u.Q(jnp.abs(jnp.cos(t.ustrip("Myr"))) / 2 + 1.1, "")

        fields["a1"] = u.Q(2, "")
        fields["a2"] = cos_a2
        fields["a3"] = u.Q(1, "")
        pot = pot_cls(**fields)
        assert pot.a2(t=u.Q(0, "Myr")) == u.Q(1.6, "")


class ShapeA3ParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_a3(self) -> u.Quantity[""]:
        return u.Q(1.0, "")

    # =====================================================

    def test_a3_constant(self, pot_cls, fields):
        """Test the `a3` parameter."""
        fields["a3"] = u.Q(1.0, "")
        pot = pot_cls(**fields)
        assert pot.a3(t=u.Q(0, "Myr")) == u.Q(1.0, "")

    def test_a3_userfunc(self, pot_cls, fields):
        """Test the `a3` parameter."""

        def cos_a3(t: u.Quantity["time"]) -> ParametricQuantity[""]:
            return u.Q(jnp.abs(jnp.cos(t.ustrip("Myr"))) + 0.1, "")

        fields["a1"] = u.Q(1.3, "")
        fields["a2"] = u.Q(1.2, "")
        fields["a3"] = cos_a3
        pot = pot_cls(**fields)
        assert pot.a3(t=u.Q(0, "Myr")) == u.Q(1.1, "")


################################################################################
# An independent reference for the potential and the density.
#
# Every function of the `leesutotriaxialnfw` module is built from `potential`
# (the density is Poisson's equation applied to its autodiff Laplacian), so the
# conformance tests cannot see an error in either. This is the Lee & Suto (2003)
# first-order potential, eqs. 19-22 and appendix A with alpha = 1 (the
# formulas of the class docstring), and the closed-form density that is its
# Laplacian over 4 pi G, in mpmath at 30 digits. Nothing of the module's code
# (series, autodiff, Poisson kernel) is shared with it.

mp.mp.dps = 30

G_GAL = 4.498502151469554e-12  # kpc^3 / (Msun Myr^2); a literal, so the test is
# independent of the constants of the installed astropy.
M_LS, RS_LS = 1e11, 15.0  # Msun, kpc
AXES = (1.0, 0.8, 0.6)


def _params(a1: float, a2: float, a3: float) -> dict[str, Any]:
    return {
        "G": np.float64(G_GAL),
        "m": np.float64(M_LS),
        "r_s": np.float64(RS_LS),
        "a1": np.float64(a1),
        "a2": np.float64(a2),
        "a3": np.float64(a3),
    }


def _lee_suto(
    axes: tuple[float, float, float], xyz: tuple[float, float, float]
) -> tuple[mp.mpf, mp.mpf]:
    """``(Phi, rho)`` at ``xyz`` in kpc, in galactic units, from the formulas."""
    a1, a2, a3 = (mp.mpf(a) for a in axes)
    x, y, z = (mp.mpf(c) for c in xyz)
    g, m, rs = mp.mpf(G_GAL), mp.mpf(M_LS), mp.mpf(RS_LS)

    eb2, ec2 = 1 - (a2 / a1) ** 2, 1 - (a3 / a1) ** 2
    ln2 = mp.log(2)
    phi0 = g * m / rs / (ln2 - mp.mpf(1) / 2 + (ln2 - mp.mpf(3) / 4) * (eb2 + ec2))

    r2 = x**2 + y**2 + z**2
    s = mp.sqrt(r2) / rs
    ang = (eb2 * y**2 + ec2 * z**2) / (2 * r2)  # (eb2 y^2 + ec2 z^2) / (2 r^2)
    log1p = mp.log(1 + s)
    f1 = -log1p / s
    f2 = (
        -mp.mpf(1) / 3
        + (2 * s**2 - 3 * s + 6) / (6 * s**2)
        + (1 / s - 1 / s**3) * log1p
    )
    f3 = (s**2 - 3 * s - 6) / (2 * s**2 * (1 + s)) + 3 * log1p / s**3
    phi = phi0 * (f1 + (eb2 + ec2) / 2 * f2 + ang * f3)

    rho_s = phi0 / (4 * mp.pi * g * rs**2)
    rho = rho_s / (s * (1 + s) ** 2) * (1 - ang * (1 + 3 * s) / (1 + s))
    return phi, rho


# (axes, xyz) in kpc, r_s = 15 kpc. Distinct x, y, z and a2 / a1 != a3 / a1
# everywhere: inside and outside r_s, near the centre, and either side of the
# s = 2 switch between the series and the far-field form of F2 and F3 (r = 30
# kpc).
LS_CASES = (
    ((1.0, 0.8, 0.6), (3.0, -9.0, 5.0)),  # s = 0.72
    ((1.0, 0.8, 0.6), (-20.0, 14.0, 31.0)),  # s = 2.6
    ((2.0, 1.5, 0.9), (6.0, 2.0, -11.0)),  # s = 0.87
    ((1.0, 0.95, 0.5), (40.0, -25.0, 8.0)),  # s = 3.2
    ((1.0, 0.8, 0.6), (0.4, -0.7, 0.3)),  # s = 0.06
    ((1.0, 0.7, 0.6), (20.0, 15.0, -12.0)),  # s = 1.85, series
    ((1.0, 0.7, 0.6), (22.0, 16.0, -13.0)),  # s = 2.01, far field
)


@pytest.mark.parametrize(("axes", "xyz"), LS_CASES)
def test_reference_density_is_poisson(
    axes: tuple[float, float, float], xyz: tuple[float, float, float]
) -> None:
    """The closed-form density in the reference is the Laplacian of its potential.

    This checks the reference itself (a central finite difference at 30 digits,
    h = 1e-5, whose truncation error is ~ h^2 / 12 of the fourth derivative).
    """
    h = mp.mpf("1e-5")
    lap = 0
    for i in range(3):
        up, dn = list(map(mp.mpf, xyz)), list(map(mp.mpf, xyz))
        up[i] += h
        dn[i] -= h
        lap += (
            _lee_suto(axes, tuple(up))[0]
            + _lee_suto(axes, tuple(dn))[0]
            - 2 * _lee_suto(axes, xyz)[0]
        ) / h**2
    rho = _lee_suto(axes, xyz)[1]
    assert abs(lap / (4 * mp.pi * G_GAL) - rho) < 1e-7 * abs(rho)


@pytest.mark.parametrize(("axes", "xyz"), LS_CASES)
def test_kernel_matches_reference(
    axes: tuple[float, float, float], xyz: tuple[float, float, float]
) -> None:
    """Potential to 1e-10 and density to 1e-8 of the independent reference.

    The potential is the closed form in both, so it agrees to round-off (the
    cancelling terms of F2 and F3 are why the tolerance is not 1e-14). The module
    density is the autodiff Laplacian, which carries the round-off of two
    differentiations; it is ~ 1e-14 off here, and 1e-8 is not limited by finite
    differences or by a truncated series (there are none in either).
    """
    phi, rho = _lee_suto(axes, xyz)
    p, q = _params(*axes), np.asarray(xyz, dtype=np.float64)
    np.testing.assert_allclose(ls.potential(p, q), float(phi), rtol=1e-10)
    np.testing.assert_allclose(ls.density(p, q), float(rho), rtol=1e-8)


# Frozen values at one asymmetric point, for G = 4.498502151469554e-12
# kpc^3 / (Msun Myr^2), m = 1e11 Msun, r_s = 15 kpc, (a1, a2, a3) = (1, 0.8, 0.6),
# at (4, -3, 12) kpc. They are the potential (kpc^2 / Myr^2) and the density
# (Msun / kpc^3) of `_lee_suto` above (30-digit mpmath, no module code), printed
# with 17 digits. Swapping a2 and a3 changes them by 2 % and 47 %.
GOLDEN_XYZ = (4.0, -3.0, 12.0)
GOLDEN_PHI = -0.096910713462734843
GOLDEN_RHO = 2610340.3612071256


def test_golden_values() -> None:
    """The potential and density at an asymmetric point are frozen."""
    p, q = _params(*AXES), np.asarray(GOLDEN_XYZ, dtype=np.float64)
    np.testing.assert_allclose(ls.potential(p, q), GOLDEN_PHI, rtol=1e-12)
    np.testing.assert_allclose(ls.density(p, q), GOLDEN_RHO, rtol=1e-10)


def test_golden_point_sees_the_axes() -> None:
    """Swapping a2 and a3 changes the potential and density at the golden point."""
    phi, rho = _lee_suto(AXES, GOLDEN_XYZ)
    phi_sw, rho_sw = _lee_suto((AXES[0], AXES[2], AXES[1]), GOLDEN_XYZ)
    assert abs(phi_sw - phi) > 1e-3 * abs(phi)
    assert abs(rho_sw - rho) > 1e-3 * abs(rho)


################################################################################


class TestLeeSutoTriaxialNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    ShapeA1ParameterMixin,
    ShapeA2ParameterMixin,
    ShapeA3ParameterMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.LeeSutoTriaxialNFWPotential]:
        return gp.LeeSutoTriaxialNFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_a1: u.Quantity,
        field_a2: u.Quantity,
        field_a3: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "a1": field_a1,
            "a2": field_a2,
            "a3": field_a3,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-9.68797618, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/leesutotriaxialnfw", atol=1e-8
    )
    def test_gradient(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(4.89753338e09, pot.units["mass density"])
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/leesutotriaxialnfw", atol=1e-8
    )
    def test_hessian(self, pot: gp.LeeSutoTriaxialNFWPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/leesutotriaxialnfw", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ==========================================================================
    # I/O

    @pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            # ("hessian", "hessian", 1e-8),  # No hessian method in gala!
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
