from typing import Any

import mpmath as mp
import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import (
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeBMixin,
    ParameterShapeCMixin,
)
from galax.potential import longmuralibar as lm

# An independent reference with b != c.
#
# The fixture of the class tests has b = c, which cannot see a swap of the two
# softening lengths. The reference is the needle integral of Long & Murali
# (1992, ApJ 397, 44), eq. 8a: the potential of a uniform needle of half-length
# a and total mass M along the bar axis, each point of it softened with the
# Miyamoto-Nagai potential of scales (b, c), by `mpmath.quad` at 30 digits. The
# closed-form logarithm (eq. 8b) and the gradient of the code are not used. The
# gradient is a central difference of the integral (step 1e-8, so a truncation
# error of ~1e-16 of the third derivative, and a round-off of 1e-22).

G_GAL = 4.498502151469554e-12  # kpc^3 / (Msun Myr^2), a literal like the units
M_BAR, A_BAR, B_BAR, C_BAR, ALPHA_BAR = 1e10, 4.0, 1.2, 0.7, 0.3  # Msun, kpc, kpc, rad


def _params(b: float = B_BAR, c: float = C_BAR) -> dict[str, Any]:
    return {
        "G": np.float64(G_GAL),
        "m_tot": np.float64(M_BAR),
        "a": np.float64(A_BAR),
        "b": np.float64(b),
        "c": np.float64(c),
        "alpha": np.float64(ALPHA_BAR),
    }


def _needle_integral(
    xyz: tuple[mp.mpf, mp.mpf, mp.mpf], b: float = B_BAR, c: float = C_BAR
) -> mp.mpf:
    """Eq. 8a: ``-G M / 2a int ds / sqrt((x' - s)^2 + y'^2 + (b + zeta)^2)``."""
    g, m, a, bb, cc, al = (mp.mpf(v) for v in (G_GAL, M_BAR, A_BAR, b, c, ALPHA_BAR))
    x, y, z = xyz
    xp, yp = x * mp.cos(al) + y * mp.sin(al), -x * mp.sin(al) + y * mp.cos(al)
    eta2 = yp**2 + (bb + mp.sqrt(cc**2 + z**2)) ** 2
    knots = [-a, xp, a] if -a < xp < a else [-a, a]
    return (
        -g * m / (2 * a) * mp.quad(lambda s: 1 / mp.sqrt((xp - s) ** 2 + eta2), knots)
    )


def _needle_gradient(
    xyz: tuple[float, float, float], b: float = B_BAR, c: float = C_BAR
) -> list[mp.mpf]:
    h = mp.mpf("1e-8")
    out = []
    for i in range(3):
        up, dn = list(map(mp.mpf, xyz)), list(map(mp.mpf, xyz))
        up[i] += h
        dn[i] -= h
        out.append(
            (_needle_integral(tuple(up), b, c) - _needle_integral(tuple(dn), b, c))
            / (2 * h)
        )
    return out


# Off every axis of the bar frame, with distinct x, y and z: inside the needle
# (|x'| < a) and outside it, above and below the plane. z = 0.3 and -3 are of the
# order of c and b, so the softening lengths matter at every point.
LM_POINTS = (
    (1.0, 0.5, 0.3),
    (3.0, -2.0, 1.2),
    (-6.0, -4.0, 0.8),
    (10.0, 4.0, -3.0),
    (-2.5, 0.7, -0.4),
)


@pytest.mark.parametrize("xyz", LM_POINTS)
def test_matches_needle_integral(xyz: tuple[float, float, float]) -> None:
    """Potential to 1e-10 and gradient to 1e-8 of the integral, with b != c."""
    p, q = _params(), np.asarray(xyz, dtype=np.float64)
    ref = _needle_integral(tuple(map(mp.mpf, xyz)))
    np.testing.assert_allclose(lm.potential(p, q), float(ref), rtol=1e-10)
    grad = [float(g) for g in _needle_gradient(xyz)]
    np.testing.assert_allclose(lm.gradient(p, q), grad, rtol=1e-8)


# Far down the bar axis, in the bar frame (x', y', z), where the potential switches to
# the log1p form (the ratio of the two logarithm arguments is above 0.5). None of
# LM_POINTS gets there: their largest ratio is 0.49.
LM_FAR_POINTS_BAR_FRAME = ((30.0, 0.0, 0.0), (-25.0, 6.0, 4.0), (60.0, -15.0, 9.0))


@pytest.mark.parametrize("xyz_bar", LM_FAR_POINTS_BAR_FRAME)
def test_far_field_matches_needle_integral(xyz_bar: tuple[float, float, float]) -> None:
    """Potential to 1e-11 of the integral in the far branch, rotated by alpha."""
    xp, yp, z = xyz_bar
    ca, sa = np.cos(ALPHA_BAR), np.sin(ALPHA_BAR)
    xyz = (xp * ca - yp * sa, xp * sa + yp * ca, z)
    ref = _needle_integral(tuple(map(mp.mpf, xyz)))
    got = lm.potential(_params(), np.asarray(xyz, dtype=np.float64))
    np.testing.assert_allclose(got, float(ref), rtol=1e-11)


@pytest.mark.parametrize("xyz", LM_POINTS)
def test_reference_sees_b_and_c(xyz: tuple[float, float, float]) -> None:
    """Swapping b and c changes the reference potential by more than 1e-3."""
    q = tuple(map(mp.mpf, xyz))
    phi, swapped = _needle_integral(q), _needle_integral(q, C_BAR, B_BAR)
    assert abs(swapped - phi) > 1e-3 * abs(phi)


def test_class_uses_b_and_c() -> None:
    """The class potential with b != c equals the integral, through its fields."""
    pot = gp.LongMuraliBarPotential(
        m_tot=u.Q(M_BAR, "Msun"),
        a=u.Q(A_BAR, "kpc"),
        b=u.Q(B_BAR, "kpc"),
        c=u.Q(C_BAR, "kpc"),
        alpha=u.Q(ALPHA_BAR, "rad"),
        units="galactic",
    )
    xyz = LM_POINTS[1]
    got = pot.potential(u.Q(list(xyz), "kpc"), t=0).ustrip("kpc2 / Myr2")
    ref = _needle_integral(tuple(map(mp.mpf, xyz)))
    # G of the class is the astropy value, which differs from the literal by 1e-7
    np.testing.assert_allclose(got, float(ref), rtol=1e-6)


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["angle"]:
        return u.Q(0.9, "rad")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = u.Q(1.0, "rad")
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Q(0, "Myr")) == u.Q(1.0, "rad")

    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""

        def cos_alpha(t: u.Quantity["time"]) -> ParametricQuantity["angle"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "deg")

        fields["alpha"] = cos_alpha
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Q(0, "Myr")) == u.Q(10, "deg")


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
        expect = u.Q(-0.9494695, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/longmuralibar", atol=1e-8
    )
    def test_gradient(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        expect = u.Q(2.02402357e08, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/longmuralibar", atol=1e-8
    )
    def test_hessian(self, pot: gp.LongMuraliBarPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/longmuralibar", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
