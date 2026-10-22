from typing import Any, ClassVar

import astropy.units as apyu
import jax
import numpy as np
import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..optional_deps import OptDeps
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin, ParameterShapeAMixin, ParameterShapeCMixin

A, C = 3.0, 1.2  # kpc
DELTA = 2.749545416973504  # sqrt(A^2 - C^2), as a float64

# Phi and grad Phi for G M = 1, a = 3, c = 1.2, from 50-digit mpmath: lambda
# and nu are the `mpmath.polyroots` roots in tau of
# tau^2 - (a^2 + c^2 + R^2 + z^2) tau + (a^2 c^2 + c^2 R^2 + a^2 z^2), which is
# R^2 / (tau - a^2) + z^2 / (tau - c^2) = 1 cleared of fractions; then
# Phi = -1 / (sqrt(lambda) + sqrt(nu)), and the gradient is `mpmath.diff` of
# that. Inputs are the float64 values below, exactly.
MPMATH_REFERENCE = [
    # generic
    (
        (1.0, 2.0, 3.0),
        -0.14981266593199087,
        (0.0038437680681181299, 0.0076875361362362599, 0.019113312182711657),
    ),
    # plane
    ((4.0, 0.0, 0.0), -0.16129032258064516, (0.02081165452653486, 0.0, 0.0)),
    # axis
    ((0.0, 0.0, 2.0), -0.1875334949612022, (0.0, 0.0, 0.030157007266217812)),
    # origin
    ((0.0, 0.0, 0.0), -0.2380952380952381, (0.0, 0.0, 0.0)),
    # focal points, where lambda = nu = a^2
    ((0.0, 0.0, DELTA), -0.16666666666666667, (0.0, 0.0, 0.025458753860865778)),
    ((0.0, 0.0, -DELTA), -0.16666666666666667, (0.0, 0.0, -0.025458753860865778)),
    (
        (1e-8, 0.0, DELTA),
        -0.16666666666666667,
        (5.3703703703703703e-11, 0.0, 0.025458753860865778),
    ),
    # just off the plane
    (
        (0.5, -0.3, 1e-9),
        -0.23495460227686393,
        (0.0090315955096128885, -0.0054189573057677329, 4.4800579148397022e-11),
    ),
    # far
    (
        (100.0, 50.0, -80.0),
        -0.007168431851065369,
        (3.7028903600844508e-5, 1.8514451800422254e-5, -3.0433311512746102e-5),
    ),
]
SPECIAL_POINTS = [xyz for xyz, *_ in MPMATH_REFERENCE]

# rho / M for very flat models (a = 3), from 60-digit mpmath: the Laplacian
# (`mpmath.diff`, order 2) of the root-form Phi above, over 4 pi. Independent of
# the closed-form density, and where a float64 Laplacian of Phi cancels to junk.
MPMATH_FLAT_DENSITY = [
    (3e-6, (1.0, 2.0, 3.0), 1.9871188956101738e-15),
    (3e-6, (0.0, 0.0, 2.0), 8.3556345122941567e-15),
    (3e-6, (100.0, 50.0, -80.0), 4.4953736661847207e-21),
    (3e-8, (1.0, 2.0, 3.0), 1.9871188956149996e-19),
    (3e-8, (0.0, 0.0, 2.0), 8.3556345123245006e-19),
    (3e-8, (100.0, 50.0, -80.0), 4.4953736661957809e-25),
]


def _sanders_delta2(pot: gp.AbstractPotential, R: float, z: float) -> jax.Array:
    """Delta^2 from Sanders (2012), eq. 9, at (x, y, z) = (R, 0, z), R > 0.

    With the sign of the first-derivative terms corrected, as in galpy's
    ``estimateDeltaStaeckel`` (flipping it fails this test by O(1)).

    Evaluated directly from the gradient and Hessian, independently of
    ``galax.dynamics``.
    """
    xyz = jnp.asarray([R, 0.0, z])
    g = pot.gradient(xyz, t=0)
    H = pot.hessian(xyz, t=0)
    num = 3 * z * g[0] - 3 * R * g[2] + R * z * (H[0, 0] - H[2, 2])
    return z**2 - R**2 + num / H[0, 2]


class TestKuzminKutuzovPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterShapeAMixin,
    ParameterShapeCMixin,
):
    """Test the `galax.potential.KuzminKutuzovPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.KuzminKutuzovPotential]:
        return gp.KuzminKutuzovPotential

    @pytest.fixture(scope="class")
    def field_a(self) -> u.Quantity["length"]:
        return u.Q(A, "kpc")

    @pytest.fixture(scope="class")
    def field_c(self) -> u.Quantity["length"]:
        return u.Q(C, "kpc")

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_a: u.Quantity,
        field_c: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m_tot": field_m_tot, "a": field_a, "c": field_c, "units": field_units}

    @pytest.fixture(scope="class")
    def GM(self, pot: gp.KuzminKutuzovPotential) -> float:
        t0 = u.Q(0, "Myr")
        return float(pot.constants["G"].value * pot.m_tot(t0, ustrip=pot.units["mass"]))

    # ==========================================================================

    def test_potential(self, pot: gp.KuzminKutuzovPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.6739326, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kuzminkutuzov", atol=1e-8
    )
    def test_gradient(self, pot: gp.KuzminKutuzovPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.KuzminKutuzovPotential, x: gt.QuSz3) -> None:
        expect = u.Q(2.2444315e08, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, rtol=1e-7, atol=u.Q(0, expect.unit)
        )

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kuzminkutuzov", atol=1e-8
    )
    def test_hessian(self, pot: gp.KuzminKutuzovPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/kuzminkutuzov", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Accuracy

    @pytest.mark.parametrize(("xyz", "phi", "grad"), MPMATH_REFERENCE)
    def test_mpmath(
        self,
        pot: gp.KuzminKutuzovPotential,
        GM: float,
        xyz: tuple[float, float, float],
        phi: float,
        grad: tuple[float, float, float],
    ) -> None:
        """Phi and its gradient match 50-digit mpmath, including at the focal points.

        The tolerances are a few ulp of Phi, and of the largest component of
        the gradient (the zero components then come out at the round-off of
        the others).
        """
        q = jnp.asarray(xyz)
        got_phi = pot.potential(q, t=0) / GM
        got_grad = pot.gradient(q, t=0) / GM
        np.testing.assert_allclose(got_phi, phi, rtol=1e-15, atol=0)
        np.testing.assert_allclose(
            got_grad, grad, rtol=4e-15, atol=4e-16 * np.max(np.abs(grad))
        )

    @pytest.mark.parametrize(("c", "xyz", "rho"), MPMATH_FLAT_DENSITY)
    def test_density_flat_mpmath(
        self,
        pot: gp.KuzminKutuzovPotential,
        c: float,
        xyz: tuple[float, float, float],
        rho: float,
    ) -> None:
        """The closed-form density stays accurate and positive for c / a ~ 1e-8."""
        kk = gp.KuzminKutuzovPotential(
            m_tot=u.Q(1.0, "Msun"), a=u.Q(A, "kpc"), c=u.Q(c, "kpc"), units=pot.units
        )
        np.testing.assert_allclose(kk.density(jnp.asarray(xyz), t=0), rho, rtol=1e-14)

    @pytest.mark.parametrize("xyz", SPECIAL_POINTS)
    def test_finite(
        self, pot: gp.KuzminKutuzovPotential, xyz: tuple[float, float, float]
    ) -> None:
        """Values and all derivatives are finite at the origin, plane, axis and foci."""
        q = jnp.asarray(xyz)
        for f in (pot.potential, pot.gradient, pot.hessian, pot.density):
            assert jnp.all(jnp.isfinite(f(q, t=0))), f.__name__
        assert pot.density(q, t=0) > 0

    @pytest.mark.parametrize("xyz", SPECIAL_POINTS)
    def test_jacfwd_jacrev(
        self, pot: gp.KuzminKutuzovPotential, xyz: tuple[float, float, float]
    ) -> None:
        """Forward and reverse mode agree with each other and with central FD.

        The FD tolerance is its round-off, ~ |Phi| eps / h ~ 1e-11 here
        (|Phi| <~ 1 kpc^2/Myr^2); its truncation error, ~ h^2 Phi''', is
        smaller.
        """

        def phi(q: jax.Array) -> jax.Array:
            return pot.potential(q, t=0)

        h = 1e-5
        q = jnp.asarray(xyz)
        fwd, rev = jax.jacfwd(phi)(q), jax.jacrev(phi)(q)
        fd = jnp.asarray(
            [(phi(q + h * e) - phi(q - h * e)) / (2 * h) for e in jnp.eye(3)]
        )
        np.testing.assert_allclose(fwd, rev, rtol=1e-15, atol=1e-17)
        np.testing.assert_allclose(fwd, fd, rtol=1e-8, atol=1e-10)

    # ---------------------------------
    # Limits and identities

    @pytest.mark.parametrize("xyz", SPECIAL_POINTS)
    def test_c_equals_a_is_isochrone(
        self, pot: gp.KuzminKutuzovPotential, xyz: tuple[float, float, float]
    ) -> None:
        """At c = a (Delta = 0) the potential is the isochrone with r_s = a."""
        kk = gp.KuzminKutuzovPotential(
            m_tot=pot.m_tot, a=pot.a, c=pot.a, units=pot.units
        )
        iso = gp.IsochronePotential(m_tot=pot.m_tot, r_s=pot.a, units=pot.units)
        q = jnp.asarray(xyz)
        np.testing.assert_allclose(
            kk.potential(q, t=0), iso.potential(q, t=0), rtol=1e-15
        )
        g_kk, g_iso = kk.gradient(q, t=0), iso.gradient(q, t=0)
        np.testing.assert_allclose(
            g_kk, g_iso, rtol=1e-14, atol=1e-15 * np.max(np.abs(g_iso))
        )
        assert jnp.all(jnp.isfinite(kk.hessian(q, t=0)))
        np.testing.assert_allclose(kk.density(q, t=0), iso.density(q, t=0), rtol=1e-12)

    @pytest.mark.parametrize(
        "xyz", [(1.0, 2.0, 3.0), (4.0, 0.0, 0.5), (0.5, -0.3, -2.0)]
    )
    def test_c_to_0_is_kuzmin(
        self, pot: gp.KuzminKutuzovPotential, xyz: tuple[float, float, float]
    ) -> None:
        """As c -> 0 the potential tends to the razor-thin Kuzmin disk, r_s = a.

        Off the plane the difference is O((c / a)^2), ~1e-18 here: rounding.
        """
        kk = gp.KuzminKutuzovPotential(
            m_tot=pot.m_tot, a=pot.a, c=u.Q(1e-9 * A, "kpc"), units=pot.units
        )
        kz = gp.KuzminPotential(m_tot=pot.m_tot, r_s=pot.a, units=pot.units)
        q = jnp.asarray(xyz)
        np.testing.assert_allclose(
            kk.potential(q, t=0), kz.potential(q, t=0), rtol=1e-14
        )
        np.testing.assert_allclose(kk.gradient(q, t=0), kz.gradient(q, t=0), rtol=1e-13)

    def test_symmetry(self, pot: gp.KuzminKutuzovPotential) -> None:
        """Axisymmetric and symmetric about the plane."""
        assert pot.symmetry is gp.Symmetry.ZROTATION_ZREFLECTION
        q = jnp.asarray([1.0, 2.0, 3.0])
        rot = jnp.asarray([-2.0, 1.0, 3.0])  # q rotated by 90 deg about z
        flip = jnp.asarray([1.0, 2.0, -3.0])
        assert pot.potential(rot, t=0) == pot.potential(q, t=0)
        assert pot.potential(flip, t=0) == pot.potential(q, t=0)

    @pytest.mark.parametrize(("R", "z"), [(1.0, 2.0), (4.0, 0.5), (0.3, -5.0)])
    def test_sanders_delta(
        self, pot: gp.KuzminKutuzovPotential, R: float, z: float
    ) -> None:
        """Exact Staeckel: Sanders (2012) eq. 9 returns a^2 - c^2 everywhere."""
        np.testing.assert_allclose(_sanders_delta2(pot, R, z), A**2 - C**2, rtol=1e-11)

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(not OptDeps.GALPY.installed, reason="requires galpy")
    def test_method_galpy(
        self, pot: gp.KuzminKutuzovPotential, batchx: gt.BBtQuSz3
    ) -> None:
        """Potential and forces match galpy's ``KuzminKutuzovStaeckelPotential``.

        galpy parameterises by the axis ratio ``ac = a / c`` and the focal
        distance ``Delta``, and its ``amp`` is the total mass.
        """
        # Deferred: galpy is optional, and this module is collected by the CI job
        # that installs no extras.
        from galpy.potential import (
            KuzminKutuzovStaeckelPotential,
            evaluateDensities,
            evaluatePotentials,
            evaluateR2derivs,
            evaluateRforces,
            evaluateRzderivs,
            evaluatez2derivs,
            evaluatezforces,
        )

        galpy_pot = KuzminKutuzovStaeckelPotential(
            amp=convert(pot.m_tot(t=u.Q(0, "Myr")), apyu.Quantity),
            ac=A / C,
            Delta=DELTA * apyu.kpc,
        )
        xyz = convert(batchx, apyu.Quantity)
        R, z = np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]

        # `quantity=False`: whether galpy returns floats or astropy Quantities
        # follows `astropy-units` in the reader's `~/.galpyrc`. Values are
        # physical either way.
        galax_pot = convert(pot.potential(batchx, t=0), apyu.Quantity)
        np.testing.assert_allclose(
            galax_pot.to_value("km2 / s2"),
            evaluatePotentials(galpy_pot, R, z, quantity=False),
            rtol=1e-12,
        )

        acc = convert(pot.acceleration(batchx, t=0), apyu.Quantity)
        acc = acc.to_value("km / (s Myr)")  # galpy's force unit
        a_R = (acc[:, 0] * xyz[:, 0].value + acc[:, 1] * xyz[:, 1].value) / R.value
        np.testing.assert_allclose(
            a_R, evaluateRforces(galpy_pot, R, z, quantity=False), rtol=1e-10
        )
        np.testing.assert_allclose(
            acc[:, 2], evaluatezforces(galpy_pot, R, z, quantity=False), rtol=1e-10
        )

        # Second derivatives and density: galpy's are analytic in (lambda, nu),
        # ours are autodiff of the closed form, so this is an independent check.
        H = convert(pot.hessian(batchx, t=0), apyu.Quantity).to_value(
            "1 / Gyr2"
        )  # galpy unit
        e_R = np.stack(
            [xyz[:, 0].value / R.value, xyz[:, 1].value / R.value, np.zeros(len(R))],
            axis=-1,
        )
        H_eR = np.einsum("nij,nj->ni", H, e_R)
        for got, f in [
            (np.einsum("ni,ni->n", e_R, H_eR), evaluateR2derivs),
            (H_eR[:, 2], evaluateRzderivs),
            (H[:, 2, 2], evaluatez2derivs),
        ]:
            np.testing.assert_allclose(
                got, f(galpy_pot, R, z, quantity=False), rtol=1e-10
            )

        dens = convert(pot.density(batchx, t=0), apyu.Quantity)
        np.testing.assert_allclose(
            dens.to_value("Msun / pc3"),
            evaluateDensities(galpy_pot, R, z, quantity=False),
            rtol=1e-10,
        )
