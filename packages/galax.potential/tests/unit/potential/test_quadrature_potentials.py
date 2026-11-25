r"""Tests for the three quadrature potentials.

The potential of `AxisymmetricGaussianPotential`, `TriaxialGaussianPotential`
and `TriaxialNFWPotential` is Chandrasekhar's integral, which the kernels
evaluate by Gauss-Legendre quadrature. The references here are `mpmath.quad`
of the same integral in its original form, the one over :math:`\tau \in
[0, \infty)`, at 30 digits.
"""

import dataclasses
import functools as ft

import jax
import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

G = "galactic"
M = u.Q(1e12, "Msun")
RS = u.Q(5, "kpc")
# q1 != q2, so that a swap of the two shows.
CASES = {
    "axisymmetricgaussian": gp.AxisymmetricGaussianPotential(
        m_tot=M, r_s=RS, q2=0.7, units=G
    ),
    "triaxialgaussian": gp.TriaxialGaussianPotential(
        m_tot=M, r_s=RS, q1=0.9, q2=0.6, units=G
    ),
    "triaxialnfw": gp.TriaxialNFWPotential(m=M, r_s=RS, q1=0.9, q2=0.6, units=G),
}
# A direction with all three components different, as a fraction of the radius.
DIRECTION = (2.0 / 7, 3.0 / 7, 6.0 / 7)
# Inside, near and far: 0.1, 1 and 30 r_s.
RADII_IN_RS = (0.1, 1.0, 30.0)


def mp_rho0(name, p):
    """Return the central density of the model, from its mass parameter."""
    with mp.workdps(30):
        rs = mp.mpf(float(p["r_s"]))
        if name == "triaxialnfw":
            return mp.mpf(float(p["m"])) / (4 * mp.pi * rs**3)
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        q2 = mp.mpf(float(p["q2"]))
        return mp.mpf(float(p["m_tot"])) / (q1 * q2 * (2 * mp.pi) ** 1.5 * rs**3)


def mp_integral(name, p, xyz, *, derivatives=False):  # noqa: C901
    """Chandrasekhar's integral over tau in [0, inf) at 30 digits.

    Returns the potential, and with derivatives also its gradient and Hessian,
    which are the integrals of the derivatives of the integrand.
    """
    with mp.workdps(30):
        G_, rs = mp.mpf(float(p["G"])), mp.mpf(float(p["r_s"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        q2 = mp.mpf(float(p["q2"]))
        c = [mp.mpf(1), q1**2, q2**2]
        v = [mp.mpf(float(x)) for x in xyz]
        nfw = name == "triaxialnfw"
        rho0 = mp_rho0(name, p)
        pref = -mp.pi * G_ * q1 * q2 * rho0 * rs**2

        def F(X):  # Delta psi / (rho0 r_s^2)
            return 2 / (1 + mp.sqrt(X)) if nfw else 2 * mp.exp(-X / 2)

        def dF(X):  # its first two derivatives in X (singular at X = 0 for the NFW)
            if nfw:
                s = mp.sqrt(X)
                return -1 / (s * (1 + s) ** 2), (1 + 3 * s) / (2 * s**3 * (1 + s) ** 3)
            e = mp.exp(-X / 2)
            return -e, e / 2

        def a(i, t):
            return 1 / (rs**2 * (c[i] + t))

        def X(t):
            return sum(a(i, t) * v[i] ** 2 for i in range(3))

        def D(t):
            return mp.sqrt((1 + t) * (c[1] + t) * (c[2] + t))

        r = max(mp.sqrt(sum(w * w for w in v)) / rs, mp.mpf("1e-3"))
        edges = sorted({0, *(r**2 * mp.mpf(10) ** k for k in range(-6, 7, 2)), mp.inf})

        def integral(fun):
            return pref * mp.quad(fun, edges)

        phi = integral(lambda t: F(X(t)) / D(t))
        if not derivatives:
            return float(phi)
        grad = [
            integral(lambda t, i=i: dF(X(t))[0] * 2 * a(i, t) * v[i] / D(t))
            for i in range(3)
        ]
        hess = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(i, 3):

                def fun(t, i=i, j=j):
                    f1, f2 = dF(X(t))
                    diag = f1 * 2 * a(i, t) if i == j else 0
                    return (f2 * 4 * a(i, t) * a(j, t) * v[i] * v[j] + diag) / D(t)

                hess[i][j] = hess[j][i] = integral(fun)
        return (
            float(phi),
            [float(g) for g in grad],
            [[float(h) for h in row] for row in hess],
        )


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("r_in_rs", RADII_IN_RS)
def test_potential_matches_mpmath(name, r_in_rs):
    """The order-50 kernel equals `mp.quad` of the integral, inside, near and far."""
    pot = CASES[name]
    module = getattr(gp, name)
    p = pot.params(0.0)
    xyz = jnp.asarray(DIRECTION) * r_in_rs * 5.0  # kpc
    got = float(module.potential(p, xyz))
    want = mp_integral(name, p, xyz)
    # Measured: <= 3e-14 at these radii, see the class docstrings.
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("order", [10, 20])
@pytest.mark.parametrize("name", CASES)
def test_integration_order_is_used_and_static(name, order):
    """``integration_order`` reaches the quadrature, in the class and the module."""
    pot = CASES[name]
    module = getattr(gp, name)
    p = pot.params(0.0)
    xyz = jnp.asarray(DIRECTION) * 20.0 * 5.0  # 20 r_s

    low = module.potential(p, xyz, integration_order=order)
    high = module.potential(p, xyz)
    assert module.potential(p, xyz, integration_order=50) == high
    assert low != high

    pot_low = dataclasses.replace(pot, integration_order=order)
    np.testing.assert_allclose(pot_low.potential(u.Q(xyz, "kpc"), 0.0).value, low)
    # The class gradient and the module gradient agree at the same order.
    np.testing.assert_allclose(
        pot_low.gradient(xyz, 0.0),
        module.gradient(p, xyz, integration_order=order),
        rtol=1e-10,
    )
    # `hessian` and `gradient` take the keyword, and the order changes them.
    for fn in (module.gradient, module.hessian):
        assert not np.array_equal(fn(p, xyz, integration_order=order), fn(p, xyz))


def test_spherical_limits():
    """For q1 = q2 = 1 the three are the spherical Gaussian and NFW."""
    xyz = jnp.array([1.0, 2.0, 3.0])
    gaussian = gp.GaussianPotential(m_tot=M, r_s=RS, units=G)
    nfw = gp.NFWPotential(m=M, r_s=RS, units=G)
    for name, sph in (
        ("axisymmetricgaussian", gaussian),
        ("triaxialgaussian", gaussian),
        ("triaxialnfw", nfw),
    ):
        pot = with_q(name, 1.0, 1.0)
        np.testing.assert_allclose(
            pot.potential(xyz, 0.0), sph.potential(xyz, 0.0), rtol=1e-12
        )
        np.testing.assert_allclose(
            pot.density(xyz, 0.0), sph.density(xyz, 0.0), rtol=1e-12
        )


@pytest.mark.parametrize("q2", [0.7, 1.5])
def test_axisymmetric_central_potential(q2):
    """``Phi(0) = -sqrt(2/pi) G m / r_s * f(q2)``: arcsin for q2 < 1, arcsinh above."""
    pot = gp.AxisymmetricGaussianPotential(m_tot=M, r_s=RS, q2=q2, units=G)
    p = pot.params(0.0)
    if q2 < 1:
        e = np.sqrt(1 - q2**2)
        f = np.arcsin(e) / e
    else:
        e = np.sqrt(q2**2 - 1)
        f = np.arcsinh(e) / e
    want = -np.sqrt(2 / np.pi) * p["G"] * p["m_tot"] / p["r_s"] * f
    got = gp.axisymmetricgaussian.potential(p, jnp.zeros(3))
    np.testing.assert_allclose(got, want, rtol=1e-12)


@pytest.mark.parametrize("name", ["axisymmetricgaussian", "triaxialgaussian"])
def test_gaussian_density_integrates_to_the_mass(name):
    """The sum of the density over a fine box equals ``m_tot``, for any axis ratio.

    The trapezoid rule is spectrally accurate for a Gaussian that is negligible at
    the edge of the box (9 widths along the widest axis, 4 nodes per width).
    """
    p = CASES[name].params(0.0)
    h = float(p["r_s"]) / 4
    n = 9 * 4 * 2 + 1
    x = (np.arange(n) - n // 2) * h
    xyz = jnp.asarray(np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1))
    mass = float(jnp.sum(getattr(gp, name).density(p, xyz))) * h**3
    np.testing.assert_allclose(mass, float(p["m_tot"]), rtol=1e-8)


# ---------------------------------------------------------------------------
# Asymptotic limits, the centre and strong flattening, against analytic values.
#
# The error at each point is bounded by 1.5 times the error that the classes of
# the commit before the kernels (f5b3911e) had at the same point, plus one unit
# of round-off of the dtype, in float64 and in float32. The classes keep their
# results bit for bit, so the bounds say that the round-off did not get worse
# (a kernel under jax.vmap can differ from the class by a few ulp, from XLA).
# The unit of round-off is larger than some old float32 errors, so a bound can
# let through an error that is 3 to 5 times the old one: the float32 values stored
# at the end of this file catch that, to two ulp. The flattened bounds are large
# (1e-7 at the centre of a q = 0.1 model): that is the order-50 quadrature of an
# integrand that narrows, as in the docstrings, and the old code had it too.
# Each bound is the largest error of the Gaussians, or of the NFW, over the
# points below, with the centre apart.

FLAT = {  # strongly flattened axis ratios (q1, q2)
    "axisymmetricgaussian": (1.0, 0.1),
    "triaxialgaussian": (0.2, 0.1),
    "triaxialnfw": (0.2, 0.1),
}
GAUSSIANS = ("axisymmetricgaussian", "triaxialgaussian")
# (r in r_s, direction): near the centre, on the z axis, and far out.
POINTS = ((1e-3, "gen"), (1.0, "z"), (30.0, "gen"))
# The flattened model's gradient near the centre is limited by the quadrature
# (6e-6 at 1e-3 r_s), so it is not tested there.
GRAD_POINTS = {"sph": POINTS, "flat": POINTS[1:]}
UNIT = {"z": (0, 0, 1), "gen": DIRECTION}

# (quantity, "sph" or "flat", dtype) -> the bound at each of POINTS (the centre:
# one number), for the Gaussians and for the NFW.
BOUND = {
    "gauss": {
        ("centre", "sph", "f64"): 4.5e-16,
        ("potential", "sph", "f64"): (4.5e-16, 4.9e-16, 4.2e-14),
        ("density", "sph", "f64"): (4.0e-16, 5.1e-16, None),
        ("gradient", "sph", "f64"): (2.5e-15, 1.1e-15, 3.3e-15),
        ("hessian", "sph", "f64"): (2.4e-15, 2.6e-15, 1.6e-14),
        ("centre", "flat", "f64"): 1.3e-7,
        ("potential", "flat", "f64"): (1.3e-7, 1.1e-10, 4.2e-14),
        ("density", "flat", "f64"): (5.0e-16, 2.4e-15, None),
        ("gradient", "flat", "f64"): (None, 2.8e-9, 4.0e-15),
        ("hessian", "flat", "f64"): (None, 4.3e-8, 2.4e-14),
        ("centre", "sph", "f32"): 2.0e-7,
        ("potential", "sph", "f32"): (3.3e-7, 2.7e-7, 4.6e-7),
        ("density", "sph", "f32"): (1.2e-7, 1.5e-7, None),
        ("gradient", "sph", "f32"): (1.7e-7, 3.7e-7, 3.1e-7),
        ("hessian", "sph", "f32"): (2.5e-7, 3.3e-7, 2.7e-7),
        ("centre", "flat", "f32"): 2.8e-7,
        ("potential", "flat", "f32"): (3.8e-7, 2.4e-7, 2.8e-7),
        ("density", "flat", "f32"): (2.5e-7, 5.8e-6, None),
        ("gradient", "flat", "f32"): (None, 3.5e-7, 2.3e-7),
        ("hessian", "flat", "f32"): (None, 9.0e-7, 6.2e-7),
    },
    "nfw": {
        ("centre", "sph", "f64"): 2.2e-16,
        ("potential", "sph", "f64"): (4.1e-16, 4.9e-16, 1.4e-14),
        ("density", "sph", "f64"): (8.0e-16, 2.2e-16, 1.2e-15),
        ("gradient", "sph", "f64"): (4.9e-16, 1.7e-15, 6.2e-15),
        ("hessian", "sph", "f64"): (5.4e-16, 1.9e-15, 9.5e-14),
        ("centre", "flat", "f64"): 1.3e-7,
        ("potential", "flat", "f64"): (1.3e-7, 2.5e-10, 3.3e-12),
        ("density", "flat", "f64"): (5.4e-16, 1.2e-15, 2.2e-16),
        ("gradient", "flat", "f64"): (None, 1.1e-9, 1.2e-11),
        ("hessian", "flat", "f64"): (None, 2.9e-9, 3.9e-11),
        ("centre", "sph", "f32"): 1.7e-7,
        ("potential", "sph", "f32"): (2.9e-7, 2.3e-7, 2.9e-7),
        ("density", "sph", "f32"): (2.5e-7, 2.6e-7, 1.4e-7),
        ("gradient", "sph", "f32"): (3.4e-7, 3.0e-7, 2.9e-7),
        ("hessian", "sph", "f32"): (4.0e-7, 2.6e-7, 4.3e-7),
        ("centre", "flat", "f32"): 3.0e-7,
        ("potential", "flat", "f32"): (2.7e-7, 1.2e-7, 1.8e-7),
        ("density", "flat", "f32"): (2.9e-7, 3.5e-7, 5.7e-7),
        ("gradient", "flat", "f32"): (None, 1.7e-7, 2.2e-7),
        ("hessian", "flat", "f32"): (None, 5.3e-7, 2.7e-7),
    },
}  # fmt: skip


def with_q(name, q1, q2):
    """``CASES[name]`` with the axis ratios replaced (the axisymmetric has no q1)."""
    pot = CASES[name]
    new = {"q1": q1, "q2": q2} if hasattr(pot, "q1") else {"q2": q2}
    return dataclasses.replace(pot, **new)


def mp_density(name, p, xyz):
    with mp.workdps(30):
        rs, q2 = mp.mpf(float(p["r_s"])), mp.mpf(float(p["q2"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        x, y, z = (mp.mpf(float(c)) for c in xyz)
        xi2 = (x**2 + y**2 / q1**2 + z**2 / q2**2) / rs**2
        if name == "triaxialnfw":
            xi = mp.sqrt(xi2)
            return mp_rho0(name, p) / (xi * (1 + xi) ** 2)
        return mp_rho0(name, p) * mp.exp(-xi2 / 2)


def mp_spherical(name, p, xyz):
    """Closed-form potential, gradient and Hessian of the spherical (q = 1) model."""
    with mp.workdps(30):
        G_, rs = mp.mpf(float(p["G"])), mp.mpf(float(p["r_s"]))
        m = mp.mpf(float(p["m"] if name == "triaxialnfw" else p["m_tot"]))
        v = [mp.mpf(float(c)) for c in xyz]
        r = mp.sqrt(sum(c * c for c in v))
        x = r / rs
        if not r:  # the centre: only the potential
            if name == "triaxialnfw":
                return float(-G_ * m / rs), None, None
            return float(-G_ * m * mp.sqrt(2 / mp.pi) / rs), None, None
        if name == "triaxialnfw":
            phi = -G_ * m * (mp.log(1 + x) / r)
            mass = m * (mp.log(1 + x) - x / (1 + x))
            rho = m / (4 * mp.pi * rs**3) / (x * (1 + x) ** 2)
        else:
            e = mp.exp(-(x**2) / 2)
            phi = -G_ * m * (mp.erf(x / mp.sqrt(2)) / r)
            mass = m * (mp.erf(x / mp.sqrt(2)) - mp.sqrt(2 / mp.pi) * x * e)
            rho = m / ((2 * mp.pi) ** 1.5 * rs**3) * e
        d1 = G_ * mass / r**2
        d2 = 4 * mp.pi * G_ * rho - 2 * d1 / r
        grad = [d1 * c / r for c in v]
        hess = [
            [d1 / r * (i == j) + (d2 - d1 / r) * v[i] * v[j] / r**2 for j in range(3)]
            for i in range(3)
        ]
        return (
            float(phi),
            [float(g) for g in grad],
            [[float(h) for h in row] for row in hess],
        )


@ft.cache
def reference(name, group, r, d):
    """Analytic (xyz, potential, density, gradient, Hessian) at r r_s along d."""
    q1, q2 = (1.0, 1.0) if group == "sph" else FLAT[name]
    p = with_q(name, q1, q2).params(0.0)
    direction = np.asarray(UNIT[d], float)
    xyz = direction / np.linalg.norm(direction) * r * float(p["r_s"])
    rho = float(mp_density(name, p, xyz))
    if (r, d) in GRAD_POINTS[group]:
        if group == "sph":
            phi, grad, hess = mp_spherical(name, p, xyz)
        else:
            phi, grad, hess = mp_integral(name, p, xyz, derivatives=True)
    else:
        phi = (
            mp_spherical(name, p, xyz)[0]
            if group == "sph"
            else mp_integral(name, p, xyz)
        )
        grad = hess = None
    return xyz, phi, rho, grad, hess


def as_dtype(tree, dtype):
    return jax.tree.map(lambda a: np.asarray(a, dtype), tree)


def worst(got, want):
    """Largest error relative to the largest entry of want."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.abs(got - want).max() / np.abs(want).max())


@pytest.mark.parametrize("dtype", ["f64", "f32"])
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("group", ["sph", "flat"])
def test_errors_do_not_exceed_the_old_code(name, group, dtype):
    """Against analytic values the error is within 1.5x that of the old classes."""
    q1, q2 = (1.0, 1.0) if group == "sph" else FLAT[name]
    kind = "gauss" if name in GAUSSIANS else "nfw"
    module = getattr(gp, name)
    np_dtype = np.float64 if dtype == "f64" else np.float32
    p = as_dtype(with_q(name, q1, q2).params(0.0), np_dtype)

    def bound(quantity, i=None):
        b = BOUND[kind][(quantity, group, dtype)]
        return b if i is None else b[i]

    # The references and the parameters are built in the default precision.
    p64 = with_q(name, q1, q2).params(0.0)
    centre = (
        mp_spherical(name, p64, np.zeros(3))[0]
        if group == "sph"
        else mp_integral(name, p64, np.zeros(3))
    )
    refs = {pt: reference(name, group, *pt) for pt in POINTS}

    with jax.enable_x64(dtype == "f64"):
        got = float(module.potential(p, np.zeros(3, np_dtype)))
        assert abs(got - centre) / abs(centre) <= bound("centre"), "centre"

        for i, (r, d) in enumerate(POINTS):
            xyz64, phi, rho, grad, hess = refs[r, d]
            xyz = np.asarray(xyz64, np_dtype)
            out = module.potential(p, xyz)
            assert out.dtype == np_dtype
            err = abs(float(out) - phi) / abs(phi)
            assert err <= bound("potential", i), (r, d, "potential", err)
            if rho > 1e-30:
                err = abs(float(module.density(p, xyz)) - rho) / rho
                assert err <= bound("density", i), (r, d, "density", err)
            if grad is not None:
                err = worst(module.gradient(p, xyz), grad)
                assert err <= bound("gradient", i), (r, d, "gradient", err)
                err = worst(module.hessian(p, xyz), hess)
                assert err <= bound("hessian", i), (r, d, "hessian", err)


@pytest.mark.parametrize("name", CASES)
def test_rho0(name):
    """rho0_of_m and the class rho0 are the central density."""
    pot = CASES[name]
    p = pot.params(0.0)
    r_s3 = float(p["r_s"]) ** 3
    if name == "triaxialnfw":
        want = float(p["m"]) / (4 * np.pi * r_s3)
    else:
        q1 = float(p["q1"]) if "q1" in p else 1.0
        want = float(p["m_tot"]) / (q1 * float(p["q2"]) * (2 * np.pi) ** 1.5 * r_s3)
    np.testing.assert_allclose(getattr(gp, name).rho0_of_m(p), want, rtol=1e-14)
    np.testing.assert_allclose(pot.rho0(0.0, ustrip=True), want, rtol=1e-14)
    assert str(pot.rho0(0.0).unit) == "solMass / kpc3"


# float32 outputs of the classes before this PR (f5b3911e; the kernels reproduce
# them bit for bit) for the flattened models: [potential, gradient_z, hessian_xx,
# hessian_zz] at 1 r_s on the z axis and at 30 r_s in the generic direction.
# The bounds above can hide a 3 to 5 times larger error, because they allow one
# unit of float32 round-off; these values do not. A change of two ulp (2.4e-7) or
# more fails, which includes the triaxial NFW with its radius formed as
# sqrt(xi^2) (it differs by 4.1e-7 in hessian_zz at 30 r_s).
BASE_FLOAT32 = {
    ("axisymmetricgaussian", 1.0, "z"): [
        -0.5913220643997192,
        0.06229864060878754,
        0.005653107538819313,
        -0.011306222528219223,
    ],
    ("axisymmetricgaussian", 30.0, "gen"): [
        -0.02997014857828617,
        0.00017118015966843814,
        1.0042966778200935e-06,
        -1.6051783404691378e-06,
    ],
    ("triaxialgaussian", 1.0, "z"): [
        -0.704182505607605,
        0.09859040379524231,
        0.007974565960466862,
        -0.026454534381628036,
    ],
    ("triaxialgaussian", 30.0, "gen"): [
        -0.0299773458391428,
        0.00017120347183663398,
        1.0047156138170976e-06,
        -1.6028116078814492e-06,
    ],
    ("triaxialnfw", 1.0, "z"): [
        -0.0195752140134573,
        0.0017351536080241203,
        0.00013315113028511405,
        -0.000400826393160969,
    ],
    ("triaxialnfw", 30.0, "gen"): [
        -0.0023932859767228365,
        1.0682641914172564e-05,
        4.9046370520500204e-08,
        -8.770801684931939e-08,
    ],
}


@pytest.mark.parametrize(("name", "r", "d"), list(BASE_FLOAT32))
def test_float32_matches_the_old_classes(name, r, d):
    """In float32 the functions stay within two ulp of the old classes."""
    p = as_dtype(with_q(name, *FLAT[name]).params(0.0), np.float32)
    direction = np.asarray(UNIT[d], float)
    xyz = direction / np.linalg.norm(direction) * r * float(p["r_s"])
    module = getattr(gp, name)
    float32 = False  # x64 off, so that the arithmetic is float32
    with jax.enable_x64(float32):
        x = np.asarray(xyz, np.float32)
        grad, hess = np.asarray(module.gradient(p, x)), np.asarray(module.hessian(p, x))
        got = [float(module.potential(p, x)), grad[2], hess[0, 0], hess[2, 2]]
    np.testing.assert_allclose(got, BASE_FLOAT32[name, r, d], rtol=2.4e-7)
