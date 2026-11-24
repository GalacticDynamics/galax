r"""Tests for the three quadrature potentials.

The potential of `AxisymmetricGaussianPotential`, `TriaxialGaussianPotential`
and `TriaxialNFWPotential` is Chandrasekhar's integral, which the kernels
evaluate by Gauss-Legendre quadrature. The references here are `mpmath.quad`
of the same integral in its original form, the one over :math:`\tau \in
[0, \infty)`, at 30 digits.
"""

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


def mp_potential(name, p, xyz, dps=30):
    """Chandrasekhar's integral over tau in [0, inf) at ``dps`` digits."""
    with mp.workdps(dps):
        G_, rs = mp.mpf(float(p["G"])), mp.mpf(float(p["r_s"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        q2 = mp.mpf(float(p["q2"]))
        x, y, z = (mp.mpf(float(c)) for c in xyz)
        if name == "triaxialnfw":
            rho0 = mp.mpf(float(p["m"])) / (4 * mp.pi * rs**3)

            def dpsi(xi2):
                return 2 / (1 + mp.sqrt(xi2))

        else:
            m = mp.mpf(float(p["m_tot"]))
            rho0 = m / (q1 * q2 * (2 * mp.pi) ** mp.mpf("1.5") * rs**3)

            def dpsi(xi2):
                return 2 * mp.exp(-xi2 / 2)

        def f(tau):
            xi2 = x**2 / (1 + tau) + y**2 / (q1**2 + tau) + z**2 / (q2**2 + tau)
            xi2 /= rs**2
            return dpsi(xi2) / mp.sqrt((1 + tau) * (q1**2 + tau) * (q2**2 + tau))

        r = max(mp.sqrt(x**2 + y**2 + z**2) / rs, mp.mpf("1e-3"))
        edges = [0, *(r**2 * mp.mpf(10) ** k for k in range(-6, 7, 2)), mp.inf]
        edges = sorted(set(edges))
        integral = mp.quad(f, edges)
        return -mp.pi * G_ * q1 * q2 * rho0 * rs**2 * integral


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("r_in_rs", RADII_IN_RS)
def test_potential_matches_mpmath(name, r_in_rs):
    """The order-50 kernel equals `mp.quad` of the integral, inside, near and far."""
    pot = CASES[name]
    module = getattr(gp, name)
    p = pot.params(0.0)
    xyz = jnp.asarray(DIRECTION) * r_in_rs * 5.0  # kpc
    got = float(module.potential(p, xyz))
    want = float(mp_potential(name, p, xyz))
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

    pot_low = type(pot)(
        **{
            f: getattr(pot, f)
            for f in ("m_tot", "m", "r_s", "q1", "q2")
            if hasattr(pot, f)
        },
        units=G,
        integration_order=order,
    )
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
    """For ``q1 = q2 = 1`` the three are the spherical Gaussian and NFW."""
    xyz = jnp.array([1.0, 2.0, 3.0])
    for name, sph in (
        ("axisymmetricgaussian", gp.GaussianPotential(m_tot=M, r_s=RS, units=G)),
        ("triaxialgaussian", gp.GaussianPotential(m_tot=M, r_s=RS, units=G)),
        ("triaxialnfw", gp.NFWPotential(m=M, r_s=RS, units=G)),
    ):
        pot = type(CASES[name])(
            **{f: getattr(sph, f) for f in ("m_tot", "m", "r_s") if hasattr(sph, f)},
            units=G,
        )
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
# The error of the kernels at each point is bounded by twice the error that the
# classes of the commit before the kernels (f5b3911e) had at the same point
# (rounded up), in float64 and in float32. The kernels reproduce those classes
# bit for bit, so the bounds say that the round-off did not get worse. Some of
# the bounds are large (1e-7 for a flattened centre): that is the order-50
# quadrature of an integrand that narrows, as in the docstrings, and the old
# code had it too.

FLAT = {  # strongly flattened axis ratios (q1, q2)
    "axisymmetricgaussian": (1.0, 0.1),
    "triaxialgaussian": (0.2, 0.1),
    "triaxialnfw": (0.2, 0.1),
}
GAUSSIANS = ("axisymmetricgaussian", "triaxialgaussian")
# (r in r_s, direction): the centre, near it, on the z axis, and far out.
POINTS = ((0.0, "x"), (1e-3, "gen"), (1.0, "z"), (30.0, "gen"), (100.0, "gen"))
# The points where the gradient and Hessian are tested (the flattened ones: z).
GRAD_POINTS = ((1e-3, "gen"), (1.0, "z"), (30.0, "gen"))
UNIT = {"x": (1, 0, 0), "z": (0, 0, 1), "gen": DIRECTION}

# Error bounds: (quantity, "sph" or "flat", dtype) -> bound, for the Gaussians
# and for the NFW. The values are 2 x the measured error of f5b3911e, rounded up.
BOUND = {
    "gauss": {
        ("potential", "sph", "f64"): 3e-8, ("potential", "flat", "f64"): 2e-7,
        ("density", "sph", "f64"): 1e-15, ("density", "flat", "f64"): 4e-15,
        ("gradient", "sph", "f64"): 5e-15, ("gradient", "flat", "f64"): 4e-9,
        ("hessian", "sph", "f64"): 3e-14, ("hessian", "flat", "f64"): 6e-8,
        ("potential", "sph", "f32"): 5e-7, ("potential", "flat", "f32"): 4e-7,
        ("density", "sph", "f32"): 5e-8, ("density", "flat", "f32"): 8e-6,
        ("gradient", "sph", "f32"): 4e-7, ("gradient", "flat", "f32"): 4e-7,
        ("hessian", "sph", "f32"): 3e-7, ("hessian", "flat", "f32"): 1.1e-6,
    },
    "nfw": {
        ("potential", "sph", "f64"): 5e-9, ("potential", "flat", "f64"): 2e-7,
        ("density", "sph", "f64"): 2e-15, ("density", "flat", "f64"): 2e-15,
        ("gradient", "sph", "f64"): 1e-14, ("gradient", "flat", "f64"): 2e-9,
        ("hessian", "sph", "f64"): 2e-13, ("hessian", "flat", "f64"): 5e-9,
        ("potential", "sph", "f32"): 3e-7, ("potential", "flat", "f32"): 4e-7,
        ("density", "sph", "f32"): 8e-7, ("density", "flat", "f32"): 8e-7,
        ("gradient", "sph", "f32"): 3e-7, ("gradient", "flat", "f32"): 7e-8,
        ("hessian", "sph", "f32"): 5e-7, ("hessian", "flat", "f32"): 6e-7,
    },
}  # fmt: skip


def make(name, q1, q2):
    if name == "axisymmetricgaussian":
        return gp.AxisymmetricGaussianPotential(m_tot=M, r_s=RS, q2=q2, units=G)
    if name == "triaxialgaussian":
        return gp.TriaxialGaussianPotential(m_tot=M, r_s=RS, q1=q1, q2=q2, units=G)
    return gp.TriaxialNFWPotential(m=M, r_s=RS, q1=q1, q2=q2, units=G)


def mp_density(name, p, xyz):
    with mp.workdps(30):
        rs, q2 = mp.mpf(float(p["r_s"])), mp.mpf(float(p["q2"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        x, y, z = (mp.mpf(float(c)) for c in xyz)
        xi2 = (x**2 + y**2 / q1**2 + z**2 / q2**2) / rs**2
        if name == "triaxialnfw":
            xi = mp.sqrt(xi2)
            return mp.mpf(float(p["m"])) / (4 * mp.pi * rs**3) / (xi * (1 + xi) ** 2)
        rho0 = mp.mpf(float(p["m_tot"])) / (q1 * q2 * (2 * mp.pi) ** 1.5 * rs**3)
        return rho0 * mp.exp(-xi2 / 2)


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


def mp_derivatives(name, p, xyz):
    """Gradient and Hessian by differentiating the integral over tau under the sign."""
    with mp.workdps(30):
        G_, rs = mp.mpf(float(p["G"])), mp.mpf(float(p["r_s"]))
        q1 = mp.mpf(float(p["q1"])) if "q1" in p else mp.mpf(1)
        q2 = mp.mpf(float(p["q2"]))
        c = [mp.mpf(1), q1**2, q2**2]
        v = [mp.mpf(float(c_)) for c_ in xyz]
        if name == "triaxialnfw":
            rho0 = mp.mpf(float(p["m"])) / (4 * mp.pi * rs**3)
        else:
            rho0 = mp.mpf(float(p["m_tot"])) / (q1 * q2 * (2 * mp.pi) ** 1.5 * rs**3)
        pref = -mp.pi * G_ * q1 * q2 * rho0 * rs**2

        def dF(X):  # dF/dX and d2F/dX2 of F = Delta psi / (rho0 r_s^2)
            if name == "triaxialnfw":
                s = mp.sqrt(X)
                return -1 / (s * (1 + s) ** 2), (1 + 3 * s) / (2 * s**3 * (1 + s) ** 3)
            e = mp.exp(-X / 2)
            return -e, e / 2

        def a(i, t):
            return 1 / (rs**2 * (c[i] + t))

        def D(t):
            return mp.sqrt((1 + t) * (c[1] + t) * (c[2] + t))

        def X(t):
            return sum(a(i, t) * v[i] ** 2 for i in range(3))

        r = max(mp.sqrt(sum(w * w for w in v)) / rs, mp.mpf("1e-3"))
        edges = sorted({0, *(r**2 * mp.mpf(10) ** k for k in range(-6, 7, 2)), mp.inf})
        grad = [
            pref
            * mp.quad(lambda t, i=i: dF(X(t))[0] * 2 * a(i, t) * v[i] / D(t), edges)
            for i in range(3)
        ]
        hess = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(i, 3):

                def fun(t, i=i, j=j):
                    f1, f2 = dF(X(t))
                    diag = f1 * 2 * a(i, t) if i == j else 0
                    return (f2 * 4 * a(i, t) * a(j, t) * v[i] * v[j] + diag) / D(t)

                hess[i][j] = hess[j][i] = pref * mp.quad(fun, edges)
        return [float(g) for g in grad], [[float(h) for h in row] for row in hess]


def as_dtype(tree, dtype):
    return jax.tree.map(lambda a: np.asarray(a, dtype), tree)


def worst(got, want):
    """Largest error relative to the largest entry of ``want``."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.abs(got - want).max() / np.abs(want).max())


@pytest.mark.parametrize("dtype", ["f64", "f32"])
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("group", ["sph", "flat"])
def test_errors_do_not_exceed_the_old_code(name, group, dtype):
    """Against analytic values the error is within 2x that of the pre-kernel classes."""
    q1, q2 = (1.0, 1.0) if group == "sph" else FLAT[name]
    pot = make(name, q1, q2)
    kind = "gauss" if name in GAUSSIANS else "nfw"
    module = getattr(gp, name)
    p64 = pot.params(0.0)
    p = as_dtype(p64, np.float64 if dtype == "f64" else np.float32)
    if name == "axisymmetricgaussian":  # the reference reads q1; it is 1 here
        p64 = {k: v for k, v in p64.items() if k != "q1"}
    rs = float(p64["r_s"])
    ctx = jax.enable_x64(dtype == "f64")

    def bound(quantity):
        return BOUND[kind][(quantity, group, dtype)]

    with ctx:
        for r, d in POINTS:
            direction = np.asarray(UNIT[d], float)
            xyz64 = direction / np.linalg.norm(direction) * r * rs
            xyz = np.asarray(xyz64, np.float64 if dtype == "f64" else np.float32)
            want_phi = (
                mp_spherical(name, p64, xyz64)[0]
                if group == "sph"
                else float(mp_potential(name, p64, xyz64))
            )
            out = module.potential(p, xyz)
            assert out.dtype == (np.float64 if dtype == "f64" else np.float32)
            got = float(out)
            err = abs(got - want_phi) / abs(want_phi)
            assert err <= bound("potential"), (r, d, "potential", err)
            if r > 0:
                want_rho = float(mp_density(name, p64, xyz64))
                if want_rho > 1e-30:
                    got = float(module.density(p, xyz))
                    err = abs(got - want_rho) / want_rho
                    assert err <= bound("density"), (r, d, "density", err)
        grad_points = GRAD_POINTS if group == "sph" else ((1.0, "z"),)
        for r, d in grad_points:
            direction = np.asarray(UNIT[d], float)
            xyz64 = direction / np.linalg.norm(direction) * r * rs
            xyz = np.asarray(xyz64, np.float64 if dtype == "f64" else np.float32)
            if group == "sph":
                _, g, h = mp_spherical(name, p64, xyz64)
            else:
                g, h = mp_derivatives(name, p64, xyz64)
            err = worst(module.gradient(p, xyz), g)
            assert err <= bound("gradient"), (r, d, "gradient", err)
            err = worst(module.hessian(p, xyz), h)
            assert err <= bound("hessian"), (r, d, "hessian", err)
