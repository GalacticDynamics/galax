"""Tests for `galax.potential._src.kernels`."""

import inspect

import jax

import quaxed.numpy as jnp

from galax.potential._src import kernels as k
from galax.potential._src.utils import shell_d2phi_dr2, shell_dphi_dr

p = {"G": 1.0, "m": 2.0, "a": 3.0}


def phi(p, r):
    return -p["G"] * p["m"] / jnp.sqrt(r**2 + p["a"] ** 2)


def dphi(p, r):
    return p["G"] * p["m"] * r / (r**2 + p["a"] ** 2) ** 1.5


def d2phi(p, r):
    return p["G"] * p["m"] * (p["a"] ** 2 - 2 * r**2) / (r**2 + p["a"] ** 2) ** 2.5


def lap(p, r):
    return 3 * p["G"] * p["m"] * p["a"] ** 2 / (r**2 + p["a"] ** 2) ** 2.5


def phi3(p, xyz):
    return -p["G"] * p["m"] / jnp.sqrt(jnp.sum(xyz**2) + p["a"] ** 2)


def test_radial_gradient_matches_closed_form():
    assert jnp.allclose(k.radial_gradient(phi)(p, 1.5), dphi(p, 1.5))


def test_radial_second_derivative_matches_closed_form():
    f = k.radial_second_derivative(k.radial_gradient(phi))
    assert jnp.allclose(f(p, 1.5), d2phi(p, 1.5))


def test_mass_from_dpotential():
    r = jnp.array([0.5, 1.0, 4.0])
    want = p["m"] * r**3 / (r**2 + p["a"] ** 2) ** 1.5
    assert jnp.allclose(k.mass_from_dpotential(dphi)(p, r), want)


def test_spherical_gradient_direction():
    xyz = jnp.array([3.0, 4.0, 0.0])
    got = k.spherical_gradient(dphi)(p, xyz)
    assert jnp.allclose(got, dphi(p, 5.0) * xyz / 5.0)


def test_spherical_hessian_matches_autodiff():
    xyz = jnp.array(
        [[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [3.0, 0.1, 0.2]],
         [[-1.0, 2.0, 0.3], [4.0, 4.0, 4.0], [0.2, 0.3, -0.4]]]
    )  # fmt: skip
    got = k.spherical_hessian(dphi, d2phi, lap)(p, xyz)
    one = jax.hessian(lambda x: phi(p, jnp.linalg.norm(x)))
    want = jax.vmap(jax.vmap(one))(xyz)
    assert got.shape == (2, 3, 3, 3)
    assert jnp.allclose(got, want)


def test_spherical_origin_cored():
    z = jnp.zeros(3)
    g = k.spherical_gradient(dphi)(p, z)
    h = k.spherical_hessian(dphi, d2phi, lap)(p, z)
    assert jnp.array_equal(g, z)
    assert jnp.allclose(h, p["G"] * p["m"] / p["a"] ** 3 * jnp.eye(3))
    assert jnp.all(jnp.isfinite(h))
    # The origin value is differentiable with respect to `p` and `xyz`.
    hess = k.spherical_hessian(dphi, d2phi, lap)
    assert jnp.all(jnp.isfinite(jax.jacfwd(hess, argnums=1)(p, z)))
    assert all(jnp.all(jnp.isfinite(v)) for v in jax.jacfwd(hess)(p, z).values())


def test_spherical_origin_float32():
    """The float32 origin Hessian equals the float64 one to float32 precision."""
    z32 = jnp.zeros(3, dtype=jnp.float32)
    p32 = {key: jnp.float32(v) for key, v in p.items()}
    h32 = k.spherical_hessian(dphi, d2phi, lap)(p32, z32)
    h64 = k.spherical_hessian(dphi, d2phi, lap)(p, jnp.zeros(3))
    assert h32.dtype == jnp.float32
    assert jnp.allclose(h32, h64, rtol=1e-6)


def test_spherical_origin_cusp():
    """A cusp has an infinite diagonal and zero off-diagonal, never NaN."""
    p_ = {"G": 1.0, "m": 1.0}
    h = k.spherical_hessian(
        lambda p, r: p["G"] * p["m"] / r,  # Phi' ~ 1/r: a log cusp
        lambda p, r: -p["G"] * p["m"] / r**2,
        lambda p, r: p["G"] * p["m"] / r**2,
    )(p_, jnp.zeros(3))
    assert jnp.all(jnp.diag(h) == jnp.inf)
    assert jnp.all(h[~jnp.eye(3, dtype=bool)] == 0)


def test_spherical_float32_si():
    """float32 in SI units: no ``r**3`` forms under `jax.jit` (it overflows)."""
    # Hernquist at SI scale: M = 1e7 Msun in kg, r_s = 0.3 kpc in m.
    p64 = {"G": 6.674e-11, "m": 2e37, "r_s": 1e19}

    def mass(p, r):
        x = r / p["r_s"]
        return p["m"] * x**2 / (1 + x) ** 2

    def dmass(p, r):
        x = r / p["r_s"]
        return 2 * p["m"] / p["r_s"] * x / (1 + x) ** 3

    dphi_si, d2phi_si = shell_dphi_dr(mass, dmass), shell_d2phi_dr2(mass, dmass)

    def lap_si(p, r):
        return p["G"] * dmass(p, r) / r / r

    xyz64 = jnp.logspace(17, 19, 7)[:, None] * jnp.array([2.0, 3.0, 6.0]) / 7
    p32 = {key: jnp.float32(v) for key, v in p64.items()}
    xyz32 = xyz64.astype(jnp.float32)
    grad = k.spherical_gradient(dphi_si)
    hess = k.spherical_hessian(dphi_si, d2phi_si, lap_si)
    for f in (grad, hess):
        got, want = jax.jit(f)(p32, xyz32), f(p64, xyz64)
        assert got.dtype == jnp.float32
        assert jnp.all(got != 0)
        assert jnp.allclose(got, want, rtol=1e-4, atol=0)


def test_helper_signatures():
    """Helper-built functions show the parameters of their source function."""

    def sig(f):
        return str(inspect.signature(f))

    def rho(p, r):
        return p["m"] * jnp.exp(-r / p["a"])

    def phi4(p, xyz, t):
        return phi3(p, xyz) * jnp.cos(t)

    assert sig(k.poisson_laplacian(rho)) == "(p, r, /)"
    assert sig(k.poisson_density(rho)) == "(p, r, /)"
    assert sig(k.cartesian_gradient(phi3)) == "(p, xyz, /)"
    assert sig(k.cartesian_hessian(phi4)) == "(p, xyz, t, /)"
    assert sig(k.cartesian_laplacian(phi4)) == "(p, xyz, t, /)"


def test_poisson_roundtrip():
    def rho(p, r):
        return p["m"] * jnp.exp(-r / p["a"])

    f = k.poisson_density(k.poisson_laplacian(rho))
    assert jnp.allclose(f(p, 1.3), rho(p, 1.3))
    assert jnp.allclose(k.poisson_laplacian(rho)(p, 1.3), 4 * jnp.pi * rho(p, 1.3))


def test_cartesian_helpers_closed_form():
    xyz = jnp.array([1.0, 2.0, 3.0])
    s = jnp.sum(xyz**2) + p["a"] ** 2
    Gm = p["G"] * p["m"]
    assert jnp.allclose(k.cartesian_gradient(phi3)(p, xyz), Gm * xyz / s**1.5)
    want_h = Gm * (s * jnp.eye(3) - 3 * jnp.outer(xyz, xyz)) / s**2.5
    assert jnp.allclose(k.cartesian_hessian(phi3)(p, xyz), want_h)
    assert jnp.allclose(
        k.cartesian_laplacian(phi3)(p, xyz), 3 * Gm * p["a"] ** 2 / s**2.5
    )


def test_cartesian_helpers_forward_t():
    def phi4(p, xyz, t):
        return phi3(p, xyz) * jnp.cos(t)

    xyz = jnp.array([1.0, 2.0, 3.0])
    got = k.cartesian_gradient(phi4)(p, xyz, 0.7)
    assert jnp.allclose(got, jnp.cos(0.7) * k.cartesian_gradient(phi3)(p, xyz))
    assert k.cartesian_hessian(phi4)(p, xyz, 0.7).shape == (3, 3)
    assert k.cartesian_laplacian(phi4)(p, xyz, 0.7).shape == ()


def test_float32_preserved():
    p32 = {key: jnp.float32(v) for key, v in p.items()}
    r = jnp.float32(1.5)
    xyz = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    t = jnp.float32(0.7)
    d1 = k.radial_gradient(phi)
    d2 = k.radial_second_derivative(d1)

    def phi4(p, xyz, t):
        return phi3(p, xyz) * jnp.cos(t)

    outs = [
        d1(p32, r),
        d2(p32, r),
        k.mass_from_dpotential(d1)(p32, r),
        k.spherical_gradient(d1)(p32, xyz),
        k.spherical_hessian(d1, d2, lap)(p32, xyz),
        k.spherical_gradient(d1)(p32, jnp.zeros(3, dtype=jnp.float32)),
        k.spherical_hessian(d1, d2, lap)(p32, jnp.zeros(3, dtype=jnp.float32)),
        k.poisson_laplacian(phi)(p32, r),
        k.poisson_density(phi)(p32, r),
        k.cartesian_gradient(phi3)(p32, xyz),
        k.cartesian_hessian(phi3)(p32, xyz),
        k.cartesian_laplacian(phi3)(p32, xyz),
        k.cartesian_gradient(phi4)(p32, xyz, t),
    ]
    for o in outs:
        assert o.dtype == jnp.float32


# ---------------------------------------------------------------------------
# Keyword-only static parameters


def phi_order(p, xyz, *, order=3):
    """Toy ``phi`` with a static ``order``: ``-G m / (|x|^2 + a^2)^(order / 6)``."""
    return -p["G"] * p["m"] / (jnp.sum(xyz**2) + p["a"] ** 2) ** (order / 6)


def test_static_kwarg_signature_and_forwarding():
    xyz = jnp.array([1.0, 2.0, 3.0])
    for factory in (k.cartesian_gradient, k.cartesian_hessian, k.cartesian_laplacian):
        f = factory(phi_order)
        assert str(inspect.signature(f)) == "(p, xyz, /, *, order=3)"
        # A different `order` changes the value: the keyword reaches `phi`.
        assert not jnp.allclose(f(p, xyz), f(p, xyz, order=6))
        assert jnp.allclose(f(p, xyz), f(p, xyz, order=3))

    want = jax.grad(phi_order, 1)(p, xyz, order=7)
    assert jnp.allclose(k.cartesian_gradient(phi_order)(p, xyz, order=7), want)


def test_static_kwarg_is_static_under_jit():
    traces = []

    def phi_counted(p, xyz, *, order=3):
        traces.append(order)
        return phi_order(p, xyz, order=order)

    f = k.cartesian_gradient(phi_counted)
    xyz = jnp.array([1.0, 2.0, 3.0])
    f(p, xyz, order=7)
    f(p, xyz * 2, order=7)  # same order: cached
    assert traces == [7]
    f(p, xyz, order=5)  # a new order retraces, with a Python value
    assert traces == [7, 5]


def test_static_kwarg_poisson_density_equals_trace_hessian():
    xyz = jnp.array([1.0, 2.0, 3.0])
    lap_ = k.cartesian_laplacian(phi_order)
    rho = k.poisson_density(lap_)
    assert str(inspect.signature(rho)) == "(p, xyz, /, *, order=3)"
    want = jnp.trace(jax.hessian(phi_order, 1)(p, xyz, order=7)) / (4 * jnp.pi * p["G"])
    assert jnp.allclose(rho(p, xyz, order=7), want)
    assert jnp.allclose(
        k.poisson_laplacian(rho)(p, xyz, order=7), lap_(p, xyz, order=7)
    )
