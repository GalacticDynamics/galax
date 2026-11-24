"""Tests for `flattened_gnfw_halo` and the Koposov et al. (2023) halo."""

from typing import Any

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.builtin.flattened_gnfw import log_density

koposov2023 = gp.papers.doi_10_1093_mnras_stad551

M, RS = u.Q(5.4e11, "Msun"), u.Q(15.0, "kpc")

# The spherical, uncut limit. NFW's potential depends on all the mass outside
# a radius, so the grid reaches far enough out that the expansion's outer
# power-law continuation does not set the error.
SPHERICAL = {"r_cut": u.Q(1e8, "kpc"), "r_max": u.Q(1e5, "kpc"), "n_r": 192}

# Strongly flattened, with the flattening changing with radius.
FLAT = {"gamma": 1.27, "beta": 2.97, "q_in": 0.55, "q_out": 0.9}
FLAT_M, FLAT_RS = u.Q(10**12.05, "Msun"), u.Q(16.1, "kpc")

# Koposov et al.'s oblate fit (Table 2), with a tilted axis.
M_NFW = u.Q(11.22e11, "Msun")
KOPOSOV = {"q": 0.55, "gamma": 1.27, "beta": 2.97}
AXIS = jnp.asarray([0.25, 0.45, 0.85])

X = jnp.asarray(
    [
        [8.0, 1.0, 0.5],
        [20.0, -5.0, 10.0],
        [50.0, 30.0, -40.0],
        [0.3, 0.2, 0.1],
        [150.0, -80.0, 60.0],
    ]
)


def _flat(l_max: int, **kw: Any) -> gp.MultipoleProfilePotential:
    return gp.flattened_gnfw_halo(
        FLAT_M, FLAT_RS, l_max=l_max, units="galactic", **(FLAT | kw)
    )


def _koposov(l_max: int = 8, **kw: Any) -> gp.TransformedPotential:
    return koposov2023(
        M_NFW, FLAT_RS, l_max=l_max, units="galactic", **(KOPOSOV | {"axis": AXIS} | kw)
    )


def _true_density() -> jax.Array:
    """Return the closed-form density of `_flat` at `X`, in Msun / kpc3."""
    p = FLAT | {
        "m": FLAT_M.ustrip("Msun"),
        "r_s": FLAT_RS.ustrip("kpc"),
        "r_q": 30.0,
        "r_cut": 500.0,
    }
    return jnp.exp(log_density(p, X, 0.0))


# ==============================================================================
# flattened_gnfw_halo


def test_spherical_limit_is_nfw() -> None:
    """Gamma 1, beta 3, q 1 and a far cutoff is NFW at the same ``m``."""
    halo = gp.flattened_gnfw_halo(M, RS, units="galactic", **SPHERICAL)
    nfw = gp.NFWPotential(m=M, r_s=RS, units="galactic")
    assert halo.symmetry == gp.Symmetry.ZROTATION_ZREFLECTION
    for f, rtol in (("potential", 2e-6), ("gradient", 2e-6), ("density", 1e-5)):
        got, exp = getattr(halo, f)(X, 0), getattr(nfw, f)(X, 0)
        assert jnp.allclose(got, exp, atol=0, rtol=rtol), f
    assert jnp.allclose(halo.hessian(X, 0), nfw.hessian(X, 0), atol=1e-12, rtol=1e-5)


@pytest.mark.parametrize(("l_max", "rtol"), [(4, 5e-2), (8, 2e-3), (12, 3e-4)])
def test_poisson(l_max: int, rtol: float) -> None:
    r"""$\nabla^2\Phi = 4\pi G\rho$ against the closed-form density."""
    halo = _flat(l_max)
    lap = jnp.trace(halo.hessian(X, 0), axis1=-2, axis2=-1)
    G = halo.constants["G"].ustrip("kpc3 / (Msun Myr2)")
    assert jnp.allclose(lap, 4 * jnp.pi * G * _true_density(), rtol=rtol, atol=0)


def test_density_is_the_model() -> None:
    """The expansion's reconstructed density matches the closed form."""
    assert jnp.allclose(_flat(12).density(X, 0), _true_density(), rtol=1e-3, atol=0)


@pytest.mark.parametrize(
    ("r_q", "q"), [(u.Q(1e-8, "kpc"), "q_out"), (u.Q(1e8, "kpc"), "q_in")]
)
def test_q_of_r_limits(r_q: u.Q, q: str) -> None:
    """A tiny ``r_q`` leaves ``q_out`` everywhere and a huge one ``q_in``."""
    varying = _flat(8, r_q=r_q)
    constant = _flat(8, q_in=FLAT[q], q_out=FLAT[q])
    assert jnp.allclose(varying.potential(X, 0), constant.potential(X, 0), rtol=1e-9)


# ==============================================================================
# doi_10_1093_mnras_stad551


def _in_frame(x: jax.Array) -> jax.Array:
    """Positions in the halo's frame, up to a rotation about its axis."""
    n = AXIS / jnp.linalg.norm(AXIS)
    along = x @ n
    across = jnp.sqrt(jnp.sum(x**2, axis=-1) - along**2)
    return jnp.stack([across, jnp.zeros_like(along), along], axis=-1)


def test_koposov_axis() -> None:
    """The halo is the z-aligned one, with ``axis`` as its z-axis.

    An axisymmetric halo's potential depends only on the distance along its
    axis and the distance from it.
    """
    aligned = _koposov(axis=[0.0, 0.0, 1.0])
    got, exp = _koposov().potential(X, 0), aligned.potential(_in_frame(X), 0)
    assert jnp.allclose(got, exp, rtol=1e-12, atol=0)


@pytest.mark.parametrize("axis", [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
def test_koposov_along_z_is_unrotated(axis: list[float]) -> None:
    """Along +-z, where the axis's azimuth is undefined, nothing is rotated."""
    halo = _koposov(axis=axis)
    base = halo.base_potential
    assert jnp.allclose(halo.potential(X, 0), base.potential(X, 0), rtol=1e-14)
    assert jnp.allclose(halo.gradient(X, 0), base.gradient(X, 0), rtol=1e-10)
    assert jnp.allclose(halo.hessian(X, 0), base.hessian(X, 0), rtol=1e-10)


def test_koposov_force_on_axis() -> None:
    """On the axis the force points along it, as the z-aligned halo's does on z."""
    s = jnp.asarray([0.5, 10.0, 100.0])[:, None]
    n = AXIS / jnp.linalg.norm(AXIS)
    got = _koposov().gradient(s * n, 0)
    on_z = _koposov(axis=[0.0, 0.0, 1.0]).gradient(s * jnp.asarray([0, 0, 1.0]), 0)
    exp = on_z[:, 2:] * n
    assert jnp.allclose(got, exp, rtol=1e-10, atol=1e-12 * jnp.max(jnp.abs(exp)))


def test_koposov_poisson() -> None:
    r"""Tilting keeps $\nabla^2\Phi = 4\pi G\rho$ as good as the z-aligned halo's.

    The Laplacian through the rotation equals the z-aligned halo's at the same
    halo-frame positions, and both match the closed-form density.
    """
    halo = _koposov(12)
    lap = jnp.trace(halo.hessian(X, 0), axis1=-2, axis2=-1)
    aligned = _koposov(12, axis=[0.0, 0.0, 1.0])
    lap_z = jnp.trace(aligned.hessian(_in_frame(X), 0), axis1=-2, axis2=-1)
    assert jnp.allclose(lap, lap_z, rtol=1e-10, atol=0)
    p = {
        "m": M_NFW.ustrip("Msun") / (jnp.log(16.0) - 15 / 16),
        "r_s": FLAT_RS.ustrip("kpc"),
        "gamma": KOPOSOV["gamma"],
        "beta": KOPOSOV["beta"],
        "q_in": KOPOSOV["q"],
        "q_out": KOPOSOV["q"],
        "r_q": 30.0,
        "r_cut": 500.0,
    }
    rho = jnp.exp(log_density(p, _in_frame(X), 0.0))
    G = halo.constants["G"].ustrip("kpc3 / (Msun Myr2)")
    assert jnp.allclose(lap, 4 * jnp.pi * G * rho, rtol=1e-3, atol=0)


@pytest.mark.parametrize("axis", [[0.5, 0.9, 1.7], [-0.25, -0.45, -0.85]])
def test_koposov_axis_scale_and_sign(axis: list[float]) -> None:
    """``axis`` need not be normalized, and its sign does not matter."""
    got, exp = _koposov(axis=axis).potential(X, 0), _koposov().potential(X, 0)
    assert jnp.allclose(got, exp, rtol=1e-12, atol=0)


def test_koposov_mass() -> None:
    """``m_nfw`` is M_NFW at c_NFW = 15: NFW with ``m = M_NFW / 1.8351``, cut off."""
    halo = koposov2023(M_NFW, RS, units="galactic")
    nfw = gp.NFWPotential(m=M_NFW / 1.8351, r_s=RS, units="galactic")
    cutoff = jnp.exp(-jnp.sum(X**2, axis=-1) / 500.0**2)
    assert jnp.allclose(halo.density(X, 0), nfw.density(X, 0) * cutoff, rtol=3e-5)


def test_koposov_jit_grad() -> None:
    """Jit and grad through the build in the physical parameters and the axis."""
    x = X[1]

    @jax.jit
    def phi(theta: jax.Array) -> jax.Array:
        m, r_s, q, gamma, ax = theta
        halo = koposov2023(
            u.Q(m, "Msun"),
            u.Q(r_s, "kpc"),
            q=q,
            gamma=gamma,
            axis=AXIS.at[0].set(ax),
            l_max=4,
            n_r=64,
            units="galactic",
        )
        return halo.potential(x, 0)

    theta = jnp.asarray([1e12, 16.0, 0.6, 1.2, 0.25])
    got = jax.grad(phi)(theta)
    h = 1e-5 * theta
    fd = jnp.asarray(
        [
            (phi(theta.at[i].add(h[i])) - phi(theta.at[i].add(-h[i]))) / (2 * h[i])
            for i in range(theta.shape[0])
        ]
    )
    assert jnp.all(got != 0)
    assert jnp.allclose(got, fd, rtol=1e-6)


def test_koposov_orbit_conserves_energy_and_axial_angular_momentum() -> None:
    """A leapfrog orbit keeps E and L along ``axis``, which a tilt must preserve."""
    halo = _koposov()
    n = AXIS / jnp.linalg.norm(AXIS)
    dt = 0.5  # Myr

    def step(w: jax.Array, _: Any) -> tuple[jax.Array, jax.Array]:
        q, p = w[:3], w[3:]
        p = p - 0.5 * dt * halo.gradient(q, 0)
        q = q + dt * p
        p = p - 0.5 * dt * halo.gradient(q, 0)
        w = jnp.concat([q, p])
        return w, w

    w0 = jnp.asarray([20.0, 3.0, 5.0, 0.03, 0.17, 0.08])  # kpc, kpc / Myr
    _, ws = jax.lax.scan(step, w0, length=4000)  # 2 Gyr
    E = halo.potential(ws[:, :3], 0) + 0.5 * jnp.sum(ws[:, 3:] ** 2, axis=-1)
    Ln = jnp.cross(ws[:, :3], ws[:, 3:]) @ n
    assert jnp.all(jnp.isfinite(ws))
    assert jnp.ptp(E) / jnp.abs(E[0]) < 1e-4
    assert jnp.ptp(Ln) / jnp.abs(Ln[0]) < 1e-3
