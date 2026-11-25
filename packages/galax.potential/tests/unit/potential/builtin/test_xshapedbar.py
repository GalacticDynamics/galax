"""Tests for `galax.potential.XShapedBarPotential`."""

import functools as ft

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.builtin.xshapedbar import (
    _BAR_1,
    _BAR_2,
    _BAR_3,
    M_TOT,
    _components,
)

G = 4.498502151469554e-12  # kpc^3 / (Msun Myr^2)
T = u.Q(0.0, "Gyr")

# Sormani et al. (2022) Table 1, as printed (density in 1e10 Msun / kpc^3).
TABLE_1 = (
    (
        _BAR_1,
        {
            "rho": 0.316,
            "x": 0.490,
            "y": 0.392,
            "z": 0.229,
            "c_par": 1.991,
            "c_perp": 2.232,
            "m": 0.873,
            "alpha": 0.626,
            "n": 1.940,
            "c": 1.342,
            "x_c": 0.751,
            "y_c": 0.469,
            "r_cut": 4.370,
        },
    ),
    (
        _BAR_2,
        {
            "rho": 0.050,
            "x": 5.364,
            "y": 0.959,
            "z": 0.611,
            "n": 3.051,
            "c_perp": 0.970,
            "R_out": 3.190,
            "R_in": 0.558,
            "n_out": 16.731,
            "n_in": 3.196,
        },
    ),
    (
        _BAR_3,
        {
            "rho": 1743.049,
            "x": 0.478,
            "y": 0.267,
            "z": 0.252,
            "n": 0.980,
            "c_perp": 1.879,
            "R_out": 2.204,
            "R_in": 7.607,
            "n_out": -27.291,
            "n_in": 1.630,
        },
    ),
)


@pytest.fixture(scope="module")
def bar() -> gp.XShapedBarPotential:
    return gp.XShapedBarPotential.build()


@ft.cache
def _octant_integrals() -> tuple[np.ndarray, np.ndarray]:
    """Masses and ``int rho (x^2 - y^2) dV`` per component, midpoint rule."""
    h = 0.05
    g = np.arange(h / 2, 12, h)
    x, y = np.meshgrid(g, g, indexing="ij")
    f = jax.jit(lambda xyz: jnp.stack(_components(xyz)))
    mass, d = np.zeros(3), np.zeros(3)
    for z in np.arange(h / 2, 4, h):
        rho = np.asarray(f(jnp.stack([x, y, jnp.full_like(x, z)], -1)))
        mass += 8 * h**3 * rho.sum(axis=(1, 2))
        d += 8 * h**3 * (rho * (x**2 - y**2)).sum(axis=(1, 2))
    return mass, d


# ============================================================================
# The density


def test_parameters_round_to_table_1() -> None:
    for params, table in TABLE_1:
        assert params.keys() == table.keys()
        for name, printed in table.items():
            value = params[name] / (1e10 if name == "rho" else 1)
            assert round(value, 3) == printed, name


def test_component_masses() -> None:
    """Sormani et al. (2022) Sect. 3: 1.28, 0.33 and 0.22 x 1e10 Msun."""
    mass, _ = _octant_integrals()
    assert tuple(np.round(mass / 1e10, 2)) == (1.28, 0.33, 0.22)
    assert np.isclose(mass.sum(), M_TOT, rtol=2e-5)


def test_density_symmetry() -> None:
    xyz = jax.random.normal(jax.random.key(0), (64, 3)) * jnp.asarray([3.0, 2.0, 0.5])
    rho = gp.XShapedBarPotential.model_density(u.Q(xyz, "kpc"))
    for flip in ([-1, 1, 1], [1, -1, 1], [1, 1, -1]):
        flipped = u.Q(xyz * jnp.asarray(flip), "kpc")
        got = gp.XShapedBarPotential.model_density(flipped)
        assert jnp.array_equal(got, rho)


def test_density_is_finite_everywhere() -> None:
    """On the axes, at the origin, and far away, where the cutoffs overflow."""
    xyz = jnp.asarray(
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1e8, 0, 0], [0, 0, 1e8]],
        dtype=float,
    )
    rho = gp.XShapedBarPotential.model_density(u.Q(xyz, "kpc")).ustrip("Msun/kpc3")
    assert jnp.all(jnp.isfinite(rho))
    assert jnp.all(rho[-2:] == 0)


# ============================================================================
# The potential


def test_total_mass(bar: gp.XShapedBarPotential) -> None:
    """The monopole far away carries the model's mass."""
    r = 1e3
    phi = bar.potential(u.Q([r, 0.0, 0.0], "kpc"), T).ustrip("kpc2/Myr2")
    assert jnp.isclose(-phi * r / G, M_TOT, rtol=3e-4)


# Points in the X-shaped bar, off its symmetry planes, and the relative error
# of the Laplacian of the default build (l_max = 24) there: 5e-6, 1e-5, 2.2e-3
# and 8.3e-3 (at l_max = 16: 4e-6, 2e-5, 5e-3, 5e-2; at 8: 2e-5, 4e-3, 4e-2,
# 0.24). In the thin long bar (R ~ 3-5 kpc, |z| < 0.5 kpc) the truncated series
# rings and the Laplacian is off by tens of percent at these l_max, while the
# force is good to 1%; see `build`.
POISSON = [
    ((0.2, 0.1, 0.1), 1e-4),
    ((1.0, 0.5, 0.3), 1e-4),
    ((2.0, 0.5, 0.2), 5e-3),
    ((1.5, 1.5, 1.0), 2e-2),
]


@pytest.mark.parametrize(("xyz", "rtol"), POISSON)
def test_poisson(
    bar: gp.XShapedBarPotential, xyz: tuple[float, ...], rtol: float
) -> None:
    """The Laplacian of the potential is 4 pi G rho of the analytic density."""
    q = u.Q(jnp.asarray(xyz), "kpc")
    lap = bar.laplacian(q, T).ustrip("1/Myr2")
    rho = bar.model_density(q).ustrip("Msun/kpc3")
    assert jnp.isclose(lap, 4 * jnp.pi * G * rho, rtol=rtol)


def test_poisson_outside_the_bar(bar: gp.XShapedBarPotential) -> None:
    """Where the density is negligible the potential is harmonic."""
    q = u.Q([[10.0, 0.0, 3.0], [0.0, 12.0, 2.0], [7.0, 7.0, 5.0]], "kpc")
    lap = bar.laplacian(q, T).ustrip("1/Myr2")
    # The scale: 4 pi G rho at the centre of the bar.
    scale = (
        4 * jnp.pi * G * bar.model_density(u.Q([0.0, 0, 0], "kpc")).ustrip("Msun/kpc3")
    )
    assert jnp.all(jnp.abs(lap) < 1e-6 * scale)


@pytest.mark.parametrize(("r", "rtol"), [(20.0, 3e-2), (50.0, 3e-3), (300.0, 1e-3)])
def test_far_field_quadrupole(
    bar: gp.XShapedBarPotential, r: float, rtol: float
) -> None:
    r"""Beyond the bar the m = 2 part falls as :math:`r^{-3}`.

    It is :math:`-(3/4) G D \sin^2\theta\cos 2\phi / r^3`, so
    :math:`\Phi(r, 0, 0) - \Phi(0, r, 0) = -(3/2) G D / r^3`, up to the
    :math:`l \geq 4` terms, which fall as :math:`r^{-5}`. The last radius is
    past the grid, on the fitted power-law continuation.
    """
    _, d = _octant_integrals()
    q = u.Q([[r, 0.0, 0.0], [0.0, r, 0.0]], "kpc")
    phi = bar.potential(q, T).ustrip("kpc2/Myr2")
    assert jnp.isclose(phi[0] - phi[1], -1.5 * G * d.sum() / r**3, rtol=rtol)


def test_m_tot_rescales(bar: gp.XShapedBarPotential) -> None:
    q = u.Q([[3.0, 1.0, 0.2], [20.0, 5.0, 1.0]], "kpc")
    heavy = gp.XShapedBarPotential.build(m_tot=u.Q(2 * M_TOT, "Msun"))
    acc, acc2 = (p.gradient(q, T).ustrip("kpc/Myr2") for p in (bar, heavy))
    assert jnp.allclose(acc2, 2 * acc, rtol=1e-12, atol=0)
    rho = bar.model_density(q).ustrip("Msun/kpc3")
    rho2 = bar.model_density(q, m_tot=u.Q(2 * M_TOT, "Msun")).ustrip("Msun/kpc3")
    assert jnp.allclose(rho2, 2 * rho, rtol=1e-12, atol=0)


def test_jit_and_grad(bar: gp.XShapedBarPotential) -> None:
    xyz = jnp.asarray([2.0, 1.0, 0.3])

    @jax.jit
    def phi(xyz: jax.Array) -> jax.Array:
        return bar.potential(u.Q(xyz, "kpc"), T).ustrip("kpc2/Myr2")

    g = jax.grad(phi)(xyz)
    assert jnp.allclose(g, bar.gradient(u.Q(xyz, "kpc"), T).ustrip("kpc/Myr2"))

    # Through the build: the potential is linear in `m_tot`.
    def phi_of_m(m: jax.Array) -> jax.Array:
        pot = gp.XShapedBarPotential.build(m_tot=u.Q(m, "Msun"))
        return pot.potential(u.Q(xyz, "kpc"), T).ustrip("kpc2/Myr2")

    dphi_dm = jax.grad(phi_of_m)(M_TOT)
    assert jnp.isclose(dphi_dm, phi(xyz) / M_TOT, rtol=1e-10)
