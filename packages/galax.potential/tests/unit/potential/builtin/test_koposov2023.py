"""Tests for `galax.potential.KoposovEtAl2023HaloPotential`."""

from typing import Any, ClassVar, override

import jax
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from galax.potential._src.builtin.koposov2023 import log_density, rotation_to_z

K = gp.KoposovEtAl2023HaloPotential
M, RS = u.Q(5.4e11, "Msun"), u.Q(15.0, "kpc")

# The spherical, uncut limit. NFW's potential depends on all the mass outside
# a radius, so the grid reaches far enough out that the expansion's outer
# power-law continuation does not set the error.
SPHERICAL = {"r_cut": u.Q(1e8, "kpc"), "r_max": u.Q(1e5, "kpc"), "n_r": 192}

# Strongly flattened and tilted, with the flattening changing with radius.
TILTED = {
    "gamma": 1.27,
    "beta": 2.97,
    "q_in": 0.55,
    "q_out": 0.9,
    "axis": jnp.asarray([0.25, 0.45, 0.85]),
}
TILTED_M, TILTED_RS = u.Q(10**12.05, "Msun"), u.Q(16.1, "kpc")

X = jnp.asarray(
    [
        [8.0, 1.0, 0.5],
        [20.0, -5.0, 10.0],
        [50.0, 30.0, -40.0],
        [0.3, 0.2, 0.1],
        [150.0, -80.0, 60.0],
    ]
)


def _nfw() -> gp.NFWPotential:
    return gp.NFWPotential(m=M, r_s=RS, units="galactic")


def _tilted(l_max: int, **kw: Any) -> gp.KoposovEtAl2023HaloPotential:
    return K.from_params(
        TILTED_M, TILTED_RS, l_max=l_max, units="galactic", **(TILTED | kw)
    )


def _true_density(axis: Any = TILTED["axis"]) -> jax.Array:
    """Return the closed-form density of the tilted halo at `X`, in Msun / kpc3."""
    p = {k: v for k, v in TILTED.items() if k != "axis"} | {
        "m": TILTED_M.ustrip("Msun"),
        "r_s": TILTED_RS.ustrip("kpc"),
        "r_q": 30.0,
        "r_cut": 500.0,
    }
    return jnp.exp(log_density(p, X @ rotation_to_z(axis).T, 0.0))


class TestKoposovEtAl2023HaloPotential(AbstractSinglePotential_Test):
    """The shared suite, on the spherical limit, against `NFWPotential`."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.KoposovEtAl2023HaloPotential]:
        return K

    @pytest.fixture(scope="class")
    @override
    def fields_(self, field_units: u.AbstractUnitSystem) -> dict[str, Any]:
        built = K.from_params(M, RS, units=field_units, **SPHERICAL)
        return {"expansion": built.expansion, "axis": built.axis, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3) -> None:
        got, exp = pot.potential(x, t=0), _nfw().potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(0.0, exp.unit), rtol=1e-6)

    def test_gradient(self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3) -> None:
        got, exp = pot.gradient(x, t=0), _nfw().gradient(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Q(0.0, exp.unit), rtol=1e-6)

    def test_density(self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3) -> None:
        got, exp = pot.density(x, t=0), _nfw().density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(0.0, exp.unit), rtol=1e-6)

    def test_hessian(self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3) -> None:
        got, exp = pot.hessian(x, t=0), _nfw().hessian(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Q(1e-12, exp.unit), rtol=1e-5)

    def test_tidal_tensor(
        self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3
    ) -> None:
        got, exp = pot.tidal_tensor(x, t=0), _nfw().tidal_tensor(x, t=0)
        assert jnp.allclose(got, exp, atol=u.Q(1e-12, exp.unit), rtol=1e-5)

    @override
    def test_potential_density_correspondence(
        self, pot: gp.KoposovEtAl2023HaloPotential, x: gt.QuSz3
    ) -> None:
        """Poisson holds to the expansion's accuracy, not to round-off."""
        lhs = jnp.trace(pot.hessian(x, 0))
        rhs = 4 * jnp.pi * pot.constants["G"] * pot.density(x, 0)
        assert jnp.isclose(lhs, rhs, atol=u.Q(0.0, lhs.unit), rtol=1e-5)

    # ==========================================================================

    def test_spherical_limit_is_nfw(self, pot: gp.KoposovEtAl2023HaloPotential) -> None:
        """Gamma 1, beta 3, q 1 and a far cutoff is NFW at the same ``m``."""
        nfw = _nfw()
        assert jnp.allclose(pot.potential(X, 0), nfw.potential(X, 0), rtol=1e-6)
        assert jnp.allclose(pot.gradient(X, 0), nfw.gradient(X, 0), rtol=1e-6)


# ==============================================================================


@pytest.mark.parametrize(("l_max", "rtol"), [(4, 5e-2), (8, 2e-3), (12, 3e-4)])
def test_poisson_tilted(l_max: int, rtol: float) -> None:
    r"""$\nabla^2\Phi = 4\pi G\rho$ against the closed-form density."""
    halo = _tilted(l_max)
    lap = jnp.trace(halo.hessian(X, 0), axis1=-2, axis2=-1)
    G = halo.constants["G"].ustrip("kpc3 / (Msun Myr2)")
    assert jnp.allclose(lap, 4 * jnp.pi * G * _true_density(), rtol=rtol, atol=0)


def test_density_is_the_model() -> None:
    """The expansion's reconstructed density matches the closed form."""
    got = _tilted(12).density(X, 0)
    assert jnp.allclose(got, _true_density(), rtol=1e-3, atol=0)


def test_axis_rotates_the_halo() -> None:
    """A tilted halo is the z-aligned one in rotated coordinates."""
    tilted = _tilted(8)
    aligned = _tilted(8, axis=jnp.asarray([0.0, 0.0, 1.0]))
    R = rotation_to_z(TILTED["axis"])
    assert jnp.allclose(tilted.potential(X, 0), aligned.potential(X @ R.T, 0))
    assert jnp.allclose(tilted.gradient(X, 0), aligned.gradient(X @ R.T, 0) @ R)


@pytest.mark.parametrize("axis", [[0.5, 0.9, 1.7], [-0.25, -0.45, -0.85]])
def test_axis_scale_and_sign(axis: list[float]) -> None:
    """``axis`` need not be normalized, and its sign does not matter."""
    halo = _tilted(8)
    flipped = _tilted(8, axis=jnp.asarray(axis))
    assert jnp.allclose(halo.potential(X, 0), flipped.potential(X, 0), rtol=1e-14)


def test_rotation_to_z() -> None:
    """The rotation is orthonormal and takes the axis to z, either hemisphere."""
    for axis in ([0.25, 0.45, 0.85], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0, 0, 1.0]):
        n = jnp.asarray(axis) / jnp.linalg.norm(jnp.asarray(axis))
        R = rotation_to_z(n)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-14)
        assert jnp.allclose(jnp.abs(R @ n), jnp.asarray([0.0, 0.0, 1.0]), atol=1e-14)


@pytest.mark.parametrize(
    ("r_q", "q"), [(u.Q(1e-8, "kpc"), "q_out"), (u.Q(1e8, "kpc"), "q_in")]
)
def test_q_of_r_limits(r_q: u.Q, q: str) -> None:
    """A tiny ``r_q`` leaves ``q_out`` everywhere and a huge one ``q_in``."""
    varying = _tilted(8, r_q=r_q)
    constant = _tilted(8, q_in=TILTED[q], q_out=TILTED[q])
    assert jnp.allclose(varying.potential(X, 0), constant.potential(X, 0), rtol=1e-9)


def test_jit_grad_through_build() -> None:
    """The build is traceable: jit and grad in the physical parameters."""
    x = X[1]

    @jax.jit
    def phi(theta: jax.Array) -> jax.Array:
        m, r_s, q_in, gamma, r_cut = theta
        halo = K.from_params(
            u.Q(m, "Msun"),
            u.Q(r_s, "kpc"),
            q_in=q_in,
            q_out=0.9,
            gamma=gamma,
            r_cut=u.Q(r_cut, "kpc"),
            axis=TILTED["axis"],
            l_max=4,
            n_r=64,
            units="galactic",
        )
        return halo.potential(x, 0)

    theta = jnp.asarray([1e12, 16.0, 0.6, 1.2, 200.0])
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


def test_orbit_conserves_energy_and_axial_angular_momentum() -> None:
    """A leapfrog orbit keeps E and L along ``axis``, which a tilt must preserve."""
    halo = _tilted(8)
    n = TILTED["axis"] / jnp.linalg.norm(TILTED["axis"])
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
