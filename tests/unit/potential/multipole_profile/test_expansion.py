"""Tests for expansion evaluation."""

import jax
import pytest

import quaxed.numpy as jnp

from galax.potential._src.builtin.multipole_profile.build import build_expansion
from galax.potential._src.builtin.multipole_profile.expansion import (
    expansion_density,
    expansion_potential,
)
from galax.potential._src.harmonic import (
    default_angular_resolution,
    lm_keys,
)


def _hernquist_params(n_r: int = 256):
    """M=1, a=1 Hernquist: rho = 1/(2 pi) / (r (1+r)^3), Phi = -1/(1+r), G=1."""

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        return 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)

    keys = lm_keys(0, "spherical")
    n_theta, n_phi = default_angular_resolution(0)
    r = jnp.geomspace(1e-3, 1e3, n_r)
    p = build_expansion(
        rho, r, 0, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
    )
    return {**p, "r_knots": r}, keys


def test_expansion_potential_matches_the_hernquist_closed_form() -> None:
    p, keys = _hernquist_params()
    xyz = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.0, 0.0], [10.0, 5.0, 2.0]])
    r = jnp.linalg.norm(xyz, axis=-1)
    got = expansion_potential(p, xyz, 0, keys)
    assert jnp.allclose(got, -1.0 / (1.0 + r), rtol=1e-3)


def test_expansion_density_matches_the_hernquist_closed_form() -> None:
    """The rho_lm splines are far more accurate than laplacian(Phi)."""
    p, keys = _hernquist_params()
    xyz = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.0, 0.0], [10.0, 5.0, 2.0]])
    r = jnp.linalg.norm(xyz, axis=-1)
    expect = 1.0 / (2.0 * jnp.pi) / (r * (1.0 + r) ** 3)
    assert jnp.allclose(expansion_density(p, xyz, 0, keys), expect, rtol=1e-4)


def test_expansion_potential_is_batched_consistently() -> None:
    p, keys = _hernquist_params()
    batch = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = expansion_potential(p, batch, 0, keys)
    assert got.shape == (2,)
    assert jnp.allclose(got[0], expansion_potential(p, batch[0], 0, keys))


def test_gradient_is_finite_on_the_z_axis() -> None:
    """An angular harmonic gives NaN here; this is the regression guard."""

    def rho(xyz, t):
        return jnp.exp(-jnp.linalg.norm(xyz, axis=-1))

    keys = lm_keys(4, None)
    n_theta, n_phi = default_angular_resolution(4)
    r = jnp.geomspace(1e-2, 1e2, 64)
    p = {
        **build_expansion(
            rho, r, 4, keys, n_theta, n_phi, jnp.asarray(0.0), jnp.asarray(1.0)
        ),
        "r_knots": r,
    }

    grad = jax.grad(lambda xyz: expansion_potential(p, xyz, 4, keys))(
        jnp.asarray([0.0, 0.0, 2.0])
    )
    assert jnp.all(jnp.isfinite(grad))


def test_expansion_potential_is_jittable() -> None:
    """`l_max` and `keys` are static, so shapes are compile-time constants."""
    p, keys = _hernquist_params(n_r=32)
    jitted = jax.jit(expansion_potential, static_argnums=(2, 3))
    assert jnp.isfinite(jitted(p, jnp.asarray([1.0, 2.0, 3.0]), 0, keys))


@pytest.mark.parametrize("l_max", [0, 2, 4, 8])
def test_the_expansion_is_finite_at_the_origin(l_max: int) -> None:
    """Value, density and gradient must all stay finite at ``xyz == 0``.

    `_log_r_and_ylm` forms ``uvec = xyz / safe_vector_norm(xyz)``, and at the
    origin that really is the zero vector -- ``safe_vector_norm`` floors the
    norm at ``sqrt(tiny)``, so the division underflows to zero rather than
    blowing up. A zero vector is not a direction, which looks like it should
    poison the harmonics.

    It does not: `real_ylm` evaluates Cartesian recurrences that are
    polynomial in the components, so it never divides by the direction's
    norm and a zero input gives finite values. That is a property of
    `real_ylm`, not something the caller arranges, so it is pinned here --
    the gradient especially, since `jax.grad` through an underflowed norm is
    where a `0/0` would surface first.
    """
    keys = lm_keys(l_max, "none")

    def rho(xyz, t):
        r = jnp.linalg.norm(xyz, axis=-1)
        safe = jnp.where(r > 0, r, 1e-30)
        return 1.0 / (2.0 * jnp.pi) / (safe * (1.0 + safe) ** 3)

    r_knots = jnp.geomspace(0.05, 20.0, 64)
    p = {
        **build_expansion(
            rho, r_knots, l_max, keys, 12, 12, jnp.asarray(0.0), jnp.asarray(1.0)
        ),
        "r_knots": r_knots,
    }
    origin = jnp.zeros((1, 3))

    assert jnp.all(jnp.isfinite(expansion_potential(p, origin, l_max, keys)))
    assert jnp.all(jnp.isfinite(expansion_density(p, origin, l_max, keys)))

    grad = jax.grad(lambda x: expansion_potential(p, x[None], l_max, keys)[0])
    assert jnp.all(jnp.isfinite(grad(jnp.zeros(3))))


@pytest.mark.parametrize("log10_r", [20.0, 38.0, 150.0, 300.0])
def test_the_density_is_finite_far_outside_the_knots(log10_r: float) -> None:
    """One absurd radius must not return `nan`.

    REGRESSION: the analytic cusp term was clamped but the residual spline
    was not, and `eval_log_spline` continues the edge cubic with an
    *unbounded* local coordinate -- so `s**3` overflowed and the density came
    back `nan` from ``r = 1e20`` in float32 and ``1e300`` in float64. The
    clamp on the cusp was there specifically to stop a single bad radius
    poisoning a vmapped batch, and the residual walked straight around it.

    Saturating costs nothing real: unlike the potential, the density is not
    continued outside the knots, and its value there is already documented as
    meaningless.
    """
    p, keys = _hernquist_params(n_r=64)
    xyz = jnp.asarray([[10.0**log10_r, 0.0, 0.0]])

    got = expansion_density(p, xyz, 0, keys)
    assert jnp.all(jnp.isfinite(got)), got


@pytest.mark.parametrize("amplitude", [1e3, 1e-3])
def test_the_density_cusp_clamp_holds_in_float32(amplitude: float) -> None:
    """The clamp exists for float32, so it has to be tested in float32.

    ``exp`` overflows at ~88 in float32, which is what `galax` runs by
    default, while ``pyproject.toml`` forces x64 for the suite -- so every
    other test here exercises this code in a precision where the clamp is
    nearly unreachable. That gap is not hypothetical: the original test used
    an ``r^-2`` cusp whose ``|z|`` peaks at 85.6, just under the float32
    limit, so test and code shared a blind spot and the clamp was wrong three
    times before it was right.

    Both amplitudes matter, because the two ways to get this wrong fail on
    different sides. Bounding ``z`` alone survives ``|amp| < 1`` and
    overflows for ``|amp| > 1``; bounding ``z`` by ``ln_huge - log|amp|`` is
    *looser* than ``ln_huge`` exactly when ``|amp| < 1``. Only clamping the
    whole ``log|amp| + z`` handles both.

    The coefficients are built once under the suite's x64 and then cast, so
    only the evaluation runs in float32. Building inside the block instead
    re-traces the projection, which asks for float64 explicitly and trips
    ``filterwarnings = ["error"]`` -- a real wart, but not this test's
    subject.
    """
    p, keys = _hernquist_params(n_r=64)
    p32 = {
        k: (v.astype(jnp.float32) if hasattr(v, "astype") else v) for k, v in p.items()
    }
    p32["rho_amplitude"] = jnp.full_like(p32["rho_amplitude"], amplitude)

    # The origin and far outside the grid: where the exponent is largest in
    # each direction.
    rq = jnp.asarray([0.0, 1e-12, 1.0, 1e12, 1e20], dtype=jnp.float32)
    xyz = jnp.stack([rq, jnp.zeros_like(rq), jnp.zeros_like(rq)], -1)

    with jax.enable_x64(False):  # noqa: FBT003
        got = expansion_density(p32, xyz, 0, keys)
        assert got.dtype == jnp.float32  # no silent promotion to x64

    assert jnp.all(jnp.isfinite(got)), got
