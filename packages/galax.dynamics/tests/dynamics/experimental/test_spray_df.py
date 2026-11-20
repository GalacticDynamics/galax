"""Tests for the experimental spray DFs and vectorized release sampling."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import ks_2samp

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics._src.experimental.df import (
    Chen2024DF,
    Fardal2015DF,
    sample_releases,
)

MSAT = 1e5
DFS = [Fardal2015DF, Chen2024DF]


def nfw(m: float = 1e12) -> gp.NFWPotential:
    return gp.NFWPotential(m=m, r_s=15.0, units="galactic")


def orbit(n: int, pot: gp.AbstractPotential) -> tuple[np.ndarray, ...]:
    """``n`` points along a progenitor orbit: t, x, v as raw arrays."""
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km/s"), t=u.Q(0, "Myr")
    )
    prog = gd.evaluate_orbit(pot, w0, u.Q(jnp.linspace(-3000.0, 0.0, n), "Myr"))
    pw = np.asarray(prog.w(units=pot.units))
    return np.asarray(u.ustrip("Myr", prog.t)), pw[:, :3], pw[:, 3:]


def frame(
    pot: gp.AbstractPotential, t: np.ndarray, x: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Release frame (x, y, z unit vectors) and tidal radius, shape (n, 1)."""
    r = np.linalg.norm(x, axis=1, keepdims=True)
    xh = x / r
    zh = np.cross(x, v)
    zh /= np.linalg.norm(zh, axis=1, keepdims=True)
    yh = np.cross(zh, xh)
    rt = np.asarray(
        u.ustrip(
            "kpc",
            gd.tidal_radius(
                pot,
                u.Q(x, "kpc"),
                u.Q(v, "kpc/Myr"),
                mass=u.Q(MSAT, "Msun"),
                t=u.Q(t, "Myr"),
            ),
        )
    )[:, None]
    return xh, yh, zh, rt


def vscale(df: object, pot: gp.AbstractPotential, x, v, rt) -> np.ndarray:
    """Each DF's velocity scale: v_esc(r_t) for Chen+24, omega r_t for Fardal+15."""
    if isinstance(df, Chen2024DF):
        G = u.ustrip("kpc3 / (Msun Myr2)", pot.constants["G"])
        return np.sqrt(2 * G * MSAT / rt)
    L = np.linalg.norm(np.cross(x, v), axis=1, keepdims=True)
    return L / np.linalg.norm(x, axis=1, keepdims=True) ** 2 * rt


def offsets(df, pot, t, x, v, sample, arm: str, sign: float) -> np.ndarray:
    """Standardized offsets: size, x, y, z of positions then velocities.

    x and y are rotated back for the leading arm.
    """
    xh, yh, zh, rt = frame(pot, t, x, v)
    xa, va = sample[:2] if arm == "lead" else sample[2:]
    xa, va = np.asarray(xa), np.asarray(va)
    cols = []
    for d in ((xa - x) / rt, (va - v) / vscale(df, pot, x, v, rt)):
        cols += [np.linalg.norm(d, axis=1), sign * (d * xh).sum(1)]
        cols += [sign * (d * yh).sum(1), (d * zh).sum(1)]
    return np.column_stack(cols)


@pytest.mark.parametrize("df_cls", DFS)
def test_sample_releases_matches_loop(df_cls: type) -> None:
    """The vectorized sampler equals a loop over ``jr.split(key, n)`` keys.

    Bitwise for Fardal; to the last bits for Chen (see below).
    """
    df, pot, n = df_cls(), nfw(), 5
    t, x, v = orbit(n, pot)
    Ms = np.full(n, MSAT)
    key = jr.key(3)
    got = sample_releases(df, key, pot, t, x, v, Ms)
    keys = jr.split(key, n)
    for i in range(n):
        want = df.sample(keys[i], pot, t[i], x[i], v[i], Ms[i])
        for g, w in zip(got, want, strict=True):
            assert g.shape == (n, 3)
            if df_cls is Fardal2015DF:
                assert np.array_equal(np.asarray(g[i]), np.asarray(w))
            else:
                # The batched SVD in Chen's draw differs from the unbatched one
                # at the last bit.
                np.testing.assert_allclose(g[i], w, rtol=1e-13, atol=1e-13)


def test_chen_matches_legacy_distribution() -> None:
    """Each standardized offset component matches the legacy ``ChenStreamDF``."""
    pot, n = nfw(), 4000
    t, x, v = orbit(n, pot)
    df = Chen2024DF()
    new = sample_releases(df, jr.key(0), pot, t, x, v, np.full(n, MSAT))

    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km/s"), t=u.Q(0, "Myr")
    )
    prog = gd.evaluate_orbit(pot, w0, u.Q(jnp.linspace(-3000.0, 0.0, n), "Myr"))
    mock = gd.ChenStreamDF().sample(jr.key(1), pot, prog, u.Q(MSAT, "Msun"))
    old = tuple(
        np.asarray(mock[arm].w(units=pot.units))[:, sl]
        for arm in ("lead", "trail")
        for sl in (slice(0, 3), slice(3, 6))
    )
    for arm, sign in (("lead", -1.0), ("trail", 1.0)):
        a = offsets(df, pot, t, x, v, new, arm, sign)
        b = offsets(df, pot, t, x, v, old, arm, sign)
        for i in range(a.shape[1]):
            assert ks_2samp(a[:, i], b[:, i]).pvalue > 1e-3, (arm, i)


@pytest.mark.parametrize("df_cls", DFS)
def test_arms_independent(df_cls: type) -> None:
    """Lead and trail are separate draws: no correlation between them."""
    pot, n = nfw(), 4000
    t, x, v = orbit(n, pot)
    df = df_cls()
    s = sample_releases(df, jr.key(0), pot, t, x, v, np.full(n, MSAT))
    lead, trail = (
        offsets(df, pot, t, x, v, s, a, g) for a, g in (("lead", -1.0), ("trail", 1.0))
    )
    for i in range(lead.shape[1]):
        if np.std(trail[:, i]) < 1e-8:  # e.g. Fardal's y position offset is 0
            continue
        assert abs(np.corrcoef(trail[:, i], lead[:, i])[0, 1]) < 0.1, i


@pytest.mark.parametrize("df_cls", DFS)
def test_draws_independent_of_parameters(df_cls: type) -> None:
    """The same key gives the same standardized draws at different hosts."""
    df, n = df_cls(), 50
    out = []
    for m in (1e12, 3e11):
        pot = nfw(m)
        t, x, v = orbit(n, nfw())  # one orbit, so only the host differs
        s = sample_releases(df, jr.key(7), pot, t, x, v, np.full(n, MSAT))
        out.append(
            [
                offsets(df, pot, t, x, v, s, a, g)
                for a, g in (("lead", -1.0), ("trail", 1.0))
            ]
        )
    for a, b in zip(*out, strict=True):
        # Fardal's velocity offsets scale as omega r_t exactly; Chen's
        # standardized Dv carries 1/sqrt(Dr/r_t), also parameter independent.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
