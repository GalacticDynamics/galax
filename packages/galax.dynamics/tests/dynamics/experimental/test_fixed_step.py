"""Tests for the fixed-time-step stream generators."""

import math
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics import experimental
from galax.dynamics._src.experimental.fixed_step import (
    _leapfrog,
    _release_steps,
)

PROG_W0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])  # kpc, kpc/Myr
PROG_MASS = 1e7  # Msun


def nfw(m=1e12, r_s=15.0):
    return gp.NFWPotential(m=m, r_s=r_s, units="galactic")


def make_df(name):
    if name == "fardal":
        return gd.FardalStreamDF()
    with warnings.catch_warnings():  # "only the no-progenitor version" warning
        warnings.simplefilter("ignore", RuntimeWarning)
        return gd.ChenStreamDF()


def msg_stream(pot, df, key, ts):
    """`MockStreamGenerator` positions (2N, 3), lead then trail."""
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(PROG_W0[:3], "kpc"), p=u.Q(PROG_W0[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
    )
    stream, _ = gd.MockStreamGenerator(df, pot).run(
        key, u.Q(ts, "Myr"), w0, u.Q(PROG_MASS, "Msun")
    )
    arms = [stream["lead"], stream["trail"]]
    return jnp.concat(
        [jnp.stack([u.ustrip("kpc", getattr(a.q, c)) for c in "xyz"], -1) for a in arms]
    )


@pytest.mark.parametrize("df_name", ["fardal", "chen"])
def test_matches_mockstreamgenerator(df_name):
    """Same DF, release times and key as `MockStreamGenerator`: median < 10 pc."""
    pot, df, key = nfw(), make_df(df_name), jr.key(0)
    t_age, dt, n_release = 3000.0, 1.0, 1001  # a release every 3 steps
    stream_w, _ = experimental.fixed_step_stream(
        pot, df, key, PROG_W0, PROG_MASS, t_age,
        dt=dt, n_release=n_release, t_age_max=t_age,
    )  # fmt: skip
    ts = jnp.linspace(-t_age, 0.0, n_release)  # on the grid: 3000 / (1001 - 1) = 3
    ref = msg_stream(pot, df, key, ts)
    d = jnp.linalg.norm(stream_w[:, :3] - ref, axis=1)
    assert jnp.median(d) < 0.01
    assert jnp.max(d) < 0.05  # measured max 9.5 pc (Fardal), 2.8 pc (Chen)


def test_batch_mixed_t_age():
    """One vmap batch with different t_age gives the same streams as separate calls."""
    pot, df, key = nfw(), make_df("fardal"), jr.key(1)
    kw = {"dt": 1.0, "n_release": 64, "t_age_max": 7000.0}
    t_ages = jnp.array([3000.0, 4500.0, 7000.0])

    batched = jax.jit(
        jax.vmap(
            lambda t: experimental.fixed_step_stream(
                pot, df, key, PROG_W0, PROG_MASS, t, **kw
            )
        )
    )(t_ages)
    for i, t in enumerate(t_ages):
        single = experimental.fixed_step_stream(
            pot, df, key, PROG_W0, PROG_MASS, t, **kw
        )
        # To 1e-12 of the stream's coordinate scale. Elementwise, near-zero
        # components differ by ~5e-12 kpc after 7000 steps: vmap reorders the
        # floating-point sums. A wrong t_age handling would differ by kpc.
        for got, want in zip((batched[0][i], batched[1][i]), single, strict=True):
            assert jnp.max(jnp.abs(got - want)) <= 1e-12 * jnp.max(jnp.abs(want))


@pytest.mark.parametrize("df_name", ["fardal", "chen"])
def test_reproducible(df_name):
    """The same key gives the same stream, inside and outside `jit`."""
    pot, df = nfw(), make_df(df_name)
    kw = {"dt": 1.0, "n_release": 32, "t_age_max": 1000.0}

    def run(pot, key):
        return experimental.fixed_step_stream(
            pot, df, key, PROG_W0, PROG_MASS, 800.0, **kw
        )

    a, b = run(pot, jr.key(2)), run(pot, jr.key(2))
    c = jax.jit(run)(pot, jr.key(2))
    for x, y in ((a, b), (a, c)):
        assert jnp.array_equal(x[0], y[0])
        assert jnp.array_equal(x[1], y[1])
    assert not jnp.allclose(a[0], run(pot, jr.key(3))[0])


def test_vmap_over_potential():
    """`jit(vmap)` over potential parameters and progenitor ICs (X-Stream's use)."""
    df = make_df("fardal")
    ms = jnp.array([5e11, 1e12, 2e12, 4e12])
    w0s = PROG_W0 * jnp.array([1.0, 1.0, 1.0, 1.1])[:, None]

    @jax.jit
    @jax.vmap
    def run(m, w0, key):
        return experimental.fixed_step_stream(
            nfw(m=m), df, key, w0, PROG_MASS, 2000.0,
            dt=1.0, n_release=16, t_age_max=3000.0,
        )  # fmt: skip

    stream_w, prog_w = run(ms, w0s, jr.split(jr.key(4), 4))
    assert stream_w.shape == (4, 32, 6)
    assert prog_w.shape == (4, 6)
    assert jnp.all(jnp.isfinite(stream_w))
    # the progenitor comes back to where it started (round trip on the grid)
    assert jnp.allclose(prog_w, w0s, atol=1e-9)


def test_leapfrog_round_trip():
    """The shared KDK kernel run forward then back returns the initial state."""
    pot = nfw()

    def acc(q, t):
        return -pot._gradient(q, t)

    q0, p0 = PROG_W0[None, :3], PROG_W0[None, 3:]
    q1, p1, a1 = _leapfrog(acc, (q0, p0, acc(q0, 0.0)), 0.0, 1.0, 1000)
    q2, p2, _ = _leapfrog(acc, (q1, p1, a1), 1000.0, -1.0, 1000)
    assert jnp.max(jnp.abs(q2 - q0)) <= 1e-10
    assert jnp.max(jnp.abs(p2 - p0)) <= 1e-12


def td_nfw():
    """NFW whose mass grows by 2e9 Msun/Myr: makes every force time matter."""
    m = gp.params.LinearParameter(
        slope=u.Q(2e9, "Msun/Myr"),
        point_time=u.Q(0, "Myr"),
        point_value=u.Q(1e12, "Msun"),
    )
    return gp.NFWPotential(m=m, r_s=15.0, units="galactic")


def per_particle_reference(pot, df, key, t_age, dt, n_release, t_age_max, prog_mass):
    """Each particle integrated on its own from its release step (plain KDK)."""
    n = math.ceil(t_age_max / dt)
    h, t0 = t_age / n, -t_age

    def kdk(q, p, t, hh):
        p = p - 0.5 * hh * pot._gradient(q, t)
        q = q + hh * p
        return q, p - 0.5 * hh * pot._gradient(q, t + hh)

    qs, ps = [PROG_W0[:3]], [PROG_W0[3:]]  # back from t = 0, at t = -j h
    for j in range(n):
        q, p = kdk(qs[-1], ps[-1], -j * h, -h)
        qs.append(q)
        ps.append(p)
    qs, ps = jnp.stack(qs[::-1]), jnp.stack(ps[::-1])  # step k: t = t0 + k h

    k_rel = _release_steps(n, n_release)
    x_l, v_l, x_t, v_t = df._sample(
        key,
        pot,
        u.Q(qs[k_rel], "kpc"),
        u.Q(ps[k_rel], "kpc/Myr"),
        u.Q(jnp.full(n_release, prog_mass), "Msun"),
        u.Q(t0 + k_rel * h, "Myr"),
    )

    def forward(q, p, k):
        def body(i, qp):
            return kdk(*qp, t0 + i * h, h)

        return jax.lax.fori_loop(k, n, body, (q, p))

    q, p = jax.vmap(forward)(
        jnp.concat([u.ustrip("kpc", x_l), u.ustrip("kpc", x_t)]),
        jnp.concat([u.ustrip("kpc/Myr", v_l), u.ustrip("kpc/Myr", v_t)]),
        jnp.asarray(np.concat([k_rel, k_rel])),
    )
    return jnp.concat([q, p], axis=1)


@pytest.mark.parametrize("df_name", ["fardal", "chen"])
@pytest.mark.parametrize(
    ("t_age", "dt", "n_release", "t_age_max"),
    [
        (37.3, 1.0, 9, 200.0),  # t_age < t_age_max, not a multiple of dt
        (50.0, 1.0, 157, 50.0),  # more releases than steps: shared release steps
        (150.5, 0.7, 13, 160.0),  # t_age_max / dt not a whole number of steps
    ],
)
@pytest.mark.parametrize("prog_mass", [PROG_MASS, 1e5])
def test_matches_per_particle_reference(
    df_name, t_age, dt, n_release, t_age_max, prog_mass
):
    """Time-dependent host: release times, force times and holds all line up."""
    pot, df, key = td_nfw(), make_df(df_name), jr.key(7)
    got, _ = experimental.fixed_step_stream(
        pot, df, key, PROG_W0, prog_mass, t_age,
        dt=dt, n_release=n_release, t_age_max=t_age_max,
    )  # fmt: skip
    want = per_particle_reference(
        pot, df, key, t_age, dt, n_release, t_age_max, prog_mass
    )
    assert jnp.max(jnp.abs(got - want)) <= 1e-12


@pytest.mark.parametrize("t_age", [-50.0, 450.0, float("nan")])
def test_t_age_out_of_range_raises(t_age):
    """t_age outside [0, t_age_max] is an error, not a silently coarser step."""
    with pytest.raises(Exception, match="t_age must be in"):
        experimental.fixed_step_stream(
            nfw(), make_df("fardal"), jr.key(0), PROG_W0, PROG_MASS, t_age,
            dt=1.0, n_release=4, t_age_max=300.0,
        )  # fmt: skip


def test_release_steps():
    """Release steps run from the first grid step to the last, rounded."""
    assert _release_steps(10, 3).tolist() == [0, 5, 10]
    assert _release_steps(4, 7).tolist() == [0, 1, 1, 2, 3, 3, 4]  # shared steps


def test_float32_t_age_keeps_full_precision():
    """A float32 t_age gives the same stream as the same value in float64."""
    t32 = jnp.float32(1999.7)
    kw = {"dt": 1.0, "n_release": 8, "t_age_max": 2000.0}
    a = experimental.fixed_step_stream(
        nfw(), make_df("fardal"), jr.key(0), PROG_W0, PROG_MASS, t32, **kw
    )
    b = experimental.fixed_step_stream(
        nfw(), make_df("fardal"), jr.key(0), PROG_W0, PROG_MASS, float(t32), **kw
    )
    assert jnp.array_equal(a[0], b[0])


@pytest.mark.parametrize(("dt", "n_release"), [(0.0, 4), (-1.0, 4), (1.0, 0)])
def test_bad_static_args_raise(dt, n_release):
    """A non-positive dt or no releases is rejected up front."""
    with pytest.raises(ValueError, match="need dt > 0 and n_release >= 1"):
        experimental.fixed_step_stream(
            nfw(), make_df("fardal"), jr.key(0), PROG_W0, PROG_MASS, 100.0,
            dt=dt, n_release=n_release, t_age_max=300.0,
        )  # fmt: skip
