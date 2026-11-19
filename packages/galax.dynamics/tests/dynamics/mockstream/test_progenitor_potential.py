"""`MockStreamGenerator(progenitor_potential=...)`: particles feel the progenitor.

This is the default model of Chen et al. 2024 (arXiv:2408.01496, Sections 3-4):
particles move in the host potential plus the progenitor's potential, centred on
the progenitor as it orbits the host. The progenitor itself orbits in the host
alone. gala does the same with
``gala.dynamics.mockstream.MockStreamGenerator(df, H, progenitor_potential=...)``.
"""

import platform
import warnings
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jaxlib
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.fields as gf
import galax.potential as gp
from ..optional_deps import OptDeps

USYS = u.unitsystem("galactic")
DATA = Path(__file__).parent / "data"
DFS = {"fardal": gd.FardalStreamDF, "chen": gd.ChenStreamDF}
NO_PROG_MSG = r'"no progenitor" variant'

# The setup of `test_mockstreamgenerator.py`.
HOST = gp.NFWPotential(m=1e12, r_s=15.0, units=USYS)
PROG_W0 = gc.PhaseSpaceCoordinate(
    q=u.Q([30, 10, 20], "kpc"), p=u.Q([10, -150, -20], "km/s"), t=u.Q(0.0, "Myr")
)


def adaptive(tol: float, /) -> gd.integrate.Integrator:
    return gd.integrate.Integrator(
        gd.OrbitSolver(
            solver=dfx.Dopri8(scan_kind="bounded"),
            stepsize_controller=dfx.PIDController(rtol=tol, atol=tol),
            max_steps=2**16,
        )
    )


def leapfrog(dt: float, t_max: float, /) -> gd.integrate.Integrator:
    n = int(t_max / dt) + 16
    solver = gd.OrbitSolver(
        solver=gf.Leapfrog(),
        stepsize_controller=dfx.ConstantStepSize(),
        max_steps=n,
    )
    return gd.integrate.Integrator(solver, diffeq_kw={"dt0": dt, "max_steps": n})


def stream_w(mock: gd.MockStream, /) -> np.ndarray:
    """(2 n, 6): lead rows then trail rows, in the host units."""
    return np.concatenate(
        [np.asarray(mock[k].w(units=USYS)) for k in ("lead", "trail")]
    )


# ============================================================================


@pytest.mark.parametrize("vmapped", [False, True], ids=["scan", "vmap"])
@pytest.mark.parametrize("dfname", ["fardal", "chen"])
@pytest.mark.filterwarnings(f"ignore:.*{NO_PROG_MSG}")
def test_no_progenitor_unchanged(dfname: str, vmapped: bool) -> None:
    """``progenitor_potential=None`` gives what galax gave before it existed.

    The reference is from galax main at 019696b3, the last commit before
    ``progenitor_potential`` existed, with the stream DFs of #1050 (one draw per
    arm), run with ``JAX_ENABLE_X64=True`` and diffrax 0.7.2, with the default
    adaptive solver and with fixed-step Leapfrog. diffrax 0.7.2 moves the
    Leapfrog streams by about 2e-8 of their scale relative to 0.7.0, so the
    reference has to be regenerated when the diffrax pin changes how Leapfrog
    steps. XLA's float results depend on the platform and the jax version, so
    where those match the reference's both are compared bitwise. Elsewhere the
    last-bit differences can change the adaptive solver's step choices, so only
    Leapfrog is compared, to 1e-12 of the position and velocity scales.
    """
    ref = np.load(DATA / "no_progenitor_reference.npz")
    same_build = (
        str(ref["platform"]) == f"{platform.system()}-{platform.machine()}"
        and str(ref["jax"]) == jax.__version__
        and str(ref["jaxlib"]) == jaxlib.__version__
    )
    lf = leapfrog(0.5, 4e3)
    runs = {"lf_": {"progenitor_integrator": lf, "stream_integrator": lf}}
    if same_build:
        runs[""] = {}
    ts = u.Q(jnp.linspace(0.0, 4e3, 10), "Myr")
    k = f"{dfname}_{'vmap' if vmapped else 'scan'}"
    for prefix, integrators in runs.items():
        gen = gd.MockStreamGenerator(
            DFS[dfname](), HOST, progenitor_potential=None, **integrators
        )
        mock, prog = gen.run(
            jax.random.key(12), ts, PROG_W0, u.Q(1e4, "Msun"), vmapped=vmapped
        )
        got = {
            "lead": np.asarray(mock["lead"].w(units=USYS)),
            "trail": np.asarray(mock["trail"].w(units=USYS)),
            "prog": np.asarray(prog.w(units=USYS)),
        }
        for part, arr in got.items():
            want = ref[f"{prefix}{k}_{part}"]
            if same_build:
                np.testing.assert_array_equal(arr, want)
                continue
            d = np.abs(arr - want).reshape(-1, 6)
            scale = np.abs(want).reshape(-1, 6)
            for name, c in (("q", slice(0, 3)), ("p", slice(3, 6))):
                assert d[:, c].max() <= 1e-12 * scale[:, c].max(), (
                    f"{prefix}{k}_{part} {name}: max diff {d[:, c].max():.2e}, "
                    f"scale {scale[:, c].max():.2e}"
                )


# ============================================================================
# Cross-check against gala


def gala_stream(
    host: gp.AbstractPotential,
    prog_pot: gp.AbstractPotential,
    prog_w_t0: np.ndarray,
    ts: np.ndarray,
    w0: np.ndarray,
    t_f: float,
    /,
    *,
    tol: float = 1e-10,
) -> np.ndarray:
    """Gala's mockstream integration of the particles ``w0`` (lead, trail per ts[i]).

    Calls gala's DOP853 kernel directly so it starts from galax's sampled
    initial conditions and ends at galax's final time.
    """
    import astropy.units as apyu
    import gala.dynamics as gad
    from gala.dynamics.mockstream import mockstream_dop853
    from gala.dynamics.nbody import DirectNBody
    from gala.units import galactic

    prog = gad.PhaseSpacePosition(
        pos=np.array(prog_w_t0[:3]) * apyu.kpc,
        vel=np.array(prog_w_t0[3:]) * apyu.kpc / apyu.Myr,
    )
    nbody = DirectNBody(
        prog, [to_gala(prog_pot)], external_potential=to_gala(host), units=galactic
    )
    # gala's kernel takes writable buffers; arrays from JAX are read-only.
    ts, w0 = np.array(ts, dtype=float), np.array(w0, dtype=float)
    n = len(ts)
    # gala wants the particles released at each time to be consecutive.
    w0_gala = np.ascontiguousarray(np.stack([w0[:n], w0[n:]], axis=1).reshape(-1, 6))
    _, raw = mockstream_dop853(
        nbody,
        np.ascontiguousarray(ts),
        w0_gala,
        np.ascontiguousarray(ts),
        t_f,
        np.full(n, 2, dtype="i4"),
        atol=tol,
        rtol=tol,
    )
    raw = raw.reshape(n, 2, 6)
    return np.concatenate([raw[:, 0], raw[:, 1]])


def to_gala(pot: gp.AbstractPotential, /) -> object:
    return gp.io.convert_potential(gp.io.GalaLibrary, pot)


PROG_POTS = {
    "kepler": lambda m: gp.KeplerPotential(m_tot=m, units=USYS),
    "plummer": lambda m: gp.PlummerPotential(m_tot=m, r_s=4e-3, units=USYS),
}


@pytest.mark.slow
@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.parametrize("progname", ["kepler", "plummer"])
@pytest.mark.parametrize("dfname", ["fardal", "chen"])
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_matches_gala_progenitor(dfname: str, progname: str) -> None:
    """Particles match gala's mockstream with the same progenitor potential.

    NFW 1e12 Msun / 15 kpc host, a 1e5 Msun cluster, 3 Gyr, 200 releases. Both
    codes start from galax's sampled initial conditions, so RNG differences are
    out of the comparison. Both run DOP853 at rtol = atol = 1e-12: at 1e-10 gala's
    own error reaches 5e-6 kpc, over the 1e-6 bound.
    """
    m = u.Q(1e5, "Msun")
    prog_pot = PROG_POTS[progname](m)
    ts = u.Q(jnp.linspace(-3000.0, 0.0, 200), "Myr")
    key = jax.random.key(3)
    df = DFS[dfname]()

    # The progenitor orbit and the particles' initial conditions galax uses.
    integ = adaptive(1e-12)
    prog_o = gd.evaluate_orbit(HOST, PROG_W0, ts, integrator=integ)
    mock0 = df.sample(key, HOST, prog_o, m)
    w0 = np.concatenate([np.asarray(mock0[k].w(units=USYS)) for k in ("lead", "trail")])
    t_f = float(ts[-1].value) + 1e-3  # galax's final time (`MockStreamGenerator`)
    ref = gala_stream(
        HOST,
        prog_pot,
        np.asarray(prog_o[0].w(units=USYS)),
        np.asarray(ts.value),
        w0,
        t_f,
        tol=1e-12,
    )

    gen = gd.MockStreamGenerator(
        df, HOST, progenitor_potential=prog_pot,
        progenitor_integrator=integ, stream_integrator=integ,
    )  # fmt: skip
    mock, _ = gen.run(key, ts, PROG_W0, m)
    got = stream_w(mock)
    # The same release times went in (the positions are checked by `d` below).
    np.testing.assert_array_equal(
        np.asarray(mock["lead"].release_time.value),
        np.asarray(mock0["lead"].release_time.value),
    )
    d = np.linalg.norm(got[:, :3] - ref[:, :3], axis=1)
    # A few Fardal particles (4-5 of 400, released near -1600 Myr) are on
    # chaotic orbits: galax alone moves them by up to 0.16 kpc between 1e-10
    # and 1e-12. So Fardal is bounded by its 95th percentile, Chen by its max.
    worst = d.max() if dfname == "chen" else np.percentile(d, 95)

    # It must change something: compare with no progenitor.
    gen0 = gd.MockStreamGenerator(
        df, HOST, progenitor_integrator=integ, stream_integrator=integ
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d0 = np.linalg.norm(
            stream_w(gen0.run(key, ts, PROG_W0, m)[0])[:, :3] - ref[:, :3], axis=1
        )
    assert np.median(d0) > 100 * worst

    lf = leapfrog(0.1, 3000.0)
    gen_lf = gd.MockStreamGenerator(
        df, HOST, progenitor_potential=prog_pot,
        progenitor_integrator=lf, stream_integrator=lf,
    )  # fmt: skip
    got_lf = stream_w(gen_lf.run(key, ts, PROG_W0, m, vmapped=True)[0])
    d_lf = np.linalg.norm(got_lf[:, :3] - ref[:, :3], axis=1)
    assert np.median(d_lf) <= 0.01, f"leapfrog: median {np.median(d_lf):.2e} kpc"

    assert worst <= 1e-6, (
        f"adaptive: median {np.median(d):.2e}, p95 {np.percentile(d, 95):.2e}, "
        f"max {d.max():.2e} kpc from gala; {(d > 1e-6).sum()}/{len(d)} over 1e-6"
    )


# ============================================================================


def test_field_accelerations() -> None:
    """The particle feels the host and the progenitor; the progenitor the host."""
    from galax.dynamics._src.legacy.mockstream.mockstream_generator import (
        _ProgenitorField,
    )

    prog = gp.PlummerPotential(m_tot=1e7, r_s=0.01, units=USYS)
    field = _ProgenitorField(HOST, prog)
    x = jnp.array([[30.0, 10.0, 20.0], [30.1, 9.9, 20.05]])
    t = jnp.asarray(0.0)
    got = field.dv_dt(t, x, None)
    want_particle = -HOST._gradient(x[0], t) - prog._gradient(x[0] - x[1], t)
    np.testing.assert_allclose(got[0], want_particle, rtol=1e-14)
    np.testing.assert_allclose(got[1], -HOST._gradient(x[1], t), rtol=1e-14)


@pytest.mark.parametrize(
    ("integ", "t_release"),
    [(adaptive(1e-12), -1234.5), (leapfrog(0.5, 3000.0), -1234.5)],
    ids=["adaptive", "leapfrog-on-grid"],
)
def test_progenitor_copy_follows_its_orbit(
    integ: gd.integrate.Integrator, t_release: float
) -> None:
    """Each particle's copy of the progenitor stays on the progenitor's orbit.

    The copy starts from the progenitor's state at the release time, partway
    along the orbit, and is integrated with the particle to the final time.
    With Leapfrog the release time is a whole number of steps after the orbit's
    start, so the copy takes the progenitor's own steps and only round-off
    separates them. (Off that grid, or with another progenitor integrator, the
    copy drifts by Leapfrog's error; see `MockStreamGenerator.progenitor_potential`.)
    """
    from galax.dynamics._src.legacy.mockstream.mockstream_generator import (
        _ProgenitorField,
    )

    prog = gp.KeplerPotential(m_tot=u.Q(1e5, "Msun"), units=USYS)
    t0, t1 = u.Q(t_release, "Myr"), u.Q(0.0, "Myr")
    orbit = gd.evaluate_orbit(
        HOST, PROG_W0, u.Q([-3000.0, t_release, 0.0], "Myr"), integrator=integ
    )
    wp = orbit.w(units=USYS)[-2:]
    particle = wp[0] + jnp.array([0.05, 0.0, 0.0, 0.0, 0.001, 0.0])
    y0 = jnp.stack([particle, wp[0]])
    out = integ(_ProgenitorField(HOST, prog), y0, t0, t1, saveat=jnp.asarray([t1]))
    got = np.asarray(out.w(units=USYS)).reshape(2, 6)[1]
    np.testing.assert_allclose(got[:3], wp[1, :3], rtol=0, atol=1e-8)


def test_progenitor_units_must_match() -> None:
    """A progenitor potential in other units raises in `run`, as on `main`.

    Its forces are added to the host's as raw arrays, so other units would be
    silently wrong. Also when it is swapped in with `eqx.tree_at`, which skips
    the generator's own checks.
    """
    pc = u.unitsystem("pc", "Msun", "Myr", "radian")
    pp = gp.PlummerPotential(m_tot=u.Q(1e6, "Msun"), r_s=u.Q(4, "pc"), units=pc)
    ok = gp.PlummerPotential(m_tot=u.Q(1e6, "Msun"), r_s=u.Q(4, "pc"), units=USYS)
    gen = gd.MockStreamGenerator(gd.ChenStreamDF(), HOST, progenitor_potential=ok)
    ts = u.Q(jnp.linspace(-100.0, 0.0, 3), "Myr")
    for bad in (
        gd.MockStreamGenerator(gd.ChenStreamDF(), HOST, progenitor_potential=pp),
        eqx.tree_at(lambda g: g.progenitor_potential, gen, pp),
    ):
        with pytest.raises(ValueError, match="same unit system"):
            bad.run(jax.random.key(0), ts, PROG_W0, u.Q(1e6, "Msun"))


def test_zero_mass_limit() -> None:
    """A vanishing progenitor mass gives the stream without a progenitor.

    And a real one changes it, so the progenitor isn't silently ignored.

    Fixed-step Leapfrog, so both runs take the same steps. With an adaptive
    solver the 1e-10 Msun force, tiny as it is, can change the step
    controller's choices: 7e-10 kpc on 3 of 300 coordinates at the default
    rtol = atol = 1e-7. An exactly zero mass is bitwise equal either way.
    """
    ts = u.Q(jnp.linspace(-3000.0, 0.0, 50), "Myr")
    key, m = jax.random.key(5), u.Q(1e5, "Msun")
    df = gd.FardalStreamDF()
    lf = leapfrog(1.0, 3000.0)
    kw = {"progenitor_integrator": lf, "stream_integrator": lf}
    tiny = gp.KeplerPotential(m_tot=u.Q(1e-10, "Msun"), units=USYS)
    gen_tiny = gd.MockStreamGenerator(df, HOST, progenitor_potential=tiny, **kw)
    w_tiny = stream_w(gen_tiny.run(key, ts, PROG_W0, m)[0])
    w_none = stream_w(
        gd.MockStreamGenerator(df, HOST, **kw).run(key, ts, PROG_W0, m)[0]
    )
    np.testing.assert_allclose(w_tiny[:, :3], w_none[:, :3], rtol=0, atol=1e-10)

    real = gp.KeplerPotential(m_tot=m, units=USYS)
    gen_real = gd.MockStreamGenerator(df, HOST, progenitor_potential=real, **kw)
    w_real = stream_w(gen_real.run(key, ts, PROG_W0, m)[0])
    d = np.linalg.norm(w_real[:, :3] - w_none[:, :3], axis=1)
    assert np.median(d) > 0.01, np.median(d)


@pytest.mark.parametrize("m", [1e5, 1e8], ids=["gc", "dwarf"])
def test_progenitor_close_passage_finite(m: float) -> None:
    """A point-mass progenitor is stiff near its centre: Leapfrog at 0.5 Myr copes.

    Every particle is finite, and fewer than 1% sit within 0.1 tidal radii of
    the progenitor today (a step too long for the close passages would pile
    them up there).
    """
    mass = u.Q(m, "Msun")
    ts = u.Q(jnp.linspace(-3000.0, 0.0, 500), "Myr")
    lf = leapfrog(0.5, 3000.0)
    gen = gd.MockStreamGenerator(
        gd.ChenStreamDF(), HOST,
        progenitor_potential=gp.KeplerPotential(m_tot=mass, units=USYS),
        progenitor_integrator=lf, stream_integrator=lf,
    )  # fmt: skip
    mock, prog = gen.run(jax.random.key(7), ts, PROG_W0, mass, vmapped=True)
    w = stream_w(mock)
    assert np.all(np.isfinite(w))

    r_tid = gd.tidal_radius(HOST, prog.q, prog.p, mass=mass, t=prog.t)
    r_tid = float(u.ustrip("kpc", r_tid))
    d = np.linalg.norm(w[:, :3] - np.asarray(prog.w(units=USYS))[:3], axis=1)
    assert np.mean(d < 0.1 * r_tid) < 0.01


def test_batch_mixed_t_age_progenitor() -> None:
    """``jit(vmap)`` over t_age and the progenitor mass equals separate calls.

    Batching changes the progenitor solve at the 1e-12 level, and a particle
    that passes close to a 1e7 Msun point mass can amplify that by orders of
    magnitude (7e-10 of the scale for one particle with the current draws and
    diffrax 0.7.2; the medians are below 3e-15). So the median is held to 1e-12
    of the scale and the worst value to 1e-6; mixed-up lanes would be off by
    order 1.
    """
    t_ages = jnp.asarray([3000.0, 4500.0, 7000.0])
    masses = jnp.asarray([1e5, 1e6, 1e7])
    lf = leapfrog(1.0, 7000.0)
    w_today = jnp.asarray(PROG_W0.w(units=USYS))
    key = jax.random.key(11)

    def one(t_age: jax.Array, m: jax.Array, /) -> tuple[jax.Array, jax.Array]:
        gen = gd.MockStreamGenerator(
            gd.ChenStreamDF(), HOST,
            progenitor_potential=gp.KeplerPotential(m_tot=u.Q(m, "Msun"), units=USYS),
            progenitor_integrator=lf, stream_integrator=lf,
        )  # fmt: skip
        ts = u.Q(jnp.linspace(-t_age, 0.0, 64), "Myr")
        w0 = gc.PhaseSpaceCoordinate(
            q=u.Q(w_today[:3], "kpc"), p=u.Q(w_today[3:], "kpc/Myr"), t=u.Q(0.0, "Myr")
        )
        mock, prog = gen.run(key, ts, w0, u.Q(m, "Msun"), vmapped=True)
        w = jnp.concatenate([mock[k].w(units=USYS) for k in ("lead", "trail")])
        return w, prog.w(units=USYS)

    batch = jax.jit(jax.vmap(one))(t_ages, masses)
    for i in range(len(t_ages)):
        single = jax.jit(one)(t_ages[i], masses[i])
        for b, s in zip(batch, single, strict=True):
            d = np.abs(np.asarray(b[i]) - np.asarray(s))
            scale = float(jnp.max(jnp.abs(s)))
            assert np.median(d) <= 1e-12 * scale, (i, np.median(d) / scale)
            assert d.max() <= 1e-6 * scale, (i, d.max() / scale)
    # The lanes are different streams.
    assert float(jnp.max(jnp.abs(batch[0][0] - batch[0][2]))) > 1.0


def test_chen_warning() -> None:
    """Chen+24 without a progenitor potential warns once, at ``run``.

    That is the "no progenitor" variant of the model. With a progenitor
    potential it is the default model, and nothing warns. Constructing the DF
    doesn't warn either.
    """
    ts = u.Q(jnp.linspace(-1000.0, 0.0, 7), "Myr")
    key, m = jax.random.key(0), u.Q(1e5, "Msun")
    gd.MockStreamGenerator.run.clear_cache()  # the warning is raised at trace time

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        df = gd.ChenStreamDF()
        assert not rec

        gen = gd.MockStreamGenerator(df, HOST)
        gen.run(key, ts, PROG_W0, m)
        gen.run(key, ts, PROG_W0, m)
        assert len(rec) == 1
        assert issubclass(rec[0].category, RuntimeWarning)
        assert "no progenitor" in str(rec[0].message)

        rec.clear()
        prog_pot = gp.KeplerPotential(m_tot=m, units=USYS)
        gd.MockStreamGenerator(df, HOST, progenitor_potential=prog_pot).run(
            key, ts, PROG_W0, m
        )
        assert not rec
