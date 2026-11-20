"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

import functools as ft
from abc import ABCMeta, abstractmethod

from jaxtyping import PRNGKeyArray

import diffrax as dfx
import jax
import jax.random as jr
import jax.tree as jtu
import numpy as np
import pytest
from scipy.stats import ks_2samp

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.dynamics.custom_types as gt
import galax.fields as gf
import galax.potential as gp
from galax.dynamics._src.legacy.mockstream.mockstream_generator import (
    _ProgenitorField,
)


class MockStreamGeneratorBase_Test(metaclass=ABCMeta):
    """Test the MockStreamGenerator class."""

    @pytest.fixture
    @abstractmethod
    def df(self) -> gd.AbstractStreamDF: ...

    @pytest.fixture
    def pot(self) -> gp.NFWPotential:
        """Mock stream DF."""
        return gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")

    @pytest.fixture
    def mockgen(
        self, df: gd.AbstractStreamDF, pot: gp.AbstractPotential
    ) -> gd.MockStreamGenerator:
        """Mock stream generator."""
        # TODO: test the progenitor integrator
        # TODO: test the stream integrator
        return gd.MockStreamGenerator(df, pot)

    # ----------------------------------------

    @pytest.fixture
    def t_stripping(self) -> gt.QuSzTime:
        """Time vector for stripping."""
        return u.Q(jnp.linspace(0.0, 4e3, 10), "Myr")

    @pytest.fixture
    def prog_w0(self) -> gc.PhaseSpaceCoordinate:
        """Progenitor initial conditions."""
        return gc.PhaseSpaceCoordinate(
            q=u.Q([30, 10, 20], "kpc"),
            p=u.Q([10, -150, -20], "km/s"),
            t=u.Q(0.0, "Myr"),
        )

    @pytest.fixture
    def prog_mass(self) -> gt.QuSz0:
        """Progenitor mass."""
        return u.Q(1e4, "Msun")

    @pytest.fixture
    def rng(self) -> PRNGKeyArray:
        """Seed number for the random number generator."""
        return jr.key(12)

    @pytest.fixture(params=[False, True], ids=["scan", "vmap"])
    def vmapped(self, request: pytest.FixtureRequest) -> bool:
        """Whether to use `jax.vmap` (`True`) or `jax.lax.scan` (`False`)."""
        return request.param

    # ========================================

    def test_run(
        self,
        mockgen: gd.MockStreamGenerator,
        t_stripping: gt.QuSzTime,
        prog_w0: gc.PhaseSpaceCoordinate,
        prog_mass: gt.QuSz0,
        rng: PRNGKeyArray,
        vmapped: bool,
    ) -> None:
        """Test the run method, with `jax.lax.scan` and with `jax.vmap`."""
        mock, prog_o = mockgen.run(
            rng, t_stripping, prog_w0, prog_mass, vmapped=vmapped
        )

        # TODO: more rigorous tests
        assert mock.q.shape == (2 * len(t_stripping),)
        assert prog_o.q.shape == ()  # scalar batch shape

        # Test that the positions and momenta are finite
        allfinite = lambda x: all(
            jtu.flatten(jtu.map(lambda x: jnp.all(jnp.isfinite(x)), x))[0]
        )
        assert allfinite(mock.q)
        assert allfinite(mock.p)
        assert jnp.all(jnp.isfinite(mock.t))

    def test_run_independent_of_time_unit(
        self,
        mockgen: gd.MockStreamGenerator,
        t_stripping: gt.QuSzTime,
        prog_w0: gc.PhaseSpaceCoordinate,
        prog_mass: gt.QuSz0,
        rng: PRNGKeyArray,
        vmapped: bool,
    ) -> None:
        """The same stripping times in Myr and in Gyr give the same stream.

        The bump on the final time was `1e-3` in the unit of `ts`, which is
        1 Myr for `Gyr` and 1e-3 Myr for `Myr`.
        """
        t_gyr = t_stripping.uconvert("Gyr")
        mock_myr, _ = mockgen.run(rng, t_stripping, prog_w0, prog_mass, vmapped=vmapped)
        mock_gyr, _ = mockgen.run(rng, t_gyr, prog_w0, prog_mass, vmapped=vmapped)

        w_myr = mock_myr.w(units="galactic")
        w_gyr = mock_gyr.w(units="galactic")
        # The unit bug moved positions by ~0.2 kpc; this is round-off.
        assert jnp.max(jnp.abs(w_gyr - w_myr)) < 1e-6

    def test_particles_match_independent_orbits(
        self,
        df: gd.AbstractStreamDF,
        pot: gp.AbstractPotential,
        t_stripping: gt.QuSzTime,
        prog_w0: gc.PhaseSpaceCoordinate,
        prog_mass: gt.QuSz0,
        rng: PRNGKeyArray,
        vmapped: bool,
    ) -> None:
        """Each particle is its release state integrated to the final time.

        The reference runs `evaluate_orbit` per particle. Both use a fixed-step
        `Leapfrog`, so they take the same steps on any platform (two adaptive
        solves would not).
        """
        integrator = gd.integrate.Integrator(
            gd.OrbitSolver(
                gf.Leapfrog(),
                stepsize_controller=dfx.ConstantStepSize(),
                max_steps=5000,
            ),
            diffeq_kw={"dt0": 1.0, "max_steps": 5000},
        )
        mockgen = gd.MockStreamGenerator(
            df, pot, progenitor_integrator=integrator, stream_integrator=integrator
        )
        mock, _ = mockgen.run(rng, t_stripping, prog_w0, prog_mass, vmapped=vmapped)

        prog_orbit = gd.evaluate_orbit(pot, prog_w0, t_stripping, integrator=integrator)
        ics = df.sample(rng, pot, prog_orbit, prog_mass)
        t_f = t_stripping[-1] + u.Q(1e-3, "Myr")
        for arm in ("lead", "trail"):
            w0 = ics[arm].w(units=pot.units)
            want = jnp.stack(
                [
                    gd.evaluate_orbit(
                        pot,
                        w0[i],
                        jnp.stack([t_stripping[i], t_f]),
                        integrator=integrator,
                    ).w(units=pot.units)[-1]
                    for i in range(len(t_stripping))
                ]
            )
            got = mock[arm].w(units=pot.units)
            # Same steps, so only round-off separates them. A wrong release
            # time or arm is off by O(1).
            err = jnp.abs(got - want) / jnp.abs(want).max(0)
            assert jnp.max(err) < 1e-10


class TestFardalMockStreamGenerator(MockStreamGeneratorBase_Test):
    """Test the MockStreamGenerator class with FardalStreamDF."""

    @pytest.fixture
    def df(self) -> gd.AbstractStreamDF:
        """Mock stream DF."""
        return gd.FardalStreamDF()


@pytest.mark.filterwarnings('ignore:.*"no progenitor" variant')
class TestChenMockStreamGenerator(MockStreamGeneratorBase_Test):
    """Test the MockStreamGenerator class with ChenStreamDF."""

    @pytest.fixture
    def df(self) -> gd.AbstractStreamDF:
        """Mock stream DF."""
        return gd.ChenStreamDF()


@pytest.mark.parametrize("with_progenitor", [False, True], ids=["host", "progenitor"])
def test_integrate_to_solves_each_particle_on_its_own(with_progenitor: bool) -> None:
    """With an adaptive solver, each arm keeps its own step sizes.

    A leading and a trailing particle on very different orbits, integrated
    together by `MockStreamGenerator._integrate_to`, each equal a solve of that
    particle alone. One solve of both would share a step size and miss this.
    With a progenitor potential each particle is solved with its own copy of
    the progenitor, so the solo solve is of the pair [particle, progenitor].
    """
    pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    prog = gp.PlummerPotential(m_tot=1e6, r_s=0.004, units="galactic")
    gen = gd.MockStreamGenerator(
        gd.FardalStreamDF(), pot, progenitor_potential=prog if with_progenitor else None
    )
    w0 = jnp.array(
        [[30.0, 10.0, 20.0, 0.01, -0.15, -0.02], [2.0, 0.0, 0.5, 0.0, 0.3, 0.05]]
    )
    # 50 pc from the leading particle, so its pull changes that orbit.
    prog_w0 = (
        jnp.array([30.05, 10.0, 20.0, 0.01, -0.15, -0.02]) if with_progenitor else None
    )
    t0, t1 = u.Q(0.0, "Myr"), u.Q(300.0, "Myr")
    got = gen._integrate_to(w0, t0, t1, prog_w0)

    field: gd.fields.AbstractOrbitField
    if prog_w0 is None:
        field = gd.fields.HamiltonianField(pot)
    else:
        field = _ProgenitorField(pot, prog)
    for i in range(2):
        y0 = w0[i] if prog_w0 is None else jnp.stack([w0[i], prog_w0])
        out = gen.stream_integrator(field, y0, t0, t1, saveat=jnp.asarray([t1]))
        want = out.w(units=pot.units).reshape(-1, 6)[0]
        # Here ~1e-12 of the scale; one solve of both particles is off by > 7e-9.
        assert jnp.max(jnp.abs(got[i] - want)) < 1e-10 * jnp.max(jnp.abs(want))


@pytest.mark.parametrize("df_cls", [gd.ChenStreamDF, gd.FardalStreamDF])
def test_arms_drawn_independently(df_cls: type[gd.AbstractStreamDF]) -> None:
    """Leading and trailing particles are separate draws from the DF.

    In the progenitor's frame the leading arm is the trailing DF rotated by 180
    deg about L, so undoing that rotation gives two samples of the same DF. They
    must agree in distribution but not be the same draw (the Chen+24 authors'
    code, gala and galpy all draw each arm on its own).
    """
    pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km/s"), t=u.Q(0, "Myr")
    )
    n = 4000
    prog = gd.evaluate_orbit(pot, w0, u.Q(jnp.linspace(-3000.0, 0.0, n), "Myr"))
    mock = df_cls().sample(jr.key(0), pot, prog, u.Q(1e5, "Msun"))

    pw = np.asarray(prog.w(units=pot.units))
    x, v = pw[:, :3], pw[:, 3:]
    r = np.linalg.norm(x, axis=1, keepdims=True)
    xh = x / r
    zh = np.cross(x, v)
    omega = np.linalg.norm(zh, axis=1, keepdims=True) / r**2
    zh /= np.linalg.norm(zh, axis=1, keepdims=True)
    yh = np.cross(zh, xh)
    rt = np.asarray(
        u.ustrip(
            "kpc", gd.tidal_radius(pot, prog.q, prog.p, mass=u.Q(1e5, "Msun"), t=prog.t)
        )
    )[:, None]
    # Each DF's own velocity scale: v_esc(r_t) for Chen+24, omega r_t for Fardal+15.
    if df_cls is gd.ChenStreamDF:
        G = u.ustrip("kpc3 / (Msun Myr2)", pot.constants["G"])
        vscale = np.sqrt(2 * G * 1e5 / rt)
    else:
        vscale = omega * rt

    def offsets(arm: str, sign: float, /) -> np.ndarray:
        """Position and velocity offsets, in units of r_t and the DF's v scale.

        For each: the size, then x, y, z. x and y are rotated back for the lead
        (z is unchanged by the rotation).
        """
        w = np.asarray(mock[arm].w(units=pot.units))
        cols = []
        for d in ((w[:, :3] - x) / rt, (w[:, 3:] - v) / vscale):
            cols += [np.linalg.norm(d, axis=1), sign * (d * xh).sum(1)]
            cols += [sign * (d * yh).sum(1), (d * zh).sum(1)]
        return np.column_stack(cols)

    trail, lead = offsets("trail", 1.0), offsets("lead", -1.0)
    for i in range(trail.shape[1]):
        if np.std(trail[:, i]) < 1e-8:  # e.g. Fardal's y position offset is 0
            continue
        # Same DF.
        assert ks_2samp(trail[:, i], lead[:, i]).pvalue > 1e-3, i
        # Separate draws: one shared draw gives correlation 1; with n = 4000
        # independent draws |corr| ~ 0.016. (Scaled: the raw offsets of both
        # arms share the scale, which varies along the orbit.)
        assert abs(np.corrcoef(trail[:, i], lead[:, i])[0, 1]) < 0.1, i


def _leapfrog(adjoint: dfx.AbstractAdjoint, /) -> gd.integrate.Integrator:
    """Leapfrog at 1 Myr, bounded at 600 steps."""
    return gd.integrate.Integrator(
        gd.OrbitSolver(
            gf.Leapfrog(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=adjoint,
            max_steps=600,
        ),
        diffeq_kw={"dt0": 1.0, "max_steps": 600},
    )


@pytest.mark.filterwarnings('ignore:.*"no progenitor" variant')
@pytest.mark.parametrize("vmapped", [False, True], ids=["scan", "vmap"])
@pytest.mark.parametrize("with_progenitor", [False, True], ids=["host", "progenitor"])
@pytest.mark.parametrize("df", [gd.FardalStreamDF(), gd.ChenStreamDF()], ids=str)
def test_run_reverse_mode_gradient(
    df: gd.AbstractStreamDF, with_progenitor: bool, vmapped: bool
) -> None:
    """`jax.grad` through `run`, with `RecursiveCheckpointAdjoint`.

    The default `ForwardMode` adjoint solves in a `while` loop, which reverse
    mode can't differentiate. The gradient of the summed positions today, in
    the host's mass and the progenitor's initial conditions, matches central
    differences and `jax.jvp` with the default adjoint.
    """
    units = u.unitsystem("galactic")
    ts = u.Q(jnp.linspace(-500.0, 0.0, 10), "Myr")  # 20 particles

    def loss(x: gt.SzN, integrator: gd.integrate.Integrator) -> gt.Sz0:
        pot = gp.NFWPotential(m=1e12 * x[0], r_s=15.0, units=units)
        pp = gp.PlummerPotential(m_tot=1e5, r_s=4e-3, units=units)
        gen = gd.MockStreamGenerator(
            df,
            pot,
            progenitor_potential=pp if with_progenitor else None,
            progenitor_integrator=integrator,
            stream_integrator=integrator,
        )
        w0 = gc.PhaseSpaceCoordinate(
            q=u.Q(x[1:4], "kpc"), p=u.Q(x[4:], "kpc/Myr"), t=ts[0]
        )
        mock, _ = gen.run(jr.key(0), ts, w0, u.Q(1e5, "Msun"), vmapped=vmapped)
        return jnp.sum(mock.q.x.value + mock.q.y.value + mock.q.z.value)

    x = jnp.array([1.0, 20.0, 0.0, 5.0, 0.0, 0.2, 0.02])
    grad = jax.grad(loss)(x, _leapfrog(dfx.RecursiveCheckpointAdjoint()))

    forward = jax.jit(ft.partial(loss, integrator=_leapfrog(dfx.ForwardMode())))
    v = jr.normal(jr.key(1), x.shape)
    _, jvp = jax.jvp(forward, (x,), (v,))
    np.testing.assert_allclose(jvp, grad @ v, rtol=1e-10)

    h = 1e-5
    fd = jnp.stack(
        [(forward(x + h * e) - forward(x - h * e)) / (2 * h) for e in jnp.eye(7)]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
