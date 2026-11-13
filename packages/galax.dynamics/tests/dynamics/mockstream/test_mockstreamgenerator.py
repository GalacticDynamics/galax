"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from abc import ABCMeta, abstractmethod

from jaxtyping import PRNGKeyArray

import diffrax as dfx
import jax.random as jr
import jax.tree as jtu
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.dynamics.custom_types as gt
import galax.potential as gp


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

    def test_run_scan(
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

        The reference integrates every particle on its own with
        `galax.dynamics.evaluate_orbit`, from its release time to the final
        time (which `MockStreamGenerator` bumps by 1e-3 Myr).

        Both use a fixed-step `Leapfrog`, so they take the same steps on any
        platform. Two adaptive solves would not: their accepted steps differ
        with rounding, by an amount that varies between machines. Every leg
        runs forward here (the stripping times start at the progenitor's time),
        so a constant step works without a sign-aware ``dt0``.
        """
        integrator = gd.integrate.Integrator(
            gd.OrbitSolver(
                gd.experimental.Leapfrog(),
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
                        u.Q(jnp.stack([t_stripping[i].value, t_f.value]), "Myr"),
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


class TestChenMockStreamGenerator(MockStreamGeneratorBase_Test):
    """Test the MockStreamGenerator class with ChenStreamDF."""

    @pytest.fixture
    def df(self) -> gd.AbstractStreamDF:
        """Mock stream DF."""
        with pytest.warns(RuntimeWarning):
            return gd.ChenStreamDF()


def test_integrate_to_solves_each_particle_on_its_own():
    """With an adaptive solver, each arm keeps its own step sizes.

    A leading and a trailing particle on very different orbits, integrated
    together by `MockStreamGenerator._integrate_to`, each equal a solve of that
    particle alone. One solve of both would share a step size and miss this.
    """
    pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    gen = gd.MockStreamGenerator(gd.FardalStreamDF(), pot)
    w0 = jnp.array(
        [[30.0, 10.0, 20.0, 0.01, -0.15, -0.02], [2.0, 0.0, 0.5, 0.0, 0.3, 0.05]]
    )
    t0, t1 = u.Q(0.0, "Myr"), u.Q(300.0, "Myr")
    got = gen._integrate_to(w0, t0, t1)

    field = gd.fields.HamiltonianField(pot)
    for i in range(2):
        want = gen.stream_integrator(field, w0[i], t0, t1, saveat=jnp.asarray([t1])).w(
            units=pot.units
        )[-1]
        # A vmapped and a single adaptive solve differ at ~1e-12 of the
        # coordinate scale; one solve of both particles is off by > 7e-9.
        assert jnp.max(jnp.abs(got[i] - want)) < 1e-10 * jnp.max(jnp.abs(want))
