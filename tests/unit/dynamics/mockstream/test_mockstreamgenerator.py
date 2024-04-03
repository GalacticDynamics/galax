"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

import astropy.units as u
import jax.numpy as jnp
import jax.tree_util as tu
import pytest
import quax.examples.prng as jr

import quaxed.array_api as xp
from unxt import Quantity

import galax.coordinates as gc
import galax.typing as gt
from galax.dynamics import AbstractStreamDF, FardalStreamDF, MockStreamGenerator
from galax.potential import AbstractPotentialBase, NFWPotential


class TestMockStreamGenerator:
    """Test the MockStreamGenerator class."""

    @pytest.fixture()
    def df(self) -> AbstractStreamDF:
        """Mock stream DF."""
        return FardalStreamDF()

    @pytest.fixture()
    def pot(self) -> NFWPotential:
        """Mock stream DF."""
        return NFWPotential(m=1.0e12 * u.Msun, r_s=15.0 * u.kpc, units="galactic")

    @pytest.fixture()
    def mockgen(
        self, df: AbstractStreamDF, pot: AbstractPotentialBase
    ) -> MockStreamGenerator:
        """Mock stream generator."""
        # TODO: test the progenitor integrator
        # TODO: test the stream integrator
        return MockStreamGenerator(df, pot)

    # ----------------------------------------

    @pytest.fixture()
    def t_stripping(self) -> gt.QVecTime:
        """Time vector for stripping."""
        return Quantity(xp.linspace(0.0, 4e3, 8_000, dtype=float), "Myr")

    @pytest.fixture()
    def prog_w0(self) -> gc.PhaseSpacePosition:
        """Progenitor initial conditions."""
        return gc.PhaseSpacePosition(
            q=[30, 10, 20] * u.kpc, p=[10, -150, -20] * u.km / u.s, t=0.0 * u.Myr
        )

    @pytest.fixture()
    def prog_mass(self) -> gt.MassScalar:
        """Progenitor mass."""
        return Quantity(1e4, "Msun")

    @pytest.fixture()
    def rng(self) -> jr.PRNG:
        """Seed number for the random number generator."""
        return jr.ThreeFry(12)

    @pytest.fixture()
    def vmapped(self) -> bool:
        """Whether to use `jax.vmap`."""
        return False  # TODO: test both True and False

    # ========================================

    def test_run_scan(
        self,
        mockgen: MockStreamGenerator,
        t_stripping: gt.QVecTime,
        prog_w0: gc.PhaseSpacePosition,
        prog_mass: gt.MassScalar,
        rng: jr.PRNG,
        vmapped: bool,
    ) -> None:
        """Test the run method with ``vmapped=False``."""
        mock, prog_o = mockgen.run(
            rng, t_stripping, prog_w0, prog_mass, vmapped=vmapped
        )

        # TODO: more rigorous tests
        assert mock.q.shape == (2 * len(t_stripping),)
        assert prog_o.q.shape == ()  # scalar batch shape

        # Test that the positions and momenta are finite
        allfinite = lambda x: all(
            tu.tree_flatten(tu.tree_map(lambda x: jnp.isfinite(x).all(), x))[0]
        )
        assert allfinite(mock.q)
        assert allfinite(mock.p)
        assert xp.isfinite(mock.t).all()
