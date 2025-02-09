"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from abc import ABCMeta, abstractmethod

import jax.random as jr
import jax.tree as jtu
import pytest
from jaxtyping import PRNGKeyArray

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from galax.dynamics import (
    AbstractStreamDF,
    ChenStreamDF,
    FardalStreamDF,
    MockStreamGenerator,
)
from galax.potential import AbstractPotential, NFWPotential


class MockStreamGeneratorBase_Test(metaclass=ABCMeta):
    """Test the MockStreamGenerator class."""

    @pytest.fixture
    @abstractmethod
    def df(self) -> AbstractStreamDF: ...

    @pytest.fixture
    def pot(self) -> NFWPotential:
        """Mock stream DF."""
        return NFWPotential(
            m=u.Quantity(1.0e12, "Msun"), r_s=u.Quantity(15.0, "kpc"), units="galactic"
        )

    @pytest.fixture
    def mockgen(
        self, df: AbstractStreamDF, pot: AbstractPotential
    ) -> MockStreamGenerator:
        """Mock stream generator."""
        # TODO: test the progenitor integrator
        # TODO: test the stream integrator
        return MockStreamGenerator(df, pot)

    # ----------------------------------------

    @pytest.fixture
    def t_stripping(self) -> gt.QuSzTime:
        """Time vector for stripping."""
        return u.Quantity(jnp.linspace(0.0, 4e3, 8_000, dtype=float), "Myr")

    @pytest.fixture
    def prog_w0(self) -> gc.PhaseSpaceCoordinate:
        """Progenitor initial conditions."""
        return gc.PhaseSpaceCoordinate(
            q=u.Quantity([30, 10, 20], "kpc"),
            p=u.Quantity([10, -150, -20], "km/s"),
            t=u.Quantity(0.0, "Myr"),
        )

    @pytest.fixture
    def prog_mass(self) -> gt.MassSz0:
        """Progenitor mass."""
        return u.Quantity(1e4, "Msun")

    @pytest.fixture
    def rng(self) -> PRNGKeyArray:
        """Seed number for the random number generator."""
        return jr.key(12)

    @pytest.fixture
    def vmapped(self) -> bool:
        """Whether to use `jax.vmap`."""
        return False  # TODO: test both True and False

    # ========================================

    def test_run_scan(
        self,
        mockgen: MockStreamGenerator,
        t_stripping: gt.QuSzTime,
        prog_w0: gc.PhaseSpaceCoordinate,
        prog_mass: gt.MassSz0,
        rng: PRNGKeyArray,
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
            jtu.flatten(jtu.map(lambda x: jnp.isfinite(x).all(), x))[0]
        )
        assert allfinite(mock.q)
        assert allfinite(mock.p)
        assert jnp.isfinite(mock.t).all()


class TestFardalMockStreamGenerator(MockStreamGeneratorBase_Test):
    """Test the MockStreamGenerator class with FardalStreamDF."""

    @pytest.fixture
    def df(self) -> AbstractStreamDF:
        """Mock stream DF."""
        return FardalStreamDF()


class TestChenMockStreamGenerator(MockStreamGeneratorBase_Test):
    """Test the MockStreamGenerator class with ChenStreamDF."""

    @pytest.fixture
    def df(self) -> AbstractStreamDF:
        """Mock stream DF."""
        with pytest.warns(RuntimeWarning):
            return ChenStreamDF()
