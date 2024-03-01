"""Test :mod:`galax.dynamics.mockstream.mockstreamgenerator`."""

from typing import cast

import array_api_jax_compat as xp
import astropy.units as u
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from galax.dynamics import AbstractStreamDF, FardalStreamDF, MockStreamGenerator
from galax.potential import AbstractPotentialBase, NFWPotential
from galax.typing import Vec6, VecTime
from galax.units import galactic


class TestMockStreamGenerator:
    """Test the MockStreamGenerator class."""

    @pytest.fixture()
    def df(self) -> AbstractStreamDF:
        """Mock stream DF."""
        return FardalStreamDF()

    @pytest.fixture()
    def pot(self) -> NFWPotential:
        """Mock stream DF."""
        return NFWPotential(
            m=1.0e12 * u.Msun, r_s=15.0 * u.kpc, softening_length=0.0, units=galactic
        )

    @pytest.fixture()
    def mockstream(
        self, df: AbstractStreamDF, pot: AbstractPotentialBase
    ) -> MockStreamGenerator:
        """Mock stream generator."""
        # TODO: test the progenitor integrator
        # TODO: test the stream integrator
        return MockStreamGenerator(df, pot)

    # ----------------------------------------

    @pytest.fixture()
    def t_stripping(self) -> VecTime:
        """Time vector for stripping."""
        return cast(VecTime, xp.linspace(0.0, 4e3, 8_000, dtype=float))

    @pytest.fixture()
    def prog_w0(self) -> Vec6:
        """Progenitor initial conditions."""
        return cast(
            Vec6,
            xp.asarray(
                [
                    x.decompose(galactic).value
                    for x in [*(30, 10, 20) * u.kpc, *(10, -150, -20) * u.km / u.s]
                ],
                dtype=float,
            ),
        )

    @pytest.fixture()
    def prog_mass(self) -> float:
        """Progenitor mass."""
        return 1e4

    @pytest.fixture()
    def seed_num(self) -> int:
        """Seed number for the random number generator."""
        return 12

    @pytest.fixture()
    def vmapped(self) -> bool:
        """Whether to use `jax.vmap`."""
        return False  # TODO: test both True and False

    # ========================================

    def test_run_scan(
        self,
        mockstream: MockStreamGenerator,
        t_stripping: VecTime,
        prog_w0: Vec6,
        prog_mass: float,
        seed_num: int,
    ) -> None:
        """Test the run method with ``vmapped=False``."""
        mock, prog_o = mockstream.run(
            t_stripping,
            prog_w0,
            prog_mass,
            seed_num=seed_num,
            vmapped=False,
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
