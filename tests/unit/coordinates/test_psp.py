"""Test :class:`~galax.coordinates._psp`."""

from dataclasses import replace
from typing import Any, Self, TypeAlias

import astropy.units as u
import jax.experimental.array_api as xp
import jax.numpy as jnp
import jax.random as jr
import pytest

from galax.coordinates import AbstractPhaseSpacePosition, PhaseSpacePosition
from galax.potential import AbstractPotentialBase, KeplerPotential
from galax.potential._potential.special import MilkyWayPotential
from galax.typing import Vec3
from galax.units import galactic

from .test_base import AbstractPhaseSpacePositionBase_Test, T, return_keys

Shape: TypeAlias = tuple[int, ...]

potentials = [KeplerPotential(m=1e12 * u.Msun, units=galactic), MilkyWayPotential()]


class AbstractPhaseSpacePosition_Test(AbstractPhaseSpacePositionBase_Test[T]):
    """Test :class:`~galax.coordinates.AbstractPhaseSpacePosition`."""

    def make_w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = jr.normal(next(subkeys), (*shape, 3))
        p = jr.normal(next(subkeys), (*shape, 3))
        return w_cls(q, p)

    # ===============================================================

    # TODO: further tests for getitem
    # def test_getitem()

    # ===============================================================

    @pytest.mark.parametrize("potential", potentials)
    def potential_energy(self, w: T, potential: AbstractPotentialBase) -> None:
        """Test method ``potential_energy``."""
        pe = w.potential_energy(potential, t=0.0)
        assert pe.shape == w.shape  # confirm relation to shape and components
        assert xp.all(pe <= 0)
        # definitional
        assert jnp.array_equal(pe, potential.potential_energy(w.q, t=0.0))

    @pytest.mark.parametrize("potential", potentials)
    def energy(self, w: T, potential: AbstractPotentialBase) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.energy`."""
        pe = w.energy(potential, t=0.0)
        assert pe.shape == w.shape  # confirm relation to shape and components
        # definitional
        assert jnp.array_equal(
            pe, w.kinetic_energy() + potential.potential_energy(w.q, t=0.0)
        )


##############################################################################


class TestAbstractPhaseSpacePosition(
    AbstractPhaseSpacePosition_Test[AbstractPhaseSpacePosition]
):
    """Test :class:`~galax.coordinates.AbstractPhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""

        class PSP(AbstractPhaseSpacePosition):
            """A phase-space position."""

            q: Vec3
            p: Vec3

            @property
            def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
                return self.q.shape[:-1], (3, 3)

            def __getitem__(self, index: Any) -> Self:
                return replace(self, q=self.q[index], p=self.p[index])

        return PSP


# ##############################################################################


class TestPhaseSpacePosition(AbstractPhaseSpacePosition_Test[PhaseSpacePosition]):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""
        return PhaseSpacePosition
