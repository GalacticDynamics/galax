"""Test :class:`~galax.coordinates._pspt`."""

from dataclasses import replace
from typing import Any, Self, TypeVar

import astropy.units as u
import jax.experimental.array_api as xp
import jax.numpy as jnp
import jax.random as jr
import pytest

from galax.coordinates import AbstractPhaseSpaceTimePosition, PhaseSpaceTimePosition
from galax.potential import AbstractPotentialBase, KeplerPotential
from galax.potential._potential.special import MilkyWayPotential
from galax.typing import BatchVec7, FloatScalar, Vec3
from galax.units import UnitSystem, galactic

from .test_base import AbstractPhaseSpacePositionBase_Test, Shape, return_keys

T = TypeVar("T", bound=AbstractPhaseSpaceTimePosition)

potentials = [KeplerPotential(m=1e12 * u.Msun, units=galactic), MilkyWayPotential()]


class AbstractPhaseSpaceTimePosition_Test(AbstractPhaseSpacePositionBase_Test[T]):
    def make_w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = jr.normal(next(subkeys), (*shape, 3))
        p = jr.normal(next(subkeys), (*shape, 3))
        t = jr.normal(next(subkeys), shape)
        return w_cls(q=q, p=p, t=t)

    # ===============================================================

    def test_getitem_int(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert w[0] == replace(w, q=w.q[0], p=w.p[0], t=w.t[0])

    def test_getitem_slice(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert w[:5] == replace(w, q=w.q[:5], p=w.p[:5], t=w.t[:5])

    def test_getitem_boolarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.ones(w.q.shape[:-1], dtype=bool)
        idx = idx.at[::2].set(values=False)

        assert w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx])

    def test_getitem_intarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.asarray([0, 2, 1])
        assert w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx])

    # TODO: further tests for getitem
    # def test_getitem()

    # ===============================================================
    # Convenience methods

    def test_wt(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpaceTimePosition.wt`."""
        wt = w.wt()
        assert wt.shape == w.full_shape
        assert jnp.array_equal(wt[..., 0:3], w.q)
        assert jnp.array_equal(wt[..., 3:6], w.p)
        assert jnp.array_equal(wt[..., -1], w.t)

        with pytest.raises(NotImplementedError):
            w.wt(units=galactic)

    # ===============================================================

    @pytest.mark.parametrize("potential", potentials)
    def potential_energy(self, w: T, potential: AbstractPotentialBase) -> None:
        """Test method ``potential_energy``."""
        pe = w.potential_energy(potential)
        assert pe.shape == w.shape  # confirm relation to shape and components
        assert xp.all(pe <= 0)
        # definitional
        assert jnp.array_equal(pe, potential.potential_energy(w.q))

    @pytest.mark.parametrize("potential", potentials)
    def energy(self, w: T, potential: AbstractPotentialBase) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.energy`."""
        pe = w.energy(potential)
        assert pe.shape == w.shape  # confirm relation to shape and components
        # definitional
        assert jnp.array_equal(pe, w.kinetic_energy() + potential.potential_energy(w.q))


##############################################################################


class TestAbstractPhaseSpaceTimePosition(
    AbstractPhaseSpaceTimePosition_Test[AbstractPhaseSpaceTimePosition]
):
    """Test :class:`~galax.coordinates.AbstractPhaseSpaceTimePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""

        class PSP(AbstractPhaseSpaceTimePosition):
            """A phase-space position."""

            q: Vec3
            p: Vec3
            t: FloatScalar

            @property
            def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
                return self.q.shape[:-1], (3, 3, 1)

            def __getitem__(self, index: Any) -> Self:
                return replace(self, q=self.q[index], p=self.p[index], t=self.t[index])

            def wt(self, *, units: UnitSystem | None = None) -> BatchVec7:
                """Phase-space position as an Array[float, (*batch, Q + P + 1)].

                This is the full phase-space position, including the time.

                Parameters
                ----------
                units : `galax.units.UnitSystem`, optional keyword-only
                    The unit system If ``None``, use the current unit system.

                Returns
                -------
                wt : Array[float, (*batch, Q + P + 1)]
                    The full phase-space position, including time.
                """
                if units is not None:
                    msg = "units not yet implemented."
                    raise NotImplementedError(msg)

                batch_shape, comp_shapes = self._shape_tuple
                q = xp.broadcast_to(self.q, batch_shape + comp_shapes[0:1])
                p = xp.broadcast_to(self.p, batch_shape + comp_shapes[1:2])
                t = self.t[..., None]
                return xp.concat((q, p, t), axis=-1)

        return PSP


# ##############################################################################


class TestPhaseSpaceTimePosition(
    AbstractPhaseSpaceTimePosition_Test[PhaseSpaceTimePosition]
):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""
        return PhaseSpaceTimePosition
