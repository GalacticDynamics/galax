"""Test :class:`~galax.coordinates._pspt`."""

import jax.experimental.array_api as xp
import jax.numpy as jnp
import jax.random as jr
import pytest

from galax.coordinates import PhaseSpaceTimePosition
from galax.dynamics import Orbit
from galax.potential import AbstractPotentialBase, MilkyWayPotential
from galax.units import galactic

from ..coordinates.test_base import Shape, return_keys
from ..coordinates.test_pspt import AbstractPhaseSpaceTimePosition_Test, T


class TestOrbit(AbstractPhaseSpaceTimePosition_Test[Orbit]):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""
        return Orbit

    @pytest.fixture(scope="class")
    def potential(self) -> AbstractPotentialBase:
        """Return a potential."""
        return MilkyWayPotential()

    def make_w(
        self, w_cls: type[T], shape: Shape, potential: AbstractPotentialBase
    ) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = jr.normal(next(subkeys), (*shape, 3))
        p = jr.normal(next(subkeys), (*shape, 3))
        t = jr.normal(next(subkeys), shape[-1:])
        return w_cls(q=q, p=p, t=t, potential=potential)

    @pytest.fixture(scope="class")
    def w(self, w_cls: type[T], shape: Shape, potential: AbstractPotentialBase) -> T:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape, potential)

    # ===============================================================

    def test_getitem_int(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert not isinstance(w[0], type(w))
        assert w[0] == PhaseSpaceTimePosition(q=w.q[0], p=w.p[0], t=w.t[0])

    def test_getitem_boolarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.ones(w.q.shape[:-1], dtype=bool)
        idx = idx.at[::2].set(values=False)

        with pytest.raises(NotImplementedError):
            _ = w[idx]

    def test_getitem_intarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.asarray([0, 2, 1])
        with pytest.raises(NotImplementedError):
            _ = w[idx]

    # ===============================================================

    def test_wt(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpaceTimePosition.wt`."""
        wt = w.wt()
        assert wt.shape == w.full_shape
        assert jnp.array_equal(wt[(*(0,) * (w.ndim - 1), slice(None), 0)], w.t)
        assert jnp.array_equal(wt[..., 1:4], w.q)
        assert jnp.array_equal(wt[..., 4:7], w.p)

        with pytest.raises(NotImplementedError):
            w.wt(units=galactic)
