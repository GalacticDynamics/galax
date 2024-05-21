"""Test :class:`~galax.coordinates._pspt`."""

from typing import TypeVar

import jax.numpy as jnp
import jax.random as jr
import pytest
from plum import convert

import quaxed.array_api as xp
from unxt import Quantity
from unxt.unitsystems import galactic

import galax.typing as gt
from ..coordinates.psp.test_base_psp import AbstractPhaseSpacePosition_Test, return_keys
from galax.coordinates import PhaseSpacePosition
from galax.dynamics import Orbit
from galax.potential import AbstractPotentialBase, MilkyWayPotential

T = TypeVar("T", bound=Orbit)


class TestOrbit(AbstractPhaseSpacePosition_Test[Orbit]):
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
        self, w_cls: type[T], shape: gt.Shape, potential: AbstractPotentialBase
    ) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = Quantity(jr.normal(next(subkeys), (*shape, 3)), "kpc")
        p = Quantity(jr.normal(next(subkeys), (*shape, 3)), "km/s")
        t = Quantity(jr.normal(next(subkeys), shape[-1:]), unit=potential.units["time"])
        return w_cls(q=q, p=p, t=t, potential=potential)

    @pytest.fixture()
    def w(self, w_cls: type[T], shape: gt.Shape, potential: AbstractPotentialBase) -> T:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape, potential)

    # ===============================================================

    def test_getitem_int(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert not isinstance(w[0], type(w))
        assert w[0] == PhaseSpacePosition(q=w.q[0], p=w.p[0], t=w.t[0])

    def test_getitem_boolarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.ones(w.q.shape, dtype=bool)
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
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.wt`."""
        wt = w.wt(units=galactic)
        assert wt.shape == w.full_shape
        assert jnp.array_equal(
            wt[(*(0,) * (w.ndim - 1), slice(None), 0)], w.t.decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 1:4], convert(w.q, Quantity).decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 4:7], convert(w.p, Quantity).decompose(galactic).value
        )
