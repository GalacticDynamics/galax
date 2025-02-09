"""Test :class:`~galax.dynamics._src.orbit`."""

from typing_extensions import override

import jax.random as jr
import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import galactic

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
import galax.typing as gt
from ..coordinates.psc.test_base_single import AbstractBasicPhaseSpaceCoordinate_Test
from ..coordinates.test_base import getkeys


class TestOrbit(AbstractBasicPhaseSpaceCoordinate_Test[gd.Orbit]):
    """Test :class:`~galax.coordinates.PhaseSpaceCoordinate`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[gd.Orbit]:
        """Return the class of a phase-space position."""
        return gd.Orbit

    @pytest.fixture(scope="class")
    def potential(self) -> gp.AbstractPotential:
        """Return a potential."""
        return gp.MilkyWayPotential()

    def make_w(
        self, w_cls: type[gd.Orbit], shape: gt.Shape, potential: gp.AbstractPotential
    ) -> gd.Orbit:
        """Return a phase-space position."""
        _, subkeys = getkeys(3)

        q = u.Quantity(jr.normal(next(subkeys), (*shape, 3)), "kpc")
        p = u.Quantity(jr.normal(next(subkeys), (*shape, 3)), "km/s")
        t = u.Quantity(
            jr.normal(next(subkeys), shape[-1:]), unit=potential.units["time"]
        )
        return w_cls(
            q=q,
            p=p,
            t=t,
            potential=potential,
            interpolant=None,
            frame=gc.frames.SimulationFrame(),
        )

    @pytest.fixture
    def w(
        self, w_cls: type[gd.Orbit], shape: gt.Shape, potential: gp.AbstractPotential
    ) -> gd.Orbit:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape, potential=potential)

    # ===============================================================

    def test_t(self, w: gd.Orbit) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.t`."""
        assert w.t.shape == w.shape[-1:]
        assert isinstance(w.t, u.Quantity)
        assert u.dimension_of(w.t) == "time"

    def test_getitem_int(self, w: gd.Orbit) -> None:
        """Test ``PhaseSpaceCoordinate.__getitem__``."""
        assert not isinstance(w[0], type(w))
        assert jnp.all(w[0] == gc.PhaseSpaceCoordinate(q=w.q[0], p=w.p[0], t=w.t[0]))

    @override
    def test_getitem_boolarray(self, w: gd.Orbit, shape: gt.Shape) -> None:
        """Test ``PhaseSpaceCoordinate.__getitem__``."""
        idx = jnp.ones(len(w.q), dtype=bool)
        idx = idx.at[::2].set(values=False)
        tidx = Ellipsis if idx.ndim < w.ndim else idx

        new = w[idx]
        assert new.shape == (int(sum(idx)), *shape[1:])
        assert jnp.array_equal(new.q, w.q[idx])
        assert jnp.array_equal(new.p, w.p[idx])
        assert jnp.array_equal(new.t, w.t[tidx])

    def test_getitem_intarray(self, w: gd.Orbit, shape: gt.Shape) -> None:
        """Test ``PhaseSpaceCoordinate.__getitem__``."""
        idx = jnp.asarray([0, 2, 1])
        tidx = Ellipsis if idx.ndim < w.ndim else idx

        new = w[idx]
        assert new.shape == (int(sum(idx)), *shape[1:])
        assert jnp.array_equal(new.q, w.q[idx])
        assert jnp.array_equal(new.p, w.p[idx])
        assert jnp.array_equal(new.t, w.t[tidx])

    # ===============================================================

    def test_wt(self, w: gd.Orbit) -> None:
        """Test :meth:`~galax.coordinates.PhaseSpaceCoordinate.wt`."""
        wt = w.wt(units=galactic)
        assert wt.shape == w.full_shape
        assert jnp.array_equal(
            wt[(*(0,) * (w.ndim - 1), slice(None), 0)], w.t.decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 1:4], convert(w.q, u.Quantity).decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 4:7], convert(w.p, u.Quantity).decompose(galactic).value
        )
