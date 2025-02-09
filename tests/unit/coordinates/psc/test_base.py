"""Test `galax.coordinates.AbstractPhaseSpaceCoordinate`."""

from abc import ABCMeta
from dataclasses import replace
from typing import TypeVar

import jax.random as jr
import pytest
from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish import replace
from unxt.unitsystems import galactic

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from ..test_base import AbstractPhaseSpaceObject_Test, getkeys

WT = TypeVar("WT", bound=gc.AbstractPhaseSpaceCoordinate)


potentials = [
    gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units=galactic),
    gp.MilkyWayPotential(),
]


class AbstractPhaseSpaceCoordinate_Test(
    AbstractPhaseSpaceObject_Test[WT], metaclass=ABCMeta
):
    """Test :class:`~galax.coordinates.AbstractPhaseSpaceCoordinate`."""

    def make_w(self, w_cls: type[WT], shape: gt.Shape) -> WT:
        """Return a phase-space position."""
        _, keys = getkeys(3)

        q = u.Quantity(jr.normal(next(keys), (*shape, 3)), "kpc")
        p = u.Quantity(jr.normal(next(keys), (*shape, 3)), "km/s")
        t = u.Quantity(jr.normal(next(keys), shape), "Myr")
        return w_cls(q=q, p=p, t=t, frame=gc.frames.SimulationFrame())

    #################################################################

    # =========================================================
    # Attributes

    def test_t(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.t`."""
        assert w.t.shape == w.shape
        assert isinstance(w.t, u.Quantity)
        assert u.dimension_of(w.t) == "time"

    # =========================================================
    # Coordinate API

    def test_dimensionality(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.dimensionality`."""
        assert w._dimensionality() == 7

    def test_data_keys(self, w: WT) -> None:
        """Test :attr:`~galax.coordinates.PhaseSpacePosition.data`."""
        assert isinstance(w.data, cx.Space)

        assert "length" in w.data
        assert isinstance(w.data["length"], cx.vecs.FourVector)

        assert "speed" in w.data
        assert isinstance(w.data["speed"], cx.vecs.AbstractVel3D)

    # =========================================================
    # Array API

    # ----------------------------
    # getitem

    def test_getitem_int(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.__getitem__`."""
        assert jnp.all(w[0] == replace(w, q=w.q[0], p=w.p[0], t=w.t[0]))

    def test_getitem_slice(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.__getitem__`."""
        assert jnp.all(w[:5] == replace(w, q=w.q[:5], p=w.p[:5], t=w.t[:5]))

    def test_getitem_boolarray(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.__getitem__`."""
        idx = jnp.ones(w.q.shape, dtype=bool)
        idx = idx.at[::2].set(values=False)

        assert all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx]))

    def test_getitem_intarray(self, w: WT) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpaceCoordinate.__getitem__`."""
        idx = jnp.asarray([0, 2, 1])
        assert jnp.all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx]))

    # ----------------------------

    def test_wt(self, w: gc.AbstractPhaseSpaceCoordinate) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpaceCoordinate.wt`."""
        wt = w.wt(units=galactic)
        assert wt.shape == w.full_shape
        assert jnp.array_equal(wt[..., 0], w.t.decompose(galactic).value)
        assert jnp.array_equal(
            wt[..., 1:4], convert(w.q, u.Quantity).decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 4:7], convert(w.p, u.Quantity).decompose(galactic).value
        )

    # -----------------------------

    def test_uconvert(self, w: gc.AbstractPhaseSpaceCoordinate) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpaceCoordinate.uconvert`."""
        w2 = w.uconvert("solarsystem")
        # TODO: more detailed tests
        assert w2.q.x.unit == "AU"
        assert w2.p.x.unit == "AU/yr"
        assert w2.t.unit == "yr"

    # ------------------------------

    @pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
    def test_potential_energy(
        self, w: gc.AbstractPhaseSpaceCoordinate, pot: gp.AbstractPotential
    ) -> None:
        """Test method ``potential``."""
        pe = w.potential_energy(pot)
        assert pe.shape == w.shape  # confirm relation to shape and components
        assert jnp.all(pe <= u.Quantity(0, "km2/s2"))
        # definitional
        assert jnp.allclose(
            pe, pot.potential(w.q, t=0), atol=u.Quantity(1e-10, pe.unit)
        )

    # ------------------------------

    @pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
    def test_total_energy(
        self, w: gc.AbstractPhaseSpaceCoordinate, pot: gp.AbstractPotential
    ) -> None:
        """Test :meth:`~galax.coordinates.PhaseSpacePosition.energy`."""
        pe = w.total_energy(pot)
        assert pe.shape == w.shape  # confirm relation to shape and components
        # definitional
        assert jnp.allclose(
            pe,
            w.kinetic_energy() + pot.potential(w.q, t=0),
            atol=u.Quantity(1e-10, pe.unit),
        )
