"""Test `galax.coordinates.PhaseSpacePosition`."""

from __future__ import annotations

from dataclasses import replace

import jax.random as jr
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from ..test_base import AbstractPhaseSpaceObject_Test, getkeys


class Test_PhaseSpacePosition(AbstractPhaseSpaceObject_Test[gc.PhaseSpacePosition]):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[gc.PhaseSpacePosition]:
        """Return the class of a phase-space position."""
        return gc.PhaseSpacePosition

    def make_w(
        self, w_cls: type[gc.PhaseSpacePosition], shape: gt.Shape
    ) -> gc.PhaseSpacePosition:
        """Return a phase-space position."""
        _, keys = getkeys(3)

        q = u.Quantity(jr.normal(next(keys), (*shape, 3)), "kpc")
        p = u.Quantity(jr.normal(next(keys), (*shape, 3)), "km/s")
        return w_cls(q=q, p=p, frame=gc.frames.SimulationFrame())

    # ===============================================================
    # Array properties

    # ----------------------------

    def test_getitem_int(self, w: gc.PhaseSpacePosition) -> None:
        """Test `~galax.coordinates.PhaseSpacePosition.__getitem__`."""
        assert jnp.all(w[0] == replace(w, q=w.q[0], p=w.p[0]))

    def test_getitem_slice(self, w: gc.PhaseSpacePosition) -> None:
        """Test `~galax.coordinates.PhaseSpacePosition.__getitem__`."""
        assert jnp.all(w[:5] == replace(w, q=w.q[:5], p=w.p[:5]))

    def test_getitem_boolarray(self, w: gc.PhaseSpacePosition) -> None:
        """Test `~galax.coordinates.PhaseSpacePosition.__getitem__`."""
        idx = jnp.ones(w.q.shape, dtype=bool)
        idx = idx.at[::2].set(values=False)

        assert all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx]))

    def test_getitem_intarray(self, w: gc.PhaseSpacePosition) -> None:
        """Test `~galax.coordinates.PhaseSpacePosition.__getitem__`."""
        idx = jnp.asarray([0, 2, 1])
        assert jnp.all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx]))

    # TODO: further tests for getitem
    # def test_getitem()
