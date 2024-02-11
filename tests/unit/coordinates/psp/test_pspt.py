"""Test :class:`~galax.coordinates._pspt`."""

from dataclasses import replace
from typing import Any, Self, TypeVar

import array_api_jax_compat as xp
import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from plum import convert

from jax_quantity import Quantity
from vector import Cartesian3DVector, CartesianDifferential3D

from .test_base import AbstractPhaseSpacePositionBase_Test, Shape, return_keys
from galax.coordinates import AbstractPhaseSpaceTimePosition, PhaseSpaceTimePosition
from galax.coordinates._psp.base import _p_converter, _q_converter
from galax.potential import AbstractPotentialBase, KeplerPotential
from galax.potential._potential.special import MilkyWayPotential
from galax.typing import BatchVec7, FloatScalar
from galax.units import UnitSystem, galactic

T = TypeVar("T", bound=AbstractPhaseSpaceTimePosition)

potentials = [KeplerPotential(m=1e12 * u.Msun, units=galactic), MilkyWayPotential()]


class AbstractPhaseSpaceTimePosition_Test(AbstractPhaseSpacePositionBase_Test[T]):
    def make_w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = Quantity(jr.normal(next(subkeys), (*shape, 3)), "kpc")
        p = Quantity(jr.normal(next(subkeys), (*shape, 3)), "km/s")
        t = Quantity(jr.normal(next(subkeys), shape), "Myr")
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
        idx = xp.ones(w.q.shape, dtype=bool)
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
        wt = w.wt(units=galactic)
        assert wt.shape == w.full_shape
        assert jnp.array_equal(wt[..., 0], w.t.decompose(galactic).value)
        assert jnp.array_equal(
            wt[..., 1:4], convert(w.q, Quantity).decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 4:7], convert(w.p, Quantity).decompose(galactic).value
        )

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

            q: Cartesian3DVector = eqx.field(converter=_q_converter)
            p: CartesianDifferential3D = eqx.field(converter=_p_converter)
            t: FloatScalar

            @property
            def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
                return self.q.shape, (3, 3, 1)

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
                wt : Array[float, (*batch, 1+Q+P)]
                    The full phase-space position, including time.
                """
                batch_shape, comp_shapes = self._shape_tuple
                cart = self.represent_as(Cartesian3DVector)
                q = xp.broadcast_to(
                    convert(cart.q, Quantity).decompose(units).value,
                    (*batch_shape, comp_shapes[0]),
                )
                p = xp.broadcast_to(
                    convert(cart.p, Quantity).decompose(units).value,
                    (*batch_shape, comp_shapes[1]),
                )
                t = xp.broadcast_to(self.t.decompose(units).value, batch_shape)[
                    ..., None
                ]
                return xp.concat((t, q, p), axis=-1)

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
