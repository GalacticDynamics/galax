"""Test `galax.coordinates.AbstractOnePhaseSpacePosition`."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dataclasses import KW_ONLY, replace
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import equinox as eqx
import jax.random as jr
import pytest
from jaxtyping import Array
from plum import convert

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import galactic

import galax.coordinates as gc
import galax.typing as gt
from galax.potential import AbstractPotential, KeplerPotential, MilkyWayPotential

if TYPE_CHECKING:
    from pytest import FixtureRequest  # noqa: PT013


T = TypeVar("T", bound=gc.AbstractOnePhaseSpacePosition)

potentials = [
    KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units=galactic),
    MilkyWayPotential(),
]


def return_keys(num: int, key: Array | int = 0) -> Iterable[jr.PRNGKey]:
    """Return an iterable of keys."""
    key = jr.PRNGKey(key) if isinstance(key, int) else key
    newkey, *subkeys = jr.split(key, num=num + 1)
    return newkey, iter(subkeys)


class AbstractOnePhaseSpacePosition_Test(Generic[T], metaclass=ABCMeta):
    """Test :class:`~galax.coordinates.AbstractOnePhaseSpacePosition`."""

    @pytest.fixture(scope="class", params=[(10,), (5, 4)], ids=lambda param: str(param))
    def shape(self, request: FixtureRequest) -> gt.Shape:
        """Return a shape."""
        return request.param

    @pytest.fixture(scope="class")
    @abstractmethod
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""
        raise NotImplementedError

    def make_w(self, w_cls: type[T], shape: gt.Shape) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(3)

        q = u.Quantity(jr.normal(next(subkeys), (*shape, 3)), "kpc")
        p = u.Quantity(jr.normal(next(subkeys), (*shape, 3)), "km/s")
        t = u.Quantity(jr.normal(next(subkeys), shape), "Myr")
        return w_cls(q=q, p=p, t=t, frame=gc.frames.SimulationFrame())

    @pytest.fixture
    def w(self, w_cls: type[T], shape: gt.Shape) -> T:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape)

    # ===============================================================
    # Attributes

    def test_q(self, w: T, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractOnePhaseSpacePosition.q`."""
        assert hasattr(w, "q")
        assert w.q.shape == shape
        assert len(w.q.components) == 3

    def test_p(self, w: T, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractOnePhaseSpacePosition.p`."""
        assert hasattr(w, "p")
        assert w.p.shape == w.q.shape
        assert w.p.shape == shape
        assert len(w.p.components) == 3

    # ===============================================================
    # Array properties

    def test_shape(self, w: T, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractOnePhaseSpacePosition.shape`."""
        # Check existence
        assert hasattr(w, "shape")

        # Confirm relation to shape_tuple
        assert w.shape == w._shape_tuple[0]

        # Confirm relation to components full shape
        assert w.shape == w.q.shape

        # Confirm from definition
        assert w.shape == shape

    def test_ndim(self, w: T) -> None:
        """Test :attr:`~galax.coordinates.AbstractOnePhaseSpacePosition.ndim`."""
        # Check existence
        assert hasattr(w, "ndim")

        # Confirm relation to shape and components
        assert w.ndim == len(w.shape)
        assert w.ndim == w.q.ndim

    def test_len(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractOnePhaseSpacePosition.__len__`."""
        # Check existence
        assert hasattr(w, "__len__")

        # Confirm relation to shape and components
        assert len(w) == w.shape[0]
        assert len(w) == w.q.shape[0]

    # ----------------------------

    def test_getitem_int(self, w: T) -> None:
        """Test `~galax.coordinates.AbstractOnePhaseSpacePosition.__getitem__`."""
        assert jnp.all(w[0] == replace(w, q=w.q[0], p=w.p[0], t=w.t[0]))

    def test_getitem_slice(self, w: T) -> None:
        """Test `~galax.coordinates.AbstractOnePhaseSpacePosition.__getitem__`."""
        assert jnp.all(w[:5] == replace(w, q=w.q[:5], p=w.p[:5], t=w.t[:5]))

    def test_getitem_boolarray(self, w: T) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = jnp.ones(w.q.shape, dtype=bool)
        idx = idx.at[::2].set(values=False)

        assert all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx]))

    def test_getitem_intarray(self, w: T) -> None:
        """Test `~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = jnp.asarray([0, 2, 1])
        assert jnp.all(w[idx] == replace(w, q=w.q[idx], p=w.p[idx], t=w.t[idx]))

    # TODO: further tests for getitem
    # def test_getitem()

    # ==========================================================================

    def test_full_shape(self, w: T, shape: gt.Shape) -> None:
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.full_shape`."""
        # Definition
        batch_shape, component_shapes = w._shape_tuple
        assert w.full_shape == (*batch_shape, sum(component_shapes))

        # Sanity check
        assert w.full_shape[: len(w.shape)] == w.shape
        assert len(w.full_shape) == len(w.shape) + 1

    # ==========================================================================
    # Convenience methods

    def test_w(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractOnePhaseSpacePosition.w`."""
        # Check existence
        assert hasattr(w, "w")

        # units are not None
        assert w.w(units=galactic).shape[:-1] == w.full_shape[:-1]

    def test_wt(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractOnePhaseSpacePosition.wt`."""
        wt = w.wt(units=galactic)
        assert wt.shape == w.full_shape
        assert jnp.array_equal(wt[..., 0], w.t.decompose(galactic).value)
        assert jnp.array_equal(
            wt[..., 1:4], convert(w.q, u.Quantity).decompose(galactic).value
        )
        assert jnp.array_equal(
            wt[..., 4:7], convert(w.p, u.Quantity).decompose(galactic).value
        )

    def test_uconvert(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractOnePhaseSpacePosition.uconvert`."""
        w2 = w.uconvert("solarsystem")
        # TODO: more detailed tests
        assert w2.q.x.unit == "AU"
        assert w2.p.x.unit == "AU/yr"
        assert w2.t.unit == "yr"

    # ==========================================================================
    # Dynamical properties

    def test_kinetic_energy(self, w: T) -> None:
        """Test method ``kinetic_energy``."""
        ke = w.kinetic_energy()
        assert ke.shape == w.shape  # confirm relation to shape and components
        assert jnp.all(ke >= u.Quantity(0, "km2/s2"))
        # TODO: more tests

    @pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
    def test_potential(self, w: T, pot: AbstractPotential) -> None:
        """Test method ``potential``."""
        pe = w.potential_energy(pot)
        assert pe.shape == w.shape  # confirm relation to shape and components
        assert jnp.all(pe <= u.Quantity(0, "km2/s2"))
        # definitional
        assert jnp.allclose(
            pe, pot.potential(w.q, t=0), atol=u.Quantity(1e-10, pe.unit)
        )

    @pytest.mark.parametrize("pot", potentials, ids=lambda p: type(p).__name__)
    def test_total_energy(self, w: T, pot: AbstractPotential) -> None:
        """Test :meth:`~galax.coordinates.AbstractOnePhaseSpacePosition.energy`."""
        pe = w.total_energy(pot)
        assert pe.shape == w.shape  # confirm relation to shape and components
        # definitional
        assert jnp.allclose(
            pe,
            w.kinetic_energy() + pot.potential(w.q, t=0),
            atol=u.Quantity(1e-10, pe.unit),
        )

    def test_angular_momentum(self, w: T) -> None:
        """Test method ``angular_momentum``."""
        am = w.angular_momentum()
        assert am.shape == (*w.q.shape, len(w.q.components))
        # TODO: more tests


##############################################################################


class TestAbstractOnePhaseSpacePosition(AbstractOnePhaseSpacePosition_Test[T]):
    """Test :class:`~galax.coordinates.AbstractOnePhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""

        class PSP(gc.AbstractOnePhaseSpacePosition):
            """A phase-space position."""

            q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
            p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
            t: u.Quantity["time"]

            _: KW_ONLY
            frame: cx.frames.AbstractReferenceFrame

            @property
            def _shape_tuple(self) -> tuple[gt.Shape, gc.ComponentShapeTuple]:
                return self.q.shape, gc.ComponentShapeTuple(p=3, q=3, t=1)

            def __getitem__(self, index: Any) -> Self:
                return replace(self, q=self.q[index], p=self.p[index], t=self.t[index])

            def wt(self, *, units: AbstractUnitSystem | None = None) -> BtSz7:
                """Phase-space position as an Array[float, (*batch, Q + P + 1)].

                This is the full phase-space position, including the time.

                Parameters
                ----------
                units : `unxt.AbstractUnitSystem`, optional keyword-only
                    The unit system If ``None``, use the current unit system.

                Returns
                -------
                wt : Array[float, (*batch, 1+Q+P)]
                    The full phase-space position, including time.
                """
                batch, comps = self._shape_tuple
                cart = self.vconvert(cx.CartesianPos3D)
                q = jnp.broadcast_to(
                    convert(cart.q, u.Quantity).decompose(units).value,
                    (*batch, comps.q),
                )
                p = jnp.broadcast_to(
                    convert(cart.p, u.Quantity).decompose(units).value,
                    (*batch, comps.p),
                )
                t = jnp.broadcast_to(
                    self.t.decompose(units).value[..., None], (*batch, comps.t)
                )
                return jnp.concat((t, q, p), axis=-1)

        return PSP
