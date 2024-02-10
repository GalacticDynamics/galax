"""Test :attr:`~galax.coordinates._base`."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import array_api_jax_compat as xp
import jax.random as jr
import pytest
from jaxtyping import Array

from galax.typing import Vec3
from galax.units import galactic

if TYPE_CHECKING:
    from pytest import FixtureRequest  # noqa: PT013

from galax.coordinates import AbstractPhaseSpacePositionBase

Shape: TypeAlias = tuple[int, ...]
T = TypeVar("T", bound=AbstractPhaseSpacePositionBase)


def return_keys(num: int, key: Array | int = 0) -> Iterable[jr.PRNGKey]:
    """Return an iterable of keys."""
    key = jr.PRNGKey(key) if isinstance(key, int) else key
    newkey, *subkeys = jr.split(key, num=num + 1)
    return newkey, iter(subkeys)


class AbstractPhaseSpacePositionBase_Test(Generic[T], metaclass=ABCMeta):
    """Test :class:`~galax.coordinates.AbstractPhaseSpacePosition`."""

    @pytest.fixture(scope="class", params=[(10,), (5, 4)])
    def shape(self, request: FixtureRequest) -> Shape:
        """Return a shape."""
        return request.param

    @pytest.fixture(scope="class")
    @abstractmethod
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""
        raise NotImplementedError

    @abstractmethod
    def make_w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        raise NotImplementedError

    @pytest.fixture(scope="class")
    def w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape)

    # ===============================================================
    # Attributes

    def test_q(self, w: T, shape: Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpacePosition.q`."""
        assert hasattr(w, "q")
        assert w.q.shape == (*shape, 3)

    def test_p(self, w: T, shape: Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpacePosition.p`."""
        assert hasattr(w, "p")
        assert w.p.shape == w.q.shape
        assert w.p.shape == (*shape, 3)

    # ===============================================================
    # Array properties

    def test_shape(self, w: T, shape: Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpacePosition.shape`."""
        # Check existence
        assert hasattr(w, "shape")

        # Confirm relation to shape_tuple
        assert w.shape == w._shape_tuple[0]

        # Confirm relation to components full shape
        assert w.shape == w.q.shape[:-1]

        # Confirm from definition
        assert w.shape == shape

    def test_ndim(self, w: T) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpacePosition.ndim`."""
        # Check existence
        assert hasattr(w, "ndim")

        # Confirm relation to shape and components
        assert w.ndim == len(w.shape)
        assert w.ndim == w.q.ndim - 1

    def test_len(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__len__`."""
        # Check existence
        assert hasattr(w, "__len__")

        # Confirm relation to shape and components
        assert len(w) == w.shape[0]
        assert len(w) == w.q.shape[0]

    # ----------------------------

    def test_getitem_int(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert w[0] == replace(w, q=w.q[0], p=w.p[0])

    def test_getitem_slice(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        assert w[:5] == replace(w, q=w.q[:5], p=w.p[:5])

    def test_getitem_boolarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.ones(w.q.shape[:-1], dtype=bool)
        idx = idx.at[::2].set(values=False)

        assert w[idx] == replace(w, q=w.q[idx], p=w.p[idx])

    def test_getitem_intarray(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        idx = xp.asarray([0, 2, 1])
        assert w[idx] == replace(w, q=w.q[idx], p=w.p[idx])

    # ==========================================================================

    def test_full_shape(self, w: T, shape: Shape) -> None:
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
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.w`."""
        # Check existence
        assert hasattr(w, "w")

        # Confirm relation to shape and components
        assert w.w().shape[:-1] == w.full_shape[:-1]

        # units != None
        with pytest.raises(NotImplementedError):
            w.w(units=galactic)

    # ==========================================================================
    # Dynamical properties

    def test_kinetic_energy(self, w: T) -> None:
        """Test method ``kinetic_energy``."""
        ke = w.kinetic_energy()
        assert ke.shape == w.shape  # confirm relation to shape and components
        assert xp.all(ke >= 0)
        # TODO: more tests

    def test_angular_momentum(self, w: T) -> None:
        """Test method ``angular_momentum``."""
        am = w.angular_momentum()
        assert am.shape == w.q.shape
        # TODO: more tests


##############################################################################


class TestAbstractPhaseSpacePositionBase(AbstractPhaseSpacePositionBase_Test[T]):
    """Test :class:`~galax.coordinates.AbstractPhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[T]:
        """Return the class of a phase-space position."""

        class PSP(AbstractPhaseSpacePositionBase):
            """A phase-space position."""

            q: Vec3
            p: Vec3

            @property
            def _shape_tuple(self) -> tuple[tuple[int, ...], tuple[int, int]]:
                return self.q.shape[:-1], (3, 3)

            def __getitem__(self, index: Any) -> Self:
                return replace(self, q=self.q[index], p=self.p[index])

        return PSP

    def make_w(self, w_cls: type[T], shape: Shape) -> T:
        """Return a phase-space position."""
        _, subkeys = return_keys(2)

        q = jr.normal(next(subkeys), (*shape, 3))
        p = jr.normal(next(subkeys), (*shape, 3))
        return w_cls(q, p)

    # ===============================================================

    def test_getitem_int(self, w: T) -> None:
        """Test :meth:`~galax.coordinates.AbstractPhaseSpacePosition.__getitem__`."""
        # Check existence
        assert w[0] == replace(w, q=w.q[0], p=w.p[0])
