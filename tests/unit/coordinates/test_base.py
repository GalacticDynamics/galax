"""Test `galax.coordinates.AbstractPhaseSpaceObject`."""

from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import jax.random as jr
import optype as op
import pytest
from jaxtyping import PRNGKeyArray

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.coordinates as gc

CT = TypeVar("CT", bound=gc.AbstractPhaseSpaceObject)


def getkeys(
    num: int, key: PRNGKeyArray | int = 0
) -> tuple[PRNGKeyArray, op.CanIter[PRNGKeyArray]]:
    """Return an iterable of keys."""
    key = jr.key(key) if isinstance(key, int) else key
    newkey, *subkeys = jr.split(key, num=num + 1)
    return newkey, iter(subkeys)


class AbstractPhaseSpaceObject_Test(Generic[CT], metaclass=ABCMeta):
    """Abstract base class for testing `galax.coordinates.AbstractPhaseSpaceObject`."""

    #################################################################
    # Fixtures

    @pytest.fixture(
        scope="class",
        params=[gc.frames.simulation_frame],
        ids=lambda param: str(param),
    )
    def frame(self, request: pytest.FixtureRequest) -> cx.frames.AbstractReferenceFrame:
        """Return the frame to test."""
        return request.param

    # ----------------------------------------------------

    @pytest.fixture(scope="class", params=[(10,), (5, 4)], ids=lambda param: str(param))
    def shape(self, request: pytest.FixtureRequest) -> gt.Shape:
        """Return a shape."""
        return request.param

    @pytest.fixture(scope="class")
    @abstractmethod
    def w_cls(self) -> type[CT]:
        """Return the class of a phase-space position."""
        raise NotImplementedError

    @abstractmethod
    def make_w(
        self, w_cls: type[CT], shape: gt.Shape, frame: cx.frames.AbstractReferenceFrame
    ) -> CT:
        """Return a phase-space position."""
        raise NotImplementedError

    @pytest.fixture
    def w(self, w_cls: type[CT], shape: gt.Shape) -> CT:
        """Return a phase-space position."""
        return self.make_w(w_cls, shape)

    #################################################################

    # =========================================================
    # Attributes

    def test_q(self, w: CT, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpaceObject.q`."""
        assert hasattr(w, "q")
        assert w.q.shape == shape
        assert len(w.q.components) == 3

    def test_p(self, w: CT, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpaceObject.p`."""
        assert hasattr(w, "p")
        assert w.p.shape == w.q.shape
        assert w.p.shape == shape
        assert len(w.p.components) == 3

    def test_frame(self, w: CT, frame: cx.frames.AbstractReferenceFrame) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpaceObject.frame`."""
        assert hasattr(w, "frame")
        assert w.frame == frame

    # =========================================================
    # Vector API

    def test_data_keys(self, w: CT) -> None:
        """Test :attr:`~galax.coordinates.PhaseSpacePosition.data`."""
        assert isinstance(w.data, cx.KinematicSpace)
        assert "length" in w.data
        assert "speed" in w.data

    def test_uconvert(self, w: CT) -> None:
        """Test :meth:`~galax.coordinates.PhaseSpacePosition.uconvert`."""
        w2 = w.uconvert("solarsystem")
        # TODO: more detailed tests
        assert w2.q.x.unit == "AU"
        assert w2.p.x.unit == "AU/yr"

    # =========================================================
    # Array API

    def test_shape(self, w: CT, shape: gt.Shape) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpaceObject.shape`."""
        # Check existence
        assert hasattr(w, "shape")

        # Confirm relation to shape_tuple
        assert w.shape == w._shape_tuple[0]

        # Confirm relation to components full shape
        assert w.shape == w.q.shape

        # Confirm from definition
        assert w.shape == shape

    def test_ndim(self, w: CT) -> None:
        """Test :attr:`~galax.coordinates.AbstractPhaseSpaceObject.ndim`."""
        # Check existence
        assert hasattr(w, "ndim")

        # Confirm relation to shape and components
        assert w.ndim == len(w.shape)
        assert w.ndim == w.q.ndim

    def test_len(self, w: CT) -> None:
        """Test :meth:`~galax.coordinates.PhaseSpacePosition.__len__`."""
        # Check existence
        assert hasattr(w, "__len__")

        # Confirm relation to shape and components
        assert len(w) == w.shape[0]
        assert len(w) == w.q.shape[0]

    # def test_getitem()  # TODO: abstract tests

    # =========================================================
    # Python API

    def test_str(self, w: CT) -> None:
        """Test `__str__`."""
        assert isinstance(str(w), str)
        # TODO: more

    # =========================================================
    # Further Array properties

    def test_full_shape(self, w: CT, shape: gt.Shape) -> None:
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.full_shape`."""
        # Definition
        batch_shape, component_shapes = w._shape_tuple
        assert w.full_shape == (*batch_shape, sum(component_shapes))

        # Sanity check
        assert w.full_shape[: len(w.shape)] == w.shape
        assert len(w.full_shape) == len(w.shape) + 1

    # =========================================================

    def test_w(self, w: CT) -> None:
        """Test :meth:`~galax.coordinates.PhaseSpacePosition.w`."""
        # Check existence
        assert hasattr(w, "w")

        # units are not None
        assert w.w(units="galactic").shape[:-1] == w.full_shape[:-1]

    # =========================================================
    # Dynamical properties

    def test_kinetic_energy(self, w: CT) -> None:
        """Test method ``kinetic_energy``."""
        ke = w.kinetic_energy()
        assert ke.shape == w.shape  # confirm relation to shape and components
        assert jnp.all(ke >= u.Quantity(0, "km2/s2"))
        # TODO: more tests

    def test_angular_momentum(self, w: CT) -> None:
        """Test method ``angular_momentum``."""
        h = w.angular_momentum()
        assert h.shape == w.q.shape
        assert isinstance(h, cx.vecs.Cartesian3D)
