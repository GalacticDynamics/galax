"""ABC for phase-space positions."""

__all__ = ["PhaseSpacePosition", "ComponentShapeTuple"]

import warnings
from dataclasses import KW_ONLY, replace
from typing import Any, ClassVar, NamedTuple
from typing_extensions import override

import equinox as eqx
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

import galax._custom_types as gt
from galax.coordinates._src.base import AbstractPhaseSpaceObject
from galax.coordinates._src.frames import SimulationFrame, simulation_frame
from galax.coordinates._src.utils import PSPVConvertOptions
from galax.utils._shape import vector_batched_shape


class ComponentShapeTuple(NamedTuple):
    """Component shape of the phase-space position."""

    q: int
    """Shape of the position component."""

    p: int
    """Shape of the momentum component."""


# =============================================================================


class PhaseSpacePosition(AbstractPhaseSpaceObject):
    r"""Phase-Space Position with time.

    The phase-space position is a point in the 6-dimensional phase space
    :math:`\mathbb{R}^6` of a dynamical system. It is composed of the position
    :math:`\boldsymbol{q}`  and the conjugate momentum :math:`\boldsymbol{p}`.

    Parameters
    ----------
    q : :class:`~coordinax.AbstractPos3D`
        A 3-vector of the positions, allowing for batched inputs.  This
        parameter accepts any 3-vector, e.g. :class:`~coordinax.SphericalPos`,
        or any input that can be used to make a
        :class:`~coordinax.CartesianPos3D` via :meth:`coordinax.vector`.
    p : :class:`~coordinax.AbstractVel3D`
        A 3-vector of the conjugate specific momenta at positions ``q``,
        allowing for batched inputs.  This parameter accepts any 3-vector
        differential, e.g.  :class:`~coordinax.SphericalVelocity`, or any input
        that can be used to make a :class:`~coordinax.CartesianVel3D` via
        :meth:`coordinax.vector`.

    Notes
    -----
    The batch shape of `q` and `p` are broadcast together.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    Note that both `q` and `p` have convenience converters, allowing them to
    accept a variety of inputs when constructing a
    :class:`~coordinax.CartesianPos3D` or :class:`~coordinax.CartesianVel3D`,
    respectively.  For example,

    >>> t = u.Quantity(7, "s")
    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                           p=u.Quantity([4, 5, 6], "m/s"))
    >>> w
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      frame=SimulationFrame()
    )

    This can be done more explicitly:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> p = cx.CartesianVel3D.from_([4, 5, 6], "m/s")

    >>> w2 = gc.PhaseSpacePosition(q=q, p=p)
    >>> w2 == w
    Array(True, dtype=bool)

    When using the explicit constructors, the inputs can be any
    `coordinax.AbstractPos3D` and `coordinax.AbstractVel3D` types:

    >>> q = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(2, "deg"),
    ...                     phi=u.Quantity(3, "deg"))
    >>> w3 = gc.PhaseSpacePosition(q=q, p=p)
    >>> isinstance(w3.q, cx.SphericalPos)
    True

    Of course a similar effect can be achieved by using the `coordinax.vconvert`
    function (or convenience method on the phase-space position):

    >>> w4 = w3.vconvert(cx.SphericalPos, cx.CartesianVel3D)
    >>> w4
    PhaseSpacePosition(
      q=SphericalPos( ... ),
      p=CartesianVel3D( ... ),
      frame=SimulationFrame()
    )

    """

    q: cx.vecs.AbstractPos3D = eqx.field(converter=cx.vector)
    """Positions, e.g CartesianPos3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    p: cx.vecs.AbstractVel3D = eqx.field(converter=cx.vector)
    r"""Conjugate momenta, e.g. CartesianVel3D.

    This is a 3-vector with a batch shape allowing for vector inputs.
    """

    _: KW_ONLY

    frame: SimulationFrame = eqx.field(
        default=simulation_frame,
        converter=Unless(
            cx.frames.AbstractReferenceFrame, cx.frames.TransformedReferenceFrame.from_
        ),
    )
    """The reference frame of the phase-space position."""

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = (True, True, False)  # q, p, frame

    # ==========================================================================
    # Coordinate API

    @classmethod
    def _dimensionality(cls) -> int:
        """Return the dimensionality of the phase-space position."""
        return 6  # TODO: should it be 6? Also make it a Final

    @override
    @property
    def data(self) -> cx.KinematicSpace:  # type: ignore[misc]
        """Return the data as a space.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        We can create a phase-space position:

        >>> pos = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
        ...                             p=u.Quantity([4, 5, 6], "km/s"))
        >>> pos.data
        KinematicSpace({ 'length': CartesianPos3D( ... ),
                         'speed': CartesianVel3D( ... ) })

        """
        return cx.KinematicSpace(length=self.q, speed=self.p)

    # ==========================================================================
    # Array properties

    @property
    @override
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch, component shape."""
        qbatch, qshape = vector_batched_shape(self.q)
        pbatch, pshape = vector_batched_shape(self.p)
        batch_shape = jnp.broadcast_shapes(qbatch, pbatch)
        return batch_shape, ComponentShapeTuple(q=qshape, p=pshape)


#####################################################################
# Dispatches

# ===============================================================
# Constructor


@AbstractPhaseSpaceObject.from_.dispatch  # type: ignore[attr-defined,misc]
def from_(
    cls: type[PhaseSpacePosition],
    data: cx.KinematicSpace,
    frame: cx.frames.AbstractReferenceFrame,
    /,
) -> PhaseSpacePosition:
    """Return a new PhaseSpacePosition from the given data and frame.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> data = cx.KinematicSpace(length=cx.CartesianPos3D.from_([1, 2, 3], "kpc"),
    ...                 speed=cx.CartesianVel3D.from_([4, 5, 6], "km/s"))
    >>> frame = gc.frames.simulation_frame

    >>> gc.PhaseSpacePosition.from_(data, frame)
    PhaseSpacePosition(
      q=CartesianPos3D( ... ),
      p=CartesianVel3D( ... ),
      frame=SimulationFrame()
    )

    """
    q = data["length"]
    if isinstance(q, cx.vecs.FourVector):
        warnings.warn("taking the 3-vector part of a 4-vector", stacklevel=2)
        q = q.q

    return cls(q=q, p=data["speed"], frame=frame)


# ===============================================================
# `coordinax.vconvert` dispatches


@dispatch
def vconvert(
    target: PSPVConvertOptions,
    psp: PhaseSpacePosition,
    /,
    **kwargs: Any,
) -> PhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.vecs as cxv
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                            p=u.Quantity([4, 5, 6], "km/s"))
    >>> w.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    Converting it to a different representation and differential class:

    >>> cx.vconvert({"q": cxv.LonLatSphericalPos, "p": cxv.LonCosLatSphericalVel}, w)
    PhaseSpacePosition( q=LonLatSphericalPos( ... ),
                        p=LonCosLatSphericalVel( ... ),
                        frame=SimulationFrame() )

    """
    q_cls = target["q"]
    p_cls = q_cls.time_derivative_cls if (mayp := target.get("p")) is None else mayp
    return replace(
        psp,
        q=psp.q.vconvert(q_cls, **kwargs),
        p=psp.p.vconvert(p_cls, psp.q, **kwargs),
    )


@dispatch
def vconvert(
    target_position_cls: type[cx.vecs.AbstractPos],
    psp: PhaseSpacePosition,
    /,
    **kwargs: Any,
) -> PhaseSpacePosition:
    """Convert the phase-space position to a different representation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We can create a phase-space position and convert it to a 6-vector:

    >>> psp = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"))
    >>> psp.w(units="galactic")
    Array([1. , 2. , 3. , 0.00409085, 0.00511356, 0.00613627], dtype=float64, ...)

    Converting it to a different representation:

    >>> cx.vconvert(cx.vecs.CylindricalPos, psp)
    PhaseSpacePosition( q=CylindricalPos( ... ),
                        p=CylindricalVel( ... ),
                        frame=SimulationFrame() )

    If the new representation requires keyword arguments, they can be passed
    through:

    >>> cx.vconvert(cx.vecs.ProlateSpheroidalPos, psp, Delta=u.Quantity(2.0, "kpc"))
    PhaseSpacePosition( q=ProlateSpheroidalPos(...),
                        p=ProlateSpheroidalVel(...),
                        frame=SimulationFrame() )

    """
    target = {"q": target_position_cls, "p": target_position_cls.time_derivative_cls}
    return vconvert(target, psp, **kwargs)


# ===============================================================
# `unxt.uconvert` dispatches


@dispatch(precedence=1)  # type: ignore[call-overload, misc]  # TODO: make precedence=0
def uconvert(
    units: u.AbstractUnitSystem | str, psp: PhaseSpacePosition
) -> PhaseSpacePosition:
    """Convert the components to the given units."""
    usys = u.unitsystem(units)
    return replace(psp, q=psp.q.uconvert(usys), p=psp.p.uconvert(usys))
