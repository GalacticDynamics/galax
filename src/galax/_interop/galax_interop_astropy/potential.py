"""Compatibility."""

__all__: list[str] = []

from typing import Any

from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as APYQuantity
from jaxtyping import Real
from plum import convert, dispatch

import coordinax as cx
from unxt.quantity import BareQuantity as FastQ

import galax._custom_types as gt
import galax.potential as gp
from galax.potential._src.utils import coord_dispatcher

# =============================================================================


@coord_dispatcher(precedence=1)  # 1, b/c of ndarray subclassing
def parse_to_xyz_t(
    to_frame: cx.frames.AbstractReferenceFrame | None,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
    **kw: Any,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    return parse_to_xyz_t(to_frame, convert(xyz, FastQ), convert(t, FastQ), **kw)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cx.frames.AbstractReferenceFrame | None,
    q: BaseRepresentation,
    t: Any,
    /,
    **kw: Any,
) -> tuple[gt.BBtQuSz3, gt.BBtQuSz0]:
    """Parse input arguments to position & time."""
    cart = convert(q, cx.CartesianPos3D)
    return parse_to_xyz_t(to_frame, cart, convert(t, FastQ), **kw)


# =============================================================================


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def potential(
    pot: gp.AbstractPotential,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
) -> gt.BBtQuSz0:
    """Compute the potential energy at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return gp.potential(pot, convert(xyz, FastQ), convert(t, FastQ))


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def gradient(
    pot: gp.AbstractPotential,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
) -> gt.BBtQuSz3:
    """Compute the gradient at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return gp.gradient(pot, convert(xyz, FastQ), convert(t, FastQ))


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def density(
    pot: gp.AbstractPotential,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
) -> gt.BBtQuSz0:
    """Compute the density at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return gp.density(pot, convert(xyz, FastQ), convert(t, FastQ))


@dispatch(precedence=1)  # type: ignore[call-overload,misc]
def hessian(
    pot: gp.AbstractPotential,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
) -> gt.BBtQuSz3:
    """Compute the density at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return gp.hessian(pot, convert(xyz, FastQ), convert(t, FastQ))
