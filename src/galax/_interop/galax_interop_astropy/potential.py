"""Compatibility."""

__all__: list[str] = []

from typing import Any

from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as APYQuantity
from jaxtyping import Real
from plum import convert, dispatch

import coordinax as cx
from unxt.quantity import BareQuantity as FastQ

import galax.potential as gp
import galax.typing as gt
from galax.potential._src.utils import coord_dispatcher

# =============================================================================


@coord_dispatcher(precedence=1)  # 1, b/c of ndarray subclassing
def parse_to_xyz_t(
    to_frame: cx.frames.AbstractReferenceFrame | None,
    xyz: Real[APYQuantity, "*#batch 3"],
    t: Real[APYQuantity, "*#batch"],
    /,
    **kw: Any,
) -> tuple[gt.BBtRealQuSz3, gt.BBtRealQuSz0]:
    """Parse input arguments to position & time."""
    return parse_to_xyz_t(to_frame, convert(xyz, FastQ), convert(t, FastQ), **kw)


@coord_dispatcher
def parse_to_xyz_t(
    to_frame: cx.frames.AbstractReferenceFrame | None,
    q: BaseRepresentation,
    t: Any,
    /,
    **kw: Any,
) -> tuple[gt.BBtRealQuSz3, gt.BBtRealQuSz0]:
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
) -> gt.BBtRealQuSz0:
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
) -> gt.BBtRealQuSz3:
    """Compute the gradient at the given position(s).

    The position is in Cartesian coordinates and it and the time are assumed to
    be in the unit system of the potential.

    """
    return gp.gradient(pot, convert(xyz, FastQ), convert(t, FastQ))


# =============================================================================
# parse_to_quantity


@dispatch
def parse_to_quantity_or_array(value: APYQuantity, /, **kw: Any) -> gt.BtRealQuSz3:
    q = convert(value, FastQ)
    return parse_to_quantity_or_array(q, **kw)


@dispatch
def parse_to_quantity_or_array(rep: BaseRepresentation, /, **kw: Any) -> gt.BtRealQuSz3:
    cart = convert(rep, cx.CartesianPos3D)
    return parse_to_quantity_or_array(cart, **kw)


@dispatch
def parse_to_quantity(value: APYQuantity, /, **kw: Any) -> gt.BtRealQuSz3:
    q = convert(value, FastQ)
    return parse_to_quantity(q, **kw)


@dispatch
def parse_to_quantity(rep: BaseRepresentation, /, **kw: Any) -> gt.BtRealQuSz3:
    cart = convert(rep, cx.CartesianPos3D)
    return parse_to_quantity(cart, **kw)
