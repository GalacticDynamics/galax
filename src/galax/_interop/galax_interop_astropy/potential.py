"""Compatibility."""

__all__: list[str] = []

from typing import Any

from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as APYQuantity
from plum import convert, dispatch

import coordinax as cx
from unxt.quantity import BareQuantity as FastQ

import galax.typing as gt

# =============================================================================
# parse_to_quantity


@dispatch
def parse_to_quantity(value: APYQuantity, /, **kw: Any) -> gt.BtRealQuSz3:
    q = convert(value, FastQ)
    return parse_to_quantity(q, **kw)


@dispatch
def parse_to_quantity(rep: BaseRepresentation, /, **kw: Any) -> gt.BtRealQuSz3:
    cart = convert(rep, cx.CartesianPos3D)
    return parse_to_quantity(cart, **kw)
