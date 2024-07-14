""":mod:`galax` <-> :mod:`gala` interoperability."""

__all__: list[str] = ["galpy_to_galax", "galax_to_galpy"]

from .potential import galax_to_galpy, galpy_to_galax
