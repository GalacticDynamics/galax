"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

from lazy_loader import attach_stub

__getattr__, __dir__, __all__ = attach_stub(__name__, __file__)
