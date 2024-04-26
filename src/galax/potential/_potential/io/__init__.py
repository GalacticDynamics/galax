"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__, submod_attrs={"_gala": ["gala_to_galax"]}
)
