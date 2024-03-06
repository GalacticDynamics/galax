"""Input/output/conversion of potential objects.

This module contains the machinery for I/O and conversion of potential objects.
Conversion is useful for e.g. converting a
:class:`galax.potential.AbstractPotential` object to a
:class:`gala.potential.PotentialBase` object.
"""

__all__: list[str] = ["gala_to_galax"]


from galax.utils._optional_deps import HAS_GALA

if HAS_GALA:
    from ._gala import gala_to_galax
else:
    from ._gala_noop import gala_to_galax  # type: ignore[assignment]
