"""Interoperability."""

__all__ = ["gala_to_galax"]


from typing import TYPE_CHECKING

from galax.potential._potential.base import AbstractPotentialBase

if TYPE_CHECKING:
    from gala.potential import PotentialBase as GalaPotentialBase


def gala_to_galax(pot: "GalaPotentialBase", /) -> AbstractPotentialBase:
    """Convert a :mod:`gala` potential to a :mod:`galax` potential.

    Parameters
    ----------
    pot :  :class:`~gala.potential.PotentialBase`
        :mod:`gala` potential.

    Returns
    -------
    gala_pot : :class:`~galax.potential.AbstractPotentialBase`
        :mod:`galax` potential.
    """
    msg = "The `gala` package must be installed to use this function. "
    raise ImportError(msg)
