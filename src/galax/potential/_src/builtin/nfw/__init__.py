"""NFW-like potentials."""

__all__ = [
    "NFWPotential",
    "LeeSutoTriaxialNFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
]

from .base import NFWPotential
from .leesuto import LeeSutoTriaxialNFWPotential
from .triaxial import TriaxialNFWPotential
from .vogelsburger08 import Vogelsberger08TriaxialNFWPotential
