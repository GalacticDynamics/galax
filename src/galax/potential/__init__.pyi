# TODO: use star imports when
# https://github.com/scientific-python/lazy_loader/issues/94 is resolved

__all__ = [
    # Modules
    "io",
    # base
    "AbstractPotentialBase",
    # core
    "AbstractPotential",
    # composite
    "AbstractCompositePotential",
    "CompositePotential",
    # param
    "ParametersAttribute",
    "ParameterCallable",
    "AbstractParameter",
    "ConstantParameter",
    "UserParameter",
    "ParameterField",
    # builtin
    "BarPotential",
    "HernquistPotential",
    "IsochronePotential",
    "KeplerPotential",
    "LeeSutoTriaxialNFWPotential",
    "MiyamotoNagaiPotential",
    "NFWPotential",
    "NullPotential",
    "TriaxialHernquistPotential",
    # special
    "MilkyWayPotential",
    # frame
    "PotentialFrame",
]

from ._potential import io
from ._potential.base import AbstractPotentialBase
from ._potential.builtin import (
    BarPotential,
    HernquistPotential,
    IsochronePotential,
    KeplerPotential,
    LeeSutoTriaxialNFWPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    NullPotential,
    TriaxialHernquistPotential,
)
from ._potential.composite import AbstractCompositePotential, CompositePotential
from ._potential.core import AbstractPotential
from ._potential.frame import PotentialFrame
from ._potential.param import (
    AbstractParameter,
    ConstantParameter,
    ParameterCallable,
    ParameterField,
    ParametersAttribute,
    UserParameter,
)
from ._potential.special import MilkyWayPotential
