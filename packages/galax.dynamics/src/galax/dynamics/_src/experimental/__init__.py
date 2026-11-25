"""Experimental dynamics."""

from .df import (
    AbstractKinematicDF as AbstractKinematicDF,
    Chen2024DF as Chen2024DF,
    Fardal2015DF as Fardal2015DF,
)
from .schedule import Times as Times, Uniform as Uniform
from .simulator import (
    Copies as Copies,
    PerParticle as PerParticle,
    SharedTrack as SharedTrack,
    SprayResult as SprayResult,
    SpraySimulator as SpraySimulator,
    WholeStream as WholeStream,
    to_mockstream as to_mockstream,
)
from .stream import *

# `gd.experimental.integrate` is the module `galax.orbits._src.experimental.integrate`,
# kept for compatibility (#1047): the one allowed cross-distribution `_src` import.
from galax.orbits._src.experimental import integrate as integrate
from galax.orbits.experimental import (
    SymplecticSolverT as SymplecticSolverT,
    integrate_orbit as integrate_orbit,
)
