"""Built-in Potential classes.

This module is private API.
See the public API in `galax.potential`.

"""

# The `# type: ignore`s: the star imports also carry each module's function set,
# and `mass_enclosed` is a `jax.jit` function in some and a kernel-factory
# result in others. Nothing reads these names from here (see `galax.potential.<name>`).
from .burkert import *
from .gaussian import *
from .harmonicoscillator import (
    HarmonicOscillatorPotential as HarmonicOscillatorPotential,
)
from .henonheiles import HenonHeilesPotential as HenonHeilesPotential
from .hernquist import *  # type: ignore[assignment]
from .isochrone import *  # type: ignore[assignment]
from .jaffe import *  # type: ignore[assignment]
from .kepler import *  # type: ignore[assignment]
from .kuzmin import *
from .kuzminkutuzov import *
from .lmj09logarithmic import LMJ09LogarithmicPotential as LMJ09LogarithmicPotential
from .logarithmic import LogarithmicPotential as LogarithmicPotential
from .longmurali import *
from .milkyway import *
from .miyamotonagai import *
from .mn3 import *
from .monari2016 import *
from .multipole import *

# Not `import *`: `multipole_profile.__all__` also carries the projection,
# Poisson, build and evaluation functions, which are not potentials. They stay
# reachable through the public `galax.potential.multipole_profile`.
from .multipole_profile import (
    AbstractMultipoleProfilePotential as AbstractMultipoleProfilePotential,
    MultipoleProfilePotential as MultipoleProfilePotential,
)
from .nfw import *
from .null import NullPotential as NullPotential
from .plummer import *
from .powerlawcutoff import *
from .satoh import *

# Not `import *`: `scf.__all__` also carries the basis functions and the
# coefficient fitter, which are not potentials. They stay reachable through
# the public `galax.potential.scf`.
from .scf import SCFPotential as SCFPotential
from .sph_expansion import (
    AbstractRadialBasisExpansionPotential as AbstractRadialBasisExpansionPotential,
    AbstractSphericalHarmonicExpansionPotential as AbstractSphericalHarmonicExpansionPotential,  # noqa: E501
)
from .stoneostriker15 import *
from .tabulated_basis import *
from .triaxialhernquist import (
    TriaxialHernquistPotential as TriaxialHernquistPotential,
)
from .zhao import *
