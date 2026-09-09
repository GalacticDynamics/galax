"""Built-in Potential classes.

This module is private API.
See the public API in `galax.potential`.

"""

from .burkert import *
from .example import *
from .gaussian import *
from .hernquist import *
from .isochrone import *
from .jaffe import *
from .kepler import *
from .kuzmin import *
from .logarithmic import *
from .longmurali import *
from .milkyway import *
from .miyamotonagai import *
from .mn3 import *
from .monari2016 import *
from .multipole import *
from .nfw import *
from .null import *
from .plummer import *
from .powerlawcutoff import *
from .satoh import *

# Not `import *`: `scf.__all__` also carries the basis functions and the
# coefficient fitter, which are not potentials. They stay reachable through
# the public `galax.potential.scf`.
from .scf import SCFPotential as SCFPotential
from .stoneostriker15 import *
from .zhao import *
