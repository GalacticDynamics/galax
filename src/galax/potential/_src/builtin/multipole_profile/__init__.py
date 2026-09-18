"""Multipole profile expansion. Private API.

See the public API in `galax.potential` and `galax.potential.multipole_profile`.

Mode-ordering convention
------------------------
Radial coefficient arrays (``phi_lm``, ``dphi_lm``, etc.) are stored with
shape ``(n_r, n_modes)`` — radial axis first, mode axis second (mode-minor).
This contrasts with ``real_ylm``, which returns shape ``(n_modes, *batch)``
(mode-major). The `_log_r_and_ylm` function reconciles them via
``moveaxis(0, -1)`` when computing the expansion.
"""

__all__ = [
    "AbstractMultipoleProfilePotential",
    "MultipoleProfilePotential",
    "angular_grid",
    "build_expansion",
    "default_angular_resolution",
    "expansion_density",
    "expansion_potential",
    "lm_keys",
    "project_density",
    "radial_grid",
    "real_ylm",
    "solve_poisson_lm",
]

from .core import AbstractMultipoleProfilePotential, MultipoleProfilePotential
from .expansion import expansion_density, expansion_potential
from .funcs import build_expansion, radial_grid
from .poisson import solve_poisson_lm
from .project import (
    angular_grid,
    default_angular_resolution,
    lm_keys,
    project_density,
    real_ylm,
)
