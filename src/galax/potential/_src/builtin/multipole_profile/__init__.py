"""Multipole profile expansion. Private API.

See the public API in `galax.potential` and `galax.potential.multipole_profile`.

The projection, Poisson and spline primitives this builds on live in
`galax.potential._src.harmonic`; only the potential layer is here.

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
    "build_expansion",
    "expansion_density",
    "expansion_potential",
]

from .build import build_expansion
from .core import AbstractMultipoleProfilePotential, MultipoleProfilePotential
from .expansion import expansion_density, expansion_potential
