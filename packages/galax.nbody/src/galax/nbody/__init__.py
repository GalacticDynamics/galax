""":mod:`galax.nbody`."""

__all__ = [
    "NBodyField",
    "Component",
    "System",
    "Particles",
    "DirectSum",
    "PlummerKernel",
    "AbstractSofteningKernel",
    "AbstractForceBackend",
    "Sources",
    "Targets",
    "AbstractGradPolicy",
    "Autodiff",
    "StopGrad",
    "NoGrad",
    "total_energy",
]

from .setup_package import install_import_hook

with install_import_hook("galax.nbody"):
    from ._src.backend import AbstractForceBackend
    from ._src.diagnostics import total_energy
    from ._src.direct import DirectSum
    from ._src.field import NBodyField
    from ._src.gradpolicy import AbstractGradPolicy, Autodiff, NoGrad, StopGrad
    from ._src.kernels import AbstractSofteningKernel, PlummerKernel
    from ._src.particles import Sources, Targets
    from ._src.system import Component, Particles, System

# Cleanup
del install_import_hook
