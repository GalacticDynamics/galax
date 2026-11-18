""":mod:`galax.dynamics.nbody`, re-exporting :mod:`galax.nbody`."""

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
]

from .setup_package import install_import_hook

with install_import_hook("galax.dynamics.nbody"):
    from galax.nbody import (
        AbstractForceBackend,
        AbstractGradPolicy,
        AbstractSofteningKernel,
        Autodiff,
        Component,
        DirectSum,
        NBodyField,
        NoGrad,
        Particles,
        PlummerKernel,
        Sources,
        StopGrad,
        System,
        Targets,
    )

# Cleanup
del install_import_hook
