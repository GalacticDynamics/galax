""":mod:`galax.nbody`."""

__all__ = [
    "NBodyField",
    "Component",
    "System",
    "Particles",
    "DirectSum",
    "PallasDirectSum",
    "PlummerKernel",
    "NewtonianKernel",
    "DehnenK1Kernel",
    "DehnenK2Kernel",
    "SplineKernel",
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

with install_import_hook("galax.nbody"):
    from ._src.backend import AbstractForceBackend, Sources, Targets
    from ._src.direct import DirectSum
    from ._src.field import NBodyField
    from ._src.gradpolicy import AbstractGradPolicy, Autodiff, NoGrad, StopGrad
    from ._src.kernels import (
        AbstractSofteningKernel,
        DehnenK1Kernel,
        DehnenK2Kernel,
        NewtonianKernel,
        PlummerKernel,
        SplineKernel,
    )
    from ._src.pallas_direct import PallasDirectSum
    from ._src.system import Component, Particles, System

# Cleanup
del install_import_hook
