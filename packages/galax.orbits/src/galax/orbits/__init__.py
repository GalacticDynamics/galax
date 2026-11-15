""":mod:`galax.orbits`."""

__all__ = [
    "AbstractOrbit",
    "AbstractOrbitField",
    "HamiltonianField",
    "Orbit",
    "OrbitSolver",
    "PhaseSpaceInterpolation",
    "compute_orbit",
]

from .setup_package import install_import_hook

with install_import_hook("galax.orbits"):
    from ._src.api import compute_orbit
    from ._src.base import AbstractOrbit
    from ._src.field_base import AbstractOrbitField
    from ._src.field_hamiltonian import HamiltonianField
    from ._src.interp import PhaseSpaceInterpolation
    from ._src.orbit import Orbit
    from ._src.solver import OrbitSolver

    # Register by import
    # isort: split
    from ._src import compute, register_dfx  # noqa: F401
    from ._src.plot_helper import ProxyAbstractOrbit

ProxyAbstractOrbit.deliver(AbstractOrbit)

# Cleanup
del install_import_hook, ProxyAbstractOrbit
