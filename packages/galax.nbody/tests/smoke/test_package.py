"""Testing :mod:`galax.nbody` module."""

import galax.nbody as gn


def test_all() -> None:
    """Test the `galax.nbody` API."""
    assert set(gn.__all__) == {
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
        "evolve",
        "NBodyResult",
        "check_timestep",
    }
    for name in gn.__all__:
        assert hasattr(gn, name)
