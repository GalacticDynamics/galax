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
    }
    for name in gn.__all__:
        assert hasattr(gn, name)
