""":mod:`galax.dynamics.mockstream`."""

__all__ = [
    # Legacy
    "MockStreamGenerator",
    "MockStreamArm",
    "MockStream",
    "AbstractStreamDF",
    "FardalStreamDF",
    "ChenStreamDF",
    "ProgenitorMassCallable",
    "ConstantMassProtenitor",
]

from jaxtyping import install_import_hook

from galax.setup_package import RUNTIME_TYPECHECKER

with install_import_hook("galax.dynamics.dynamics", RUNTIME_TYPECHECKER):
    from ._src.legacy.mockstream import MockStreamGenerator
    from ._src.legacy.mockstream.df import (
        AbstractStreamDF,
        ChenStreamDF,
        ConstantMassProtenitor,
        FardalStreamDF,
        ProgenitorMassCallable,
    )
    from ._src.mockstream import MockStream, MockStreamArm

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER
