""":mod:`galax.dynamics.mockstream`."""

__all__ = [
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
    from ._src.mockstream import MockStream, MockStreamArm, MockStreamGenerator
    from ._src.mockstream.df import (
        AbstractStreamDF,
        ChenStreamDF,
        ConstantMassProtenitor,
        FardalStreamDF,
        ProgenitorMassCallable,
    )

# Cleanup
del install_import_hook, RUNTIME_TYPECHECKER
