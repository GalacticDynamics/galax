# ruff: noqa: ARG002
"""
Note: This module implements a diffrax solver for Leapfrog integration. There is a
stalled PR to add a similar integrator to diffrax, so in the meantime we implement it
here.
"""  # noqa: D205

__all__ = ["Leapfrog", "SymplecticSolverT"]

from collections.abc import Callable
from jaxtyping import ArrayLike, Float, PyTree
from typing import Any, ClassVar, TypeAlias

from diffrax import (
    RESULTS,
    AbstractSolver,
    AbstractTerm,
    LocalLinearInterpolation,
    ODETerm,
    SemiImplicitEuler,
)
from equinox.internal import ω  # noqa: PLC2403

from galax.dynamics._src.orbit.field_base import AbstractOrbitField
from galax.dynamics._src.orbit.field_hamiltonian import HamiltonianField

# diffrax doesn't publicly export these; they're type hints only (no runtime
# behavior depends on them), so redeclare as `Any` rather than importing
# diffrax's private `_custom_types` module.
VF: TypeAlias = Any
Args: TypeAlias = Any
BoolScalarLike: TypeAlias = Any
DenseInfo: TypeAlias = Any
RealScalarLike: TypeAlias = Any

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]


class Leapfrog(AbstractSolver):  # type: ignore[misc]
    """Leapfrog (velocity Verlet) symplectic integrator.

    This is a 2nd order symplectic integration method. This integrator does not support
    adaptive step sizing: it provides no error estimate, so pair it with
    `diffrax.ConstantStepSize` (e.g. via ``gd.OrbitSolver(Leapfrog(),
    stepsize_controller=dfx.ConstantStepSize())``) rather than
    `galax.dynamics.OrbitSolver`'s default adaptive `diffrax.PIDController`, which will
    raise a `RuntimeError` for a solver without error estimates. This is either known as
    kick-drift-kick leapfrog or velocity Verlet.

    Assuming that:

        x0, v0 = y0

    and:

        f, g = terms

    This method computes the next step as:

        v_half = v0 + h/2 * g(t0, x0)
        x1 = x0 + h * f(t0, v_half)
        v1 = v_half + h/2 * g(t1, x1)
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, _: Any) -> int:
        return 2

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        f, g = terms
        q0, p0 = y0
        h = t1 - t0

        v_half = (p0**ω + 0.5 * h * g.vf(t0, q0, args) ** ω).ω
        q1 = (q0**ω + h * f.vf(t0, v_half, args) ** ω).ω
        p1 = (v_half**ω + 0.5 * h * g.vf(t1, q1, args) ** ω).ω

        y1 = (q1, p1)
        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        f, g = terms
        q0, p0 = y0
        qdot = f.vf(t0, p0, args)
        pdot = g.vf(t0, q0, args)
        return qdot, pdot


Leapfrog.__init__.__doc__ = """**Arguments:** None"""


SymplecticSolverT: TypeAlias = Leapfrog | SemiImplicitEuler


# ===============================================
# Terms dispatch
#
# Registered here (rather than in `field_hamiltonian.py`) so that core orbit
# code doesn't need to import the experimental package just to support
# `Leapfrog`. See `field_hamiltonian.py` for the `SemiImplicitEuler` dispatch.


@AbstractOrbitField.terms.dispatch
def terms(self: HamiltonianField, _: Leapfrog, /) -> tuple[ODETerm, ODETerm]:
    """Return the AbstractTerm terms for the Leapfrog solver."""
    return (ODETerm(self.dx_dt), ODETerm(self.dv_dt))
