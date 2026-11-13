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

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree as jtu
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
_SolverState: TypeAlias = VF | None  # the force g(t0, q0) for the next step

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

    The force ``g(t1, x1)`` ending one step is the force starting the next, so
    it is kept in the solver state and each step costs one evaluation of ``g``.
    After a jump (``jump_ts``, or ``made_jump=True`` passed to
    `diffrax.diffeqsolve`) it is evaluated afresh. When resuming from a saved
    ``solver_state`` after an event or a change to ``y``, pass
    ``made_jump=True``.
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
        _, g = terms
        return g.vf(t0, y0[0], args)

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
        f, g = terms
        q0, p0 = y0

        # The force at (t0, q0), from the end of the previous step, or afresh
        # if the caller kept no solver state (`galax.dynamics.OrbitSolver.run`
        # stores none). diffrax resolves `made_jump` to a static False unless
        # jumps are possible.
        a0 = g.vf(t0, q0, args) if solver_state is None else solver_state
        if made_jump is not False and made_jump is not None:

            def recompute() -> VF:
                a_new = g.vf(t0, q0, args)
                return jtu.map(lambda x, y: jnp.where(made_jump, x, y), a_new, a0)

            # `unvmap_any`: under vmap, only evaluate if some lane jumped.
            a0 = jax.lax.cond(eqxi.unvmap_any(made_jump), recompute, lambda: a0)

        # Use the terms' control, not ``t1 - t0``: for ``t1 < t0`` diffrax
        # flips time and wraps the terms, and only ``contr`` carries the sign
        # back (GalacticDynamics/galax#963).
        cf, cg = f.contr(t0, t1), g.contr(t0, t1)
        half_cg = (0.5 * cg**ω).ω

        v_half = (p0**ω + g.prod(a0, half_cg) ** ω).ω
        q1 = (q0**ω + f.vf_prod(t0, v_half, args, cf) ** ω).ω
        a1 = g.vf(t1, q1, args)
        p1 = (v_half**ω + g.prod(a1, half_cg) ** ω).ω

        y1 = (q1, p1)
        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, a1, RESULTS.successful

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
