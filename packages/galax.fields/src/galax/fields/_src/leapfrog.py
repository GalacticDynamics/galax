# ruff: noqa: ARG002
"""Leapfrog (velocity Verlet) solver for diffrax.

diffrax has no second-order symplectic solver. patrick-kidger/diffrax#303 adds
one (Störmer-Verlet) but has stalled, so galax implements it here.
"""

__all__ = ["Leapfrog"]

from collections.abc import Callable
from jaxtyping import ArrayLike, Float, PyTree
from typing import Any, ClassVar, TypeAlias

import equinox.internal as eqxi
import jax
from diffrax import RESULTS, AbstractSolver, AbstractTerm, LocalLinearInterpolation
from equinox.internal import ω  # noqa: PLC2403

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

    This is a 2nd order symplectic integration method, also known as
    kick-drift-kick leapfrog or velocity Verlet. Like `diffrax.SemiImplicitEuler`
    it takes two terms and a state ``y = (q, p)``, the first term giving ``dq/dt``
    from ``p`` and the second ``dp/dt`` from ``q``.

    It provides no error estimate, so it does not support adaptive step sizing:
    pair it with `diffrax.ConstantStepSize`. An adaptive controller such as
    `diffrax.PIDController` raises a `RuntimeError` for a solver without error
    estimates.

    Assuming that:

        q0, p0 = y0

    and:

        f, g = terms

    This method computes the next step as:

        p_half = p0 + h/2 * g(t0, q0)
        q1 = q0 + h * f(t0, p_half)
        p1 = p_half + h/2 * g(t1, q1)

    Each step reuses the previous step's end force ``g(t1, q1)``, so it costs one
    evaluation of ``g``. After a jump in ``jump_ts`` the force is evaluated
    afresh. When resuming a solve from a saved ``solver_state`` after an event or
    a change to ``y``, pass ``made_jump=jnp.array(True)``: with
    ``ConstantStepSize`` a Python ``True`` stays static, so every step would
    evaluate the force afresh.

    Examples
    --------
    A harmonic oscillator, ``dq/dt = p`` and ``dp/dt = -q``, over one period:

    >>> import diffrax as dfx
    >>> import jax.numpy as jnp
    >>> import galax.fields as gf

    >>> terms = (dfx.ODETerm(lambda t, p, args: p), dfx.ODETerm(lambda t, q, args: -q))
    >>> sol = dfx.diffeqsolve(terms, gf.Leapfrog(), t0=0.0, t1=2 * jnp.pi, dt0=0.01,
    ...                       y0=(jnp.array(1.0), jnp.array(0.0)))
    >>> q1, p1 = sol.ys
    >>> bool(jnp.allclose(q1, 1.0, atol=1e-4)), bool(jnp.allclose(p1, 0.0, atol=1e-2))
    (True, True)

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

        # Force at (t0, q0): stored from the last step, or afresh if none.
        # `made_jump` is a static False unless jumps are possible or the caller
        # passed a Python bool.
        a0 = g.vf(t0, q0, args) if solver_state is None else solver_state
        if made_jump is not False:
            # `unvmap_any`: under vmap, only evaluate if some lane jumped; then
            # every lane takes the fresh force (equal to the stored one when the
            # state is consistent).
            a0 = jax.lax.cond(
                eqxi.unvmap_any(made_jump), lambda: g.vf(t0, q0, args), lambda: a0
            )

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
