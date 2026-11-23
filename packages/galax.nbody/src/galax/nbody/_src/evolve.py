"""Run an N-Body simulation on a fixed step grid."""

__all__ = ["check_timestep", "evolve"]

import warnings

from jaxtyping import Array, Real
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import unxt as u
from unxt.quantity import AllowValue

import galax.fields as gf
import galax.potential as gp
from .field import NBodyField


def check_timestep(
    field: NBodyField,
    q: Any,
    t: Any,
    dt: Any,
    *,
    eta: float = 0.025,
    eta_rho: float = 0.1,
    strict: bool = False,
    stacklevel: int = 2,
) -> float:
    r"""Return the largest safe time step ``dt_max``, warning if ``dt`` exceeds it.

    ``dt_max`` is the smaller of two limits. For active particles with
    softening ``h > 0`` it is ``eta * sqrt(2 h / |a|)``, with ``a`` the
    acceleration. For all active particles it is
    ``eta_rho / sqrt(G rho_ext)``, with ``rho_ext`` the external density;
    particles where that density is not positive are skipped. With no limit
    at all ``dt_max`` is infinite. An active particle with a non-finite
    position, or a softened one with a NaN acceleration, makes ``dt_max`` NaN.

    Raises `ValueError` if ``dt`` is a number that is not a finite scalar,
    and `TypeError` if it is not a real number. A negative ``dt``, for a
    backward run, is checked by its size. If ``|dt| > dt_max``, or ``dt_max``
    is NaN, this warns with `UserWarning`, or
    raises `ValueError` when ``strict``. ``stacklevel`` is passed to
    `warnings.warn`; the default points the warning at the caller. It needs
    concrete values, so call it outside ``jit``.
    """
    units = field.units
    q = jnp.asarray(u.ustrip(AllowValue, units["length"], q), dtype=float)
    t = u.ustrip(AllowValue, units["time"], t)
    dt_np = np.asarray(u.ustrip(AllowValue, units["time"], dt))
    if dt_np.ndim != 0 or not np.isfinite(dt_np):
        msg = f"dt must be a finite scalar, got {dt_np}"
        raise ValueError(msg)
    dt = abs(float(dt_np))
    a = jnp.linalg.norm(field.acceleration(t, q), axis=-1)
    h = field.softening
    rho = jnp.asarray(gp.density(field.external_potential, q, t))
    soft = jnp.where(field.active & (h > 0), eta * jnp.sqrt(2 * h / a), jnp.inf)
    G = field._G  # noqa: SLF001
    tidal = jnp.where(field.active & (rho > 0), eta_rho / jnp.sqrt(G * rho), jnp.inf)
    bad = field.active & ~jnp.all(jnp.isfinite(q), axis=-1)
    limit = jnp.where(bad, jnp.nan, jnp.minimum(soft, tidal))
    dt_max = float(jnp.min(limit, initial=jnp.inf))
    if not dt <= dt_max:  # also catches a NaN dt_max
        msg = (
            f"the safe time step dt_max is NaN (dt={dt}); "
            "a position or acceleration is NaN or inf"
            if np.isnan(dt_max)
            else f"dt={dt} is above the safe time step dt_max={dt_max}"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=stacklevel)
    return dt_max


def evolve(
    field: NBodyField,
    q0: Any,
    p0: Any,
    t0: Any,
    dt: Any,
    n_steps: int | np.integer,
    *,
    save_every: int | np.integer | None = None,
    solver: dfx.AbstractSolver | None = None,
    adjoint: dfx.AbstractAdjoint | None = None,
) -> dfx.Solution:
    r"""Integrate ``field`` for ``n_steps`` fixed steps of ``dt`` from ``t0``.

    ``q0`` and ``p0`` are the (N, 3) positions and velocities at ``t0``. They,
    ``t0`` and ``dt`` may be arrays or Quantities, converted to the field's
    units. ``t0`` and ``dt`` can be traced, so `jax.jit`, `jax.vmap` and
    `jax.grad` work over them as well as over the field and the initial state.
    ``n_steps`` and ``save_every`` are Python ints.

    The steps end at ``t0 + k * dt`` for ``k = 1 .. n_steps``. The state is
    saved at ``t0`` and after every ``save_every`` steps, by default only at
    the end; ``n_steps`` must be a multiple of ``save_every``. The result is a
    `diffrax.Solution`: ``sol.ts`` holds the S save times and ``sol.ys`` the
    ``(q, p)`` pair of (S, N, 3) arrays. ``gn.total_energy(field, sol)`` gives
    the energy at each save, and ``system.split(sol.ys[0], axis=-2)`` splits
    the positions by component.

    ``solver`` defaults to `galax.fields.Leapfrog`. Any explicit diffrax solver
    works on the grid; a solver with an error estimate just takes the fixed
    steps. An implicit solver needs its root finder's tolerances set, since no
    step-size controller supplies them. Saves after ``t0`` are the solver's
    interpolant at the step end, equal to the step's state up to rounding.
    ``adjoint`` defaults to `diffrax.RecursiveCheckpointAdjoint`, which needs
    about ``sqrt(2 n_steps)`` checkpoints of the state; `evolve` sets
    ``max_steps=n_steps`` so the checkpoint count follows the run. Forward-mode
    derivatives (`jax.jvp`, `jax.jacfwd`) need ``adjoint=diffrax.ForwardMode()``.
    `diffrax.RecursiveCheckpointAdjoint` with ``checkpoints=n_steps`` makes the
    gradient faster, from about 1.3 times at large N to several times at small
    N, for memory that grows linearly with ``n_steps``.

    `evolve` is the fixed-step path. Adaptive steps, events and other step
    controllers go through diffrax directly, for example
    ``diffraxtra.DiffEqSolver(diffrax.Dopri8(),
    stepsize_controller=diffrax.PIDController(...), max_steps=...)(field, t0,
    t1, None, (q0, p0), saveat=...)``, or `galax.orbits.OrbitSolver`. Set
    ``max_steps`` there to about the number of steps you expect: with the
    checkpoint adjoint it sets the gradient memory.

    A negative ``dt`` runs backward in time. Flipping the velocities and
    running forward is the same only when the external potential is static.

    A tracer that is NaN or inf from the start, or whose position becomes so,
    leaves the other particles' values and their gradients with respect to the
    field and the initial state unchanged, but makes gradients with respect
    to ``t0`` and ``dt`` NaN: the step size is shared by every row. A tracer that
    reaches a finite point where the external force or its derivative is not
    finite (outside a user potential's valid region, or exactly at a singular
    point) makes reverse-mode gradients NaN; forward results are unaffected.

    galax.nbody needs float64, so `evolve` raises `ValueError` unless
    ``jax_enable_x64`` is on. It also raises `ValueError` if ``n_steps`` is not
    a positive int, ``save_every`` is not a positive int dividing it, or
    ``t0`` or ``dt`` is not a scalar. A ``dt`` that is zero or not finite
    raises at run time, also when traced (`equinox.EquinoxRuntimeError`, or
    `jax.errors.JaxRuntimeError` under `jax.jit`). A ``dt`` so small that
    ``t0 + k * dt`` repeats, near large ``|t0|``, fails in diffrax with a
    "strictly increasing" error.

    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> import jax.numpy as jnp
    >>> import galax.nbody as gn
    >>> import galax.potential as gp
    >>> field = gn.NBodyField(jnp.full(2, 1e9), 0.1, gp.NullPotential(units="galactic"))
    >>> q0 = jnp.asarray([[-1.0, 0, 0], [1.0, 0, 0]])
    >>> sol = gn.evolve(field, q0, jnp.zeros((2, 3)), 0.0, 0.01, 100, save_every=25)
    >>> sol.ts
    Array([0.  , 0.25, 0.5 , 0.75, 1.  ], dtype=float64)
    >>> sol.ys[0].shape
    (5, 2, 3)
    """
    if not jax.config.read("jax_enable_x64"):  # type: ignore[no-untyped-call]
        msg = (
            "galax.nbody needs float64, but jax_enable_x64 is off. Turn it on "
            'with jax.config.update("jax_enable_x64", True) at the start of '
            "the program, or set JAX_ENABLE_X64=1."
        )
        raise ValueError(msg)
    save_every = n_steps if save_every is None else save_every
    counts: tuple[tuple[str, Any], ...] = (
        ("n_steps", n_steps),
        ("save_every", save_every),
    )
    for name, n in counts:
        if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
            msg = f"{name} must be a positive int, got {n!r}"
            raise ValueError(msg)
    if n_steps % save_every:
        msg = f"save_every={save_every} must divide n_steps={n_steps}"
        raise ValueError(msg)
    units = field.units
    y0 = (
        jnp.asarray(u.ustrip(AllowValue, units["length"], q0), dtype=float),
        jnp.asarray(u.ustrip(AllowValue, units["speed"], p0), dtype=float),
    )
    t0 = jnp.asarray(u.ustrip(AllowValue, units["time"], t0), dtype=float)
    dt = jnp.asarray(u.ustrip(AllowValue, units["time"], dt), dtype=float)
    if t0.ndim or dt.ndim:
        msg = f"t0 and dt must be scalars, got shapes {t0.shape} and {dt.shape}"
        raise ValueError(msg)
    dt = eqx.error_if(dt, ~jnp.isfinite(dt) | (dt == 0), "dt must be finite, not 0")
    solver = gf.Leapfrog() if solver is None else solver
    adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint
    return _solve(field, y0, t0, dt, int(n_steps), int(save_every), solver, adjoint)


@eqx.filter_jit
def _solve(
    field: NBodyField,
    y0: tuple[Array, Array],
    t0: Real[Array, ""],
    dt: Real[Array, ""],
    n_steps: int,
    save_every: int,
    solver: dfx.AbstractSolver,
    adjoint: dfx.AbstractAdjoint,
) -> dfx.Solution:
    # The field is an argument, not a closure, so eager calls with a new field
    # of the same shapes reuse the compilation.
    # The barrier makes XLA build the grid once. Otherwise it may fuse t0, t1
    # and the save times into their uses separately, with and without an fma,
    # and `StepTo` rejects a t1 that is off ts[-1] by one rounding.
    grid = jax.lax.optimization_barrier(t0 + dt * jnp.arange(n_steps + 1))  # type: ignore[no-untyped-call]
    return dfx.diffeqsolve(
        field.terms(solver),
        solver,
        grid[0],
        grid[-1],
        None,
        y0,
        stepsize_controller=dfx.StepTo(ts=grid),
        saveat=dfx.SaveAt(t0=True, ts=grid[save_every::save_every]),
        max_steps=n_steps,
        adjoint=adjoint,
    )
