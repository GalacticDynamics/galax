"""Run an N-Body simulation with the leapfrog solver."""

__all__ = ["NBodyResult", "check_timestep", "evolve"]

import warnings

from jaxtyping import Array, Float
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
from .diagnostics import total_energy
from .field import NBodyField
from .system import System

# A save time may differ from its grid point by this many float spacings.
# Measured on random cases: ts built as t0 + k*dt, or with np.linspace, after
# a unit conversion through unxt (Gyr, kyr, yr or s to Myr, with dt converted
# separately) were off by at most 4 spacings; without a conversion t0 + k*dt is
# exact and np.linspace is off by at most 2. Twice the worst gives 8. It is far
# below the 100 spacings an off-grid time is checked against in the tests.
_GRID_ULPS = 8


class NBodyResult(eqx.Module):
    """The saved states of an N-Body run.

    ``ts`` (S,) are the save times, and ``q`` and ``p`` (S, N, 3) are the
    positions and velocities there, all plain arrays in ``units``. ``energy``
    (S,) is the total energy, or None if it was not asked for. ``system`` is
    the `System` the run was built from, if one was given.
    """

    ts: Float[Array, " S"]
    q: Float[Array, "S N 3"]
    p: Float[Array, "S N 3"]
    energy: Float[Array, " S"] | None
    units: u.AbstractUnitSystem = eqx.field(static=True)
    system: System | None = None

    def component(self, name: str) -> tuple[Array, Array]:
        """Return the (..., S, n, 3) positions and velocities of component ``name``.

        The split is on the particle axis, the second to last, so a result
        with leading batch axes, for example from `jax.vmap`, works too.
        Raises `ValueError` if the result has no ``system`` and `KeyError` if
        ``name`` is not one of its components.
        """
        if self.system is None:
            msg = "this result has no system; pass system= to evolve"
            raise ValueError(msg)
        a, b = dict(zip(self.system.names, self.system.slices, strict=True))[name]
        return self.q[..., a:b, :], self.p[..., a:b, :]


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

    Raises `ValueError` if ``dt`` is a number that is not a finite,
    non-negative scalar, and `TypeError` if it is not a real number. If
    ``dt > dt_max``, or ``dt_max`` is NaN, this warns with `UserWarning`, or
    raises `ValueError` when ``strict``. ``stacklevel`` is passed to
    `warnings.warn`; the default points the warning at the caller. It needs
    concrete values, so call it outside ``jit``.
    """
    units = field.units
    q = jnp.asarray(u.ustrip(AllowValue, units["length"], q), dtype=float)
    t = u.ustrip(AllowValue, units["time"], t)
    dt_np = np.asarray(u.ustrip(AllowValue, units["time"], dt))
    if dt_np.ndim != 0 or not (np.isfinite(dt_np) and dt_np >= 0):
        msg = f"dt must be a finite, non-negative scalar, got {dt_np}"
        raise ValueError(msg)
    dt = float(dt_np)
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


@eqx.filter_jit
def advance(
    field: NBodyField,
    y: tuple[Array, Array],
    step_ts: Float[Array, " steps"],
    adjoint: dfx.AbstractAdjoint,
    save_ts: Float[Array, " S"] | None = None,
) -> tuple[Array, Array]:
    """Return ``(q, p)`` after leapfrog steps from ``step_ts[0]`` along ``step_ts``.

    With ``save_ts`` it returns the (S, N, 3) states at those times instead,
    which must be points of ``step_ts``. The step and save counts come from the
    shapes, so this compiles once per distinct pair of them.
    """
    terms = (
        dfx.ODETerm(lambda t, p, a: p),  # noqa: ARG005
        dfx.ODETerm(lambda t, q, a: field.acceleration(t, q)),  # noqa: ARG005
    )
    # `StepTo` sets the steps itself and requires `dt0=None`.
    sol = dfx.diffeqsolve(
        terms,
        gf.Leapfrog(),
        step_ts[0],
        step_ts[-1],
        None,
        y,
        stepsize_controller=dfx.StepTo(ts=step_ts),
        max_steps=step_ts.shape[0] - 1,
        saveat=dfx.SaveAt(t1=True) if save_ts is None else dfx.SaveAt(ts=save_ts),
        adjoint=adjoint,
    )
    q, p = sol.ys
    return (q[0], p[0]) if save_ts is None else (q, p)


def _integrate(
    field: NBodyField,
    y: tuple[Array, Array],
    t_start: float,
    dt: float,
    n_steps: np.ndarray,
    adjoint: dfx.AbstractAdjoint,
) -> tuple[Array, Array]:
    """Return the (S, N, 3) ``(q, p)``: ``y``, then the state after each segment.

    The whole run is one `diffrax.diffeqsolve` over the step grid, saving at
    the segment ends, so the leapfrog starts once and reverse mode keeps one
    set of checkpoints.
    """
    # Each grid point is ts[0] + k * dt with an integer step count k, so it is
    # rounded once and is never a sum of segment lengths. The grid is built
    # here in numpy, and the save times are taken from it, so each save time
    # is exactly a step end. Built from a traced start time, XLA fused t1 and
    # the `StepTo` grid separately, one with an fma and one without, and
    # diffrax rejected the mismatch under jit.
    # ponytail: the full grid is held in memory, about 8 bytes per step on
    # the host and again on the device (a large constant under an outer jit).
    # Only runs of ~1e7 steps notice; build it in chunks if that matters.
    total = int(n_steps.sum())
    if total == 0:
        return y[0][None], y[1][None]
    grid = t_start + np.arange(total + 1) * dt
    # `y` is prepended rather than saved: a save at t0 interpolates inside the
    # first step, so a row that is non-finite after it would come back NaN.
    q, p = advance(
        field, y, jnp.asarray(grid), adjoint, jnp.asarray(grid[np.cumsum(n_steps)])
    )
    return jnp.concat([y[0][None], q]), jnp.concat([y[1][None], p])


@eqx.filter_jit
def _energies(field: NBodyField, ts: Array, q: Array, p: Array) -> Array:
    """Return the (S,) total energy at each saved state.

    `lax.map`, not `vmap`: batching would widen the backend's chunked map by S
    and multiply its temp memory by S. It is jitted at module level so eager
    calls reuse one compilation.
    """
    e: Array = jax.lax.map(lambda a: total_energy(field, *a), (ts, q, p))
    return e


def _grid_steps(ts: np.ndarray, dt: float) -> np.ndarray:
    """Return the integer step counts ``n`` with ``ts == ts[0] + n * dt``.

    ``ts`` is a strictly increasing 1-D float64 array and ``dt > 0``. The
    integration builds each grid point as ``ts[0] + n * dt``, so this rounds
    ``(ts - ts[0]) / dt`` to integers ``n``, rebuilds the grid the same way,
    and accepts ``ts`` when each time is within `_GRID_ULPS` float spacings of
    its grid point. The spacing is taken at the largest of ``|ts[0]|``,
    ``|ts[k]|`` and ``|n[k] * dt|``, which bounds the rounding in building the
    grid point and in building ``ts`` from ``ts[0]``. ``n`` must be strictly
    increasing, so each interval is at least one step. Raises `ValueError` if
    ``dt`` is subnormal, which XLA flushes to zero, if ``ts`` has more than one
    time and ``dt`` is under 100 float spacings of the largest ``|ts|``, where
    grid points cannot be told apart reliably, or if ``ts`` is off the grid.
    """
    if dt < np.finfo(float).tiny:
        msg = f"dt={dt} is subnormal in float64, which XLA flushes to zero"
        raise ValueError(msg)
    t_max = np.max(np.abs(ts))
    if ts.size > 1 and dt < 100 * np.spacing(t_max):
        msg = (
            f"dt={dt} is too small for the magnitude of ts (|t| up to {t_max}) "
            "in float64: whole steps cannot be told apart at that resolution"
        )
        raise ValueError(msg)
    # Near the float64 maximum n * dt can overflow to inf, which makes ulp NaN
    # and fails the check below, so the warning is not needed.
    with np.errstate(over="ignore", invalid="ignore"):
        n: np.ndarray = np.rint((ts - ts[0]) / dt).astype(np.int64)
        ndt = n * dt
        ulp = np.spacing(np.maximum(np.maximum(np.abs(ts[0]), np.abs(ts)), np.abs(ndt)))
        on_grid = np.all(np.abs(ts - (ts[0] + ndt)) <= _GRID_ULPS * ulp)
    if np.any(np.diff(n) <= 0) or not on_grid:
        msg = f"each interval of ts must be a whole number of steps dt={dt}"
        raise ValueError(msg)
    return n


def _concrete(x: Any, unit: Any, name: str) -> np.ndarray:
    if _is_traced(x):
        msg = f"{name} must be concrete, not traced; pass it from outside jit"
        raise ValueError(msg)
    # Inside a trace the unit conversion would be staged, even on a constant.
    with jax.ensure_compile_time_eval():
        return np.asarray(u.ustrip(AllowValue, unit, x), dtype=float)


def _is_traced(tree: Any) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


def _check_system(system: System, q0: Any, units: Any) -> None:
    """Raise `ValueError` if ``system`` does not match ``q0`` and ``units``."""
    n_q0 = np.shape(u.ustrip(AllowValue, units["length"], q0))[0]
    if system.slices[-1][1] != n_q0:
        msg = f"system has {system.slices[-1][1]} particles but q0 has {n_q0}"
        raise ValueError(msg)
    if system.units != units:
        msg = "system units differ from the field's units; build both in one system"
        raise ValueError(msg)


def evolve(
    field: NBodyField,
    q0: Any,
    p0: Any,
    ts: Any,
    *,
    dt: Any,
    adjoint: dfx.AbstractAdjoint | None = None,
    check_dt: bool = True,
    energy: bool = False,
    system: System | None = None,
) -> NBodyResult:
    """Integrate ``field`` from ``(q0, p0)`` at ``ts[0]`` and save at each of ``ts``.

    ``q0`` and ``p0`` are (N, 3) positions and velocities, and ``ts`` and
    ``dt`` are the save times and the fixed step. Each may be an array or a
    Quantity, and is converted to the field's units. ``ts`` must be strictly
    increasing and lie on the grid ``ts[0] + n * dt`` with integer ``n``. Each
    time may be off its grid point by a few float spacings, the rounding from
    building ``ts`` as ``t0 + k * dt``, with `numpy.linspace` or through a
    unit conversion. `numpy.arange` adds a rounding error per element, so a
    long ``arange`` far from zero drifts off the grid and is rejected.
    ``ts`` and ``dt`` must be concrete, so they cannot be traced, but the
    field and the initial state can be, for example under `jax.vmap` or
    `jax.grad`.

    The steps are leapfrog (`galax.fields.Leapfrog`) at the times
    ``ts[0] + k * dt``. The whole run is one `diffrax.diffeqsolve`, saving at
    ``ts``. It compiles once for each pair of the total number of steps and
    the number of save times, and later calls with the same pair reuse it,
    however the save times are spaced.
    ``adjoint`` defaults to `diffrax.RecursiveCheckpointAdjoint`. With
    ``check_dt`` the step is checked once at ``ts[0]`` by `check_timestep`.
    The check also runs inside an outer `jax.jit` when the field, ``q0`` and
    ``p0`` are concrete, and is skipped when any of them is traced. With
    ``energy`` the result also holds the total energy at each save time.
    ``system`` is stored on the result so `NBodyResult.component` can split
    it.

    With the default `diffrax.RecursiveCheckpointAdjoint`, reverse mode
    (`jax.grad`) works, but forward mode (`jax.jvp`, `jax.jacfwd`) needs
    ``adjoint=diffrax.ForwardMode()``, which also suits derivatives with
    respect to a few parameters. The checkpoint adjoint stores about
    ``sqrt(2 n)`` checkpoints for a run of ``n`` steps, each a few times the
    size of one (q, p) state, plus the saved states.
    ``RecursiveCheckpointAdjoint(checkpoints=n)``, a checkpoint at every step,
    makes the gradient faster, from about 1.3 times at large N to several
    times at small N, for memory that grows linearly with ``n``.

    `evolve` runs forward in time only, so ``ts`` must increase. Flipping the
    velocities and running forward is the same as running backward only when
    the external potential is static. With a time-dependent potential it
    gives the wrong forces.

    A tracer that is NaN or inf from the start, or whose position becomes so,
    leaves the other particles' values and gradients unchanged. A tracer that
    reaches a finite point where the external force or its derivative is not
    finite (outside a user potential's valid region, or exactly at a singular
    point) makes reverse-mode gradients NaN; forward results are unaffected.

    galax.nbody needs float64, so `evolve` raises `ValueError` unless
    ``jax_enable_x64`` is on. Turn it on with
    ``jax.config.update("jax_enable_x64", True)`` at the start of the program,
    or set the environment variable ``JAX_ENABLE_X64=1``.

    Raises `ValueError` before any tracing if ``ts`` or ``dt`` is traced or
    not finite, ``ts`` is empty or has more than one dimension (a scalar is
    one save time), ``dt`` is not a scalar or is not positive, ``ts`` is not
    strictly increasing, the span of ``ts`` overflows float64, a time is off
    the grid, ``dt`` is subnormal, ``ts`` has more than one time and ``dt`` is
    under 100 float spacings of the largest ``|ts|`` (too small to tell the
    step times apart), or ``system`` has a different number of particles
    than ``q0`` or different units from ``field``.
    """
    if not jax.config.read("jax_enable_x64"):  # type: ignore[no-untyped-call]
        msg = (
            "galax.nbody needs float64, but jax_enable_x64 is off. Turn it on "
            'with jax.config.update("jax_enable_x64", True) at the start of '
            "the program, or set JAX_ENABLE_X64=1."
        )
        raise ValueError(msg)
    units = field.units
    ts_np = np.atleast_1d(_concrete(ts, units["time"], "ts"))
    dt_np = _concrete(dt, units["time"], "dt")
    if ts_np.ndim != 1:
        msg = f"ts must be 1-D, got shape {ts_np.shape}"
        raise ValueError(msg)
    if ts_np.size == 0:
        msg = "ts must contain at least one time"
        raise ValueError(msg)
    if not (np.all(np.isfinite(ts_np)) and np.all(np.isfinite(dt_np))):
        msg = f"ts and dt must be finite, got ts={ts_np} and dt={dt_np}"
        raise ValueError(msg)
    if dt_np.ndim != 0 or not dt_np > 0:
        msg = f"dt must be a positive scalar, got {dt_np}"
        raise ValueError(msg)
    if np.any(ts_np[1:] <= ts_np[:-1]):
        msg = "ts must be strictly increasing"
        raise ValueError(msg)
    with np.errstate(over="ignore"):
        span = ts_np[-1] - ts_np[0]
    if not np.isfinite(span):
        msg = f"the span of ts, {ts_np[0]} to {ts_np[-1]}, overflows float64"
        raise ValueError(msg)
    n_steps = np.diff(_grid_steps(ts_np, float(dt_np)))
    if system is not None:
        _check_system(system, q0, units)

    if check_dt and not _is_traced((field, q0, p0)):
        # Inside an outer jit, ops on closed-over constants are still staged,
        # so the check runs them at trace time to see values.
        with jax.ensure_compile_time_eval():
            # stacklevel 3 points the warning at the caller of `evolve`.
            check_timestep(field, q0, ts_np[0], dt_np, stacklevel=3)
    q = jnp.asarray(u.ustrip(AllowValue, units["length"], q0), dtype=float)
    p = jnp.asarray(u.ustrip(AllowValue, units["speed"], p0), dtype=float)
    adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint

    q_out, p_out = _integrate(
        field, (q, p), float(ts_np[0]), float(dt_np), n_steps, adjoint
    )
    t_out = jnp.asarray(ts_np)
    e = _energies(field, t_out, q_out, p_out) if energy else None
    return NBodyResult(ts=t_out, q=q_out, p=p_out, energy=e, units=units, system=system)
