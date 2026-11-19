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

_REL_TOL = 1e-9


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
        """Return the (S, n, 3) positions and velocities of component ``name``.

        Raises `ValueError` if the result has no ``system``.
        """
        if self.system is None:
            msg = "this result has no system; pass system= to evolve"
            raise ValueError(msg)
        q = self.system.split(jnp.swapaxes(self.q, 0, 1))[name]
        p = self.system.split(jnp.swapaxes(self.p, 0, 1))[name]
        return jnp.swapaxes(q, 0, 1), jnp.swapaxes(p, 0, 1)


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
    particles where that density is zero are skipped. With no limit at all
    ``dt_max`` is infinite.

    If ``dt > dt_max`` this warns with `UserWarning`, or raises `ValueError`
    when ``strict``. ``stacklevel`` is passed to `warnings.warn`; the default
    points the warning at the caller. It needs concrete values, so call it
    outside ``jit``.
    """
    units = field.units
    q = jnp.asarray(u.ustrip(AllowValue, units["length"], q), dtype=float)
    t = u.ustrip(AllowValue, units["time"], t)
    dt = float(u.ustrip(AllowValue, units["time"], dt))
    a = jnp.linalg.norm(field.acceleration(t, q), axis=-1)
    h = field.softening
    rho = jnp.asarray(gp.density(field.external_potential, q, t))
    soft = jnp.where(field.active & (h > 0), eta * jnp.sqrt(2 * h / a), jnp.inf)
    G = field._G  # noqa: SLF001
    tidal = jnp.where(field.active & (rho > 0), eta_rho / jnp.sqrt(G * rho), jnp.inf)
    dt_max = float(jnp.min(jnp.minimum(soft, tidal)))
    if dt > dt_max:
        msg = f"dt={dt} is above the safe time step dt_max={dt_max}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=stacklevel)
    return dt_max


@eqx.filter_jit
def advance(
    field: NBodyField,
    y: tuple[Array, Array],
    t0: Array,
    n_steps: int,
    dt: Array,
    adjoint: dfx.AbstractAdjoint,
) -> tuple[Array, Array]:
    """Return ``(q, p)`` after ``n_steps`` leapfrog steps of size ``dt`` from ``t0``.

    ``n_steps`` is static, so this compiles once per distinct value.
    """
    terms = (
        dfx.ODETerm(lambda t, p, a: p),  # noqa: ARG005
        dfx.ODETerm(lambda t, q, a: field.acceleration(t, q)),  # noqa: ARG005
    )
    step_ts = t0 + dt * jnp.arange(n_steps + 1)
    # `StepTo` sets the steps itself and requires `dt0=None`.
    sol = dfx.diffeqsolve(
        terms,
        gf.Leapfrog(),
        step_ts[0],
        step_ts[-1],
        None,
        y,
        stepsize_controller=dfx.StepTo(ts=step_ts),
        max_steps=n_steps,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
    )
    q, p = sol.ys
    return q[0], p[0]


def _concrete(x: Any, unit: Any, name: str) -> np.ndarray:
    x = u.ustrip(AllowValue, unit, x)
    if isinstance(x, jax.core.Tracer):
        msg = f"{name} must be concrete, not traced; pass it from outside jit"
        raise ValueError(msg)  # noqa: TRY004
    return np.asarray(x, dtype=float)


def _is_traced(tree: Any) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


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
    increasing and each interval must be a whole number of steps ``dt``.
    ``ts`` and ``dt`` must be concrete, so they cannot be traced, but the
    field and the initial state can be, for example under `jax.vmap` or
    `jax.grad`.

    The steps are leapfrog (`galax.fields.Leapfrog`) at the times
    ``ts[0] + k * dt``. ``adjoint`` defaults to
    `diffrax.RecursiveCheckpointAdjoint`. With ``check_dt`` the step is checked
    once at ``ts[0]`` by `check_timestep`; the check is skipped when the field
    or the initial state is traced. With ``energy`` the result also holds the
    total energy at each save time. ``system`` is stored on the result so
    `NBodyResult.component` can split it.

    With the default `diffrax.RecursiveCheckpointAdjoint`, reverse mode
    (`jax.grad`) works, but forward mode (`jax.jvp`, `jax.jacfwd`) needs
    ``adjoint=diffrax.ForwardMode()``, which also suits derivatives with
    respect to a few parameters. The checkpoint adjoint stores about
    ``sqrt(2 n)`` states for a segment of ``n`` steps, and
    ``RecursiveCheckpointAdjoint(checkpoints=...)`` sets that number.

    `evolve` runs forward in time only, so ``ts`` must increase. Flipping the
    velocities and running forward is the same as running backward only when
    the external potential is static. With a time-dependent potential it
    gives the wrong forces.

    Raises `ValueError` before any tracing if ``ts`` or ``dt`` is traced or
    not finite, ``dt <= 0``, ``ts`` is not strictly increasing, or an interval
    is not a whole number of steps.
    """
    units = field.units
    ts_np = np.atleast_1d(_concrete(ts, units["time"], "ts"))
    dt_np = _concrete(dt, units["time"], "dt")
    if ts_np.ndim != 1:
        msg = f"ts must be 1-D, got shape {ts_np.shape}"
        raise ValueError(msg)
    if not (np.all(np.isfinite(ts_np)) and np.all(np.isfinite(dt_np))):
        msg = f"ts and dt must be finite, got ts={ts_np} and dt={dt_np}"
        raise ValueError(msg)
    if dt_np.ndim != 0 or not dt_np > 0:
        msg = f"dt must be a positive scalar, got {dt_np}"
        raise ValueError(msg)
    if np.any(np.diff(ts_np) <= 0):
        msg = "ts must be strictly increasing"
        raise ValueError(msg)
    ratio = np.diff(ts_np) / dt_np
    n_steps = np.rint(ratio).astype(int)
    if np.any(np.abs(ratio - n_steps) > _REL_TOL * ratio):
        msg = f"each interval of ts must be a whole number of steps dt={dt_np}"
        raise ValueError(msg)

    q = jnp.asarray(u.ustrip(AllowValue, units["length"], q0), dtype=float)
    p = jnp.asarray(u.ustrip(AllowValue, units["speed"], p0), dtype=float)
    if check_dt and not _is_traced((field, q)):
        # stacklevel 3 points the warning at the caller of `evolve`.
        check_timestep(field, q, ts_np[0], dt_np, stacklevel=3)
    adjoint = dfx.RecursiveCheckpointAdjoint() if adjoint is None else adjoint

    t_start, dt_arr = jnp.asarray(ts_np[0]), jnp.asarray(dt_np)
    qs, ps = [q], [p]
    before = 0
    # ponytail: each segment re-initialises the leapfrog, one extra force
    # evaluation per save interval; carry `solver_state` with made_jump=False
    # if that cost matters.
    for n in n_steps.tolist():
        # The grid is ts[0] + steps * dt, never a sum of segment lengths.
        t0 = t_start + before * dt_arr
        q, p = advance(field, (q, p), t0, n, dt_arr, adjoint)
        qs.append(q)
        ps.append(p)
        before += n

    t_out = jnp.asarray(ts_np)
    q_out, p_out = jnp.stack(qs), jnp.stack(ps)
    e = None
    if energy:
        # `lax.map`, not `vmap`: batching would widen the backend's chunked
        # map by S and multiply its temp memory by S.
        e = jax.lax.map(lambda a: total_energy(field, *a), (t_out, q_out, p_out))
    return NBodyResult(ts=t_out, q=q_out, p=p_out, energy=e, units=units, system=system)
