"""The progenitor's shared track, with a precomputed sensitivity for gradients."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import diffraxtra as dfxtra

import galax.potential as gp
from .schedule import Uniform
from .solve import is_lattice, progenitor_states

sg = jax.lax.stop_gradient

# Uniform knots of an adaptive progenitor solve: per release, and at least.
KNOTS_PER_RELEASE = 20
MIN_KNOTS = 2**14


def _hermite(t0: Array, dt: Array, x: Array, v: Array, t: Array) -> Array:
    """Cubic Hermite interpolant of the knots ``(x, v)`` at ``t0 + k dt``."""
    s = (t - t0) / dt
    i = jnp.clip(jnp.floor(s).astype(int), 0, x.shape[0] - 2)
    a = s - i
    b = 1 - a
    return (
        (b * b * (1 + 2 * a)) * x[i]
        + (a * b * b * dt) * v[i]
        + (a * a * (3 - 2 * a)) * x[i + 1]
        - (a * a * b * dt) * v[i + 1]
    )


@jax.custom_jvp
def _track(
    theta: Array,  # noqa: ARG001
    t: Array,
    t0: Array,
    dt: Array,
    x: Array,
    v: Array,
    Sx: Array,  # noqa: ARG001
    Sv: Array,  # noqa: ARG001
) -> Array:
    return _hermite(t0, dt, x, v, t)


@_track.defjvp
def _track_jvp(primals: tuple[Array, ...], tangents: tuple[Array, ...]) -> Any:
    # The knots depend on theta only through S; t0 and dt (from the schedule)
    # move the knot times. x, v, Sx and Sv are constants.
    _, t, t0, dt, x, v, Sx, Sv = primals
    theta_dot, t_dot, t0_dot, dt_dot = tangents[:4]
    out, out_dot = jax.jvp(
        lambda t0, dt, t: _hermite(t0, dt, x, v, t),
        (t0, dt, t),
        (t0_dot, dt_dot, t_dot),
    )
    return out, out_dot + _hermite(t0, dt, Sx, Sv, t) @ theta_dot


class ProgenitorTrack(eqx.Module):
    """The progenitor's position, Hermite-interpolated from knots at ``t0 + k dt``.

    ``(Sx, Sv)`` (``(n, 3, len(theta))``) is the knots' derivative in
    ``theta``. The value never uses it; the `custom_jvp` tangent is
    ``S(t) @ theta_dot + d track/dt * t_dot`` (plus the knot times' motion).
    """

    theta: Array
    t0: Array
    dt: Array
    x: Array
    v: Array
    Sx: Array
    Sv: Array

    def __call__(self, t: Array) -> Array:
        return _track(self.theta, t, self.t0, self.dt, self.x, self.v, self.Sx, self.Sv)


def make_track(
    pot: gp.AbstractPotential,
    w_today: Array,
    schedule: Uniform,
    solver: dfxtra.DiffEqSolver,
    ws: Array,
    wrt: Any,
) -> ProgenitorTrack:
    """Return the progenitor's track from ``w_today`` at ``schedule.t1``.

    The knots are the progenitor's states on the lattice for a fixed-step
    ``solver`` (``ws``, from `progenitor_states` with ``solver``, not solved
    again), else ``max(20 n_release, 2**14)`` uniform knots. ``theta`` is
    the inexact array leaves of ``(w_today, pot, schedule)`` that the
    equinox filter spec ``wrt`` selects, raveled; the sensitivity is
    `jax.jacfwd` of the knots with a `diffrax.ForwardMode` copy of
    ``solver``. Knots and sensitivity are constants (`stop_gradient`).
    """
    if not is_lattice(solver):
        n = max(KNOTS_PER_RELEASE * schedule.n_release, MIN_KNOTS)
        schedule = Uniform(schedule.t_age, n, 1, schedule.t1)
    dyn, static = eqx.partition((w_today, pot, schedule), wrt)
    dyn, rest = eqx.partition(dyn, eqx.is_inexact_array)
    static = eqx.combine(rest, static)
    theta, unravel = ravel_pytree(dyn)
    fwd = eqx.tree_at(lambda s: s.adjoint, solver, dfx.ForwardMode())

    def knots(th: Array) -> Array:
        w, p, sched = eqx.combine(unravel(th), static)
        return progenitor_states(p, w, sched, fwd)

    th0 = sg(theta)
    w = sg(ws if is_lattice(solver) else knots(th0))
    S = sg(jax.jacfwd(knots)(th0))  # (n, 6, len(theta))
    dt = jnp.asarray(schedule.t_age, dtype=float) / schedule.n_steps()
    return ProgenitorTrack(
        theta, schedule.lattice()[0], dt, w[:, :3], w[:, 3:], S[:, :3], S[:, 3:]
    )
