"""Fixed-time-step mock streams on raw arrays.

Everything here works on plain arrays in the potential's unit system (for
``units="galactic"``: kpc, Myr, Msun, kpc/Myr) and on the potential's unitless
``_gradient``, so that whole batches of streams compile to one
``jax.jit(jax.vmap(...))`` call with static shapes.

The integrator is kick-drift-kick leapfrog with a fixed step: symplectic and
time-reversible, so a forward-then-backward round trip returns the initial
conditions to round-off.
"""

__all__ = ["fixed_step_stream"]

import functools as ft
import math

from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import unxt as u

import galax.potential as gp
from galax.dynamics._src.legacy.mockstream.df import AbstractStreamDF

Carry: TypeAlias = tuple[Array, Array, Array]  # q, p, acceleration; each (N, 3)
AccFn: TypeAlias = Callable[[Array, Any], Array]


def _leapfrog(
    acc: AccFn,
    carry: Carry,
    t0: Any,
    h: Any,
    n_steps: int,
    *,
    hold: Callable[[Array, Carry], Carry] | None = None,
    save: bool = False,
) -> Any:
    """KDK leapfrog: ``n_steps`` steps of ``h`` from ``t0``.

    ``carry`` is ``(q, p, a)`` with ``a = acc(q, t0)``. The end-of-step
    acceleration is the next step's start one, so each step costs one force
    evaluation. ``hold(k, carry)``, if given, is applied before step ``k``. With
    ``save=True`` this also returns the ``(q, p)`` trajectory, shape
    ``(n_steps + 1, N, 3)``, starting with the initial state.
    """

    def step(carry: Carry, k: Array) -> tuple[Carry, Any]:
        if hold is not None:
            carry = hold(k, carry)
        q, p, a = carry
        q = q + h * (p + 0.5 * h * a)
        a2 = acc(q, t0 + (k + 1) * h)
        out = (q, p + 0.5 * h * (a + a2), a2)
        return out, (out[:2] if save else None)

    final, traj = jax.lax.scan(step, carry, jnp.arange(n_steps))
    if not save:
        return final
    qs, ps = (
        jnp.concat([x0[None], xs]) for x0, xs in zip(carry[:2], traj, strict=True)
    )
    return final, (qs, ps)


def _release_steps(n_steps: int, n_release: int) -> np.ndarray:
    """Grid step at which each of the ``n_release`` particle pairs is released.

    Evenly spread over steps ``0`` (``t = -t_age``) to ``n_steps`` (today).
    """
    return np.rint(np.linspace(0, n_steps, n_release)).astype(int)


@ft.partial(jax.jit, static_argnames=("dt", "n_release", "t_age_max"))
def fixed_step_stream(
    pot: gp.AbstractPotential,
    df: AbstractStreamDF,
    key: PRNGKeyArray,
    prog_w0: Array,
    prog_mass: Any,
    t_age: Any,
    *,
    dt: float,
    n_release: int,
    t_age_max: float,
) -> tuple[Array, Array]:
    """Particle-spray stream on a fixed time grid.

    The progenitor is integrated backward from today (``t = 0``) by ``t_age``,
    a leading and a trailing particle are released from it with ``df`` at each
    of ``n_release`` grid times, and all particles are then integrated forward
    to today in one ``jax.lax.scan``. A particle that is not yet released is
    held at its release state until its release step.

    The number of steps, ``ceil(t_age_max / dt)``, is static. The step used is
    ``t_age / n_steps`` (at most ``dt``), so ``t_age`` may vary inside one
    ``jax.vmap`` batch without changing any shape: a younger stream takes the
    same number of finer steps.

    Parameters
    ----------
    pot
        Host potential. All arrays are in ``pot.units``.
    df
        Release distribution, e.g. `galax.dynamics.FardalStreamDF` or
        `galax.dynamics.ChenStreamDF`.
    key
        Random key, passed to ``df`` as is (``MockStreamGenerator`` does the
        same, so equal keys and release times give equal draws).
    prog_w0
        Progenitor phase-space position today, shape ``(6,)``.
    prog_mass
        Progenitor mass, held constant.
    t_age
        Time since the first release, ``0 <= t_age <= t_age_max``; checked at
        run time, since a larger ``t_age`` would silently take steps above
        ``dt``. Under ``jax.vmap`` one out-of-range value raises for the whole
        batch.
    dt
        Largest allowed time step (static).
    n_release
        Number of release times; the stream has ``2 * n_release`` particles
        (static).
    t_age_max
        Largest ``t_age`` the compiled function supports (static).

    Returns
    -------
    stream_w
        Stream particles today, shape ``(2 * n_release, 6)``: rows
        ``[:n_release]`` are the leading arm, ``[n_release:]`` the trailing
        arm, each ordered by release time (oldest first).
    prog_w
        The progenitor today after the backward and forward integration, shape
        ``(6,)``; equal to ``prog_w0`` to round-off.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    >>> w0 = jnp.array([15.0, 0.0, 10.0, 0.0, 0.17, -0.05])  # kpc, kpc/Myr
    >>> stream_w, prog_w = gd.experimental.fixed_step_stream(
    ...     pot, gd.FardalStreamDF(), jr.key(0), w0, 1e5, 1000.0,
    ...     dt=1.0, n_release=100, t_age_max=2000.0)
    >>> stream_w.shape, prog_w.shape
    ((200, 6), (6,))

    """
    if not dt > 0 or n_release < 1:
        msg = f"need dt > 0 and n_release >= 1, got dt={dt}, n_release={n_release}"
        raise ValueError(msg)
    n_steps = math.ceil(t_age_max / dt)
    k_rel = _release_steps(n_steps, n_release)
    # float64 even for a float32 t_age, so the step and force times keep full
    # precision; the check is written so a NaN t_age fails it too.
    t_age = jnp.asarray(t_age, dtype=jnp.result_type(prog_w0.dtype, float))
    t_age = eqx.error_if(
        t_age, ~((t_age >= 0) & (t_age <= t_age_max)), "t_age must be in [0, t_age_max]"
    )
    h = t_age / n_steps
    t0 = -t_age

    def acc(q: Array, t: Any) -> Array:
        out: Array = -pot._gradient(q, t)  # noqa: SLF001
        return out

    # Progenitor back to -t_age, keeping its state at every grid time.
    q0, p0 = prog_w0[None, :3], prog_w0[None, 3:]
    _, (qs, ps) = _leapfrog(acc, (q0, p0, acc(q0, 0.0)), 0.0, -h, n_steps, save=True)
    qs, ps = qs[::-1, 0], ps[::-1, 0]  # now indexed by grid step, t = t0 + k h

    # Release states from the DF, at the release steps.
    us = pot.units
    t_rel = t0 + k_rel * h
    x_l, v_l, x_t, v_t = df._sample(  # noqa: SLF001
        key,
        pot,
        u.Q(qs[k_rel], us["length"]),
        u.Q(ps[k_rel], us["speed"]),
        u.Q(jnp.full(n_release, prog_mass), us["mass"]),
        u.Q(t_rel, us["time"]),
    )
    q_rel = jnp.concat([u.ustrip(us["length"], x) for x in (x_l, x_t)])
    p_rel = jnp.concat([u.ustrip(us["speed"], v) for v in (v_l, v_t)])

    # Row 0 is the progenitor, released at step 0; then lead, then trail.
    k_all = jnp.asarray(np.concat([[0], k_rel, k_rel]))
    q_ic = jnp.concat([qs[:1], q_rel])
    p_ic = jnp.concat([ps[:1], p_rel])
    ic = (q_ic, p_ic, acc(q_ic, t0 + k_all * h))

    def hold(k: Array, carry: Carry) -> Carry:
        waiting = (k <= k_all)[:, None]
        return tuple(jnp.where(waiting, x0, x) for x0, x in zip(ic, carry, strict=True))  # type: ignore[return-value]

    final = hold(jnp.asarray(n_steps), _leapfrog(acc, ic, t0, h, n_steps, hold=hold))
    w = jnp.concat(final[:2], axis=1)
    return w[1:], w[0]
