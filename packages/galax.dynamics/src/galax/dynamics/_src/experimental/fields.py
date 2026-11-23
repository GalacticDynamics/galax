"""Orbit fields of the experimental spray simulator, on raw host-unit arrays."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

import diffraxtra as dfxtra

import galax.fields as gf
import galax.potential as gp


def _check_units(host: gp.AbstractPotential, prog: gp.AbstractPotential | None) -> None:
    # The progenitor's forces are added to the host's as raw arrays.
    if prog is not None and prog.units != host.units:
        msg = "progenitor_potential must have the same unit system as potential"
        raise ValueError(msg)


def _grad(pot: gp.AbstractPotential, q: Array, t: Array) -> Array:
    return gp.gradient(pot, q, t)  # type: ignore[return-value]


def _terms(acc: Any, vel: Any, solver: dfxtra.DiffEqSolver) -> Any:
    """Terms of ``dq/dt = vel(t, p, args)``, ``dp/dt = acc(t, q, args)``.

    `Leapfrog` takes two terms on ``(q, p)`` of shape ``(R, 3)``; any other
    solver one term on ``w`` of shape ``(R, 6)``.
    """
    if isinstance(solver.solver, gf.Leapfrog):
        return dfx.ODETerm(vel), dfx.ODETerm(acc)
    return dfx.ODETerm(
        lambda t, w, args: jnp.concat(
            [vel(t, w[..., 3:], args), acc(t, w[..., :3], args)], axis=-1
        )
    )


class _WholeStreamField(eqx.Module):
    """Rows ``[progenitor, tracer_1..tracer_N]``.

    Tracers move in the host plus the progenitor potential centred on row 0;
    row 0 moves in the host. ``args["active"]`` (shape ``(N,)``) zeroes the
    derivatives of inactive tracers.
    """

    host: gp.AbstractPotential
    progenitor_potential: gp.AbstractPotential | None

    def __check_init__(self) -> None:
        _check_units(self.host, self.progenitor_potential)

    def _active(self, args: Any) -> Array:
        return jnp.concat([jnp.ones(1, bool), args["active"]])[:, None]

    def _vel(self, t: Array, p: Array, args: Any) -> Array:  # noqa: ARG002
        return jnp.where(self._active(args), p, 0.0)

    def _acc(self, t: Array, q: Array, args: Any) -> Array:
        a = -_grad(self.host, q, t)
        if self.progenitor_potential is not None:
            # An inactive tracer may sit exactly at the progenitor, where a
            # singular potential has NaN gradients that `where` would leak into
            # reverse mode (NaN * 0): evaluate it at a safe separation instead.
            on = args["active"][:, None]
            dq = jnp.where(on, q[1:] - q[0], 1.0)
            a = a.at[1:].add(-_grad(self.progenitor_potential, dq, t))
        return jnp.where(self._active(args), a, 0.0)

    def terms(self, solver: dfxtra.DiffEqSolver) -> Any:
        return _terms(self._acc, self._vel, solver)


class _CopiesField(eqx.Module):
    """Rows ``[particle, progenitor]``, as the mock-stream ``_ProgenitorField``.

    The particle moves in the host plus the progenitor potential centred on
    row 1; the progenitor moves in the host.
    """

    host: gp.AbstractPotential
    progenitor_potential: gp.AbstractPotential

    def __check_init__(self) -> None:
        _check_units(self.host, self.progenitor_potential)

    def _acc(self, t: Array, q: Array, args: Any) -> Array:  # noqa: ARG002
        a = -_grad(self.host, q, t)
        return a.at[0].add(-_grad(self.progenitor_potential, q[0] - q[1], t))

    def terms(self, solver: dfxtra.DiffEqSolver) -> Any:
        return _terms(self._acc, lambda t, p, args: p, solver)  # noqa: ARG005


class _TrackField(eqx.Module):
    """One particle (a single row) moving in the host plus the progenitor.

    The progenitor potential is centred on ``track(t)``, the progenitor's
    position at time ``t``.
    """

    host: gp.AbstractPotential
    progenitor_potential: gp.AbstractPotential
    track: Any

    def __check_init__(self) -> None:
        _check_units(self.host, self.progenitor_potential)

    def _acc(self, t: Array, q: Array, args: Any) -> Array:  # noqa: ARG002
        return -_grad(self.host, q, t) - _grad(
            self.progenitor_potential, q - self.track(t), t
        )

    def terms(self, solver: dfxtra.DiffEqSolver) -> Any:
        return _terms(self._acc, lambda t, p, args: p, solver)  # noqa: ARG005
