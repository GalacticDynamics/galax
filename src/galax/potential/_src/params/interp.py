"""Parameters interpolated over a time grid."""

__all__ = [
    "TimeInterpolatedParameter",
]

import functools as ft

from typing import Any, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .base import AbstractParameter

_Q = Unless(u.AbstractQuantity, u.Q.from_)


@final
class TimeInterpolatedParameter(AbstractParameter):
    """A parameter tabulated on a time grid and interpolated between.

    For a quantity that is known at a set of times rather than in closed
    form -- the coefficients of an expansion built at each of several epochs,
    say. The values are interpolated with the same cubic Hermite basis the
    radial profiles use, so the parameter is :math:`C^1` in time and the
    force an integrator sees has no kink at a knot.

    **Outside** ``[ts[0], ts[-1]]`` the value is *clamped* to the boundary
    value rather than extrapolated. A tabulated parameter carries no
    information about what happens beyond its grid, and continuing the edge
    cubic there would invent some: for a coefficient of an expansion that
    diverges quickly and silently. Clamping saturates instead, which is the
    same choice `MultipoleProfilePotential` makes radially for its density.
    It also keeps one out-of-range time from poisoning a whole vmapped batch,
    which an error under `jax.jit` could not avoid.

    Parameters
    ----------
    ts
        Strictly increasing times, shape ``(n_t,)``.
    values
        Values at ``ts``, shape ``(n_t, *shape)``.
    derivs
        Knot derivatives with respect to ``ts``, same shape as ``values``.
        Normally supplied by `from_values`, which fits them; they are stored
        rather than fitted per call because fitting is a linear solve and
        this is evaluated inside an integrator's scan body.

    Examples
    --------
    >>> import unxt as u
    >>> import quaxed.numpy as jnp
    >>> import galax.potential as gp

    >>> p = gp.params.TimeInterpolatedParameter.from_values(
    ...     u.Q(jnp.asarray([0.0, 1.0, 2.0]), "Gyr"),
    ...     u.Q(jnp.asarray([1e12, 2e12, 3e12]), "Msun"),
    ... )

    On a knot it returns that knot's value:

    >>> p(u.Q(1.0, "Gyr")).uconvert("Msun").round(3)
    Q(2.e+12, 'solMass')

    Between knots it interpolates:

    >>> p(u.Q(0.5, "Gyr")).uconvert("Msun").round(3)
    Q(1.5e+12, 'solMass')

    Outside the grid it clamps rather than extrapolating:

    >>> p(u.Q(-5.0, "Gyr")).uconvert("Msun").round(3)
    Q(1.e+12, 'solMass')

    >>> p(u.Q(99.0, "Gyr")).uconvert("Msun").round(3)
    Q(3.e+12, 'solMass')

    """

    ts: gt.QuSzAny = eqx.field(converter=_Q)
    """Times at which `values` is tabulated, strictly increasing."""

    values: gt.QuSzAny = eqx.field(converter=_Q)
    """Values at `ts`, with time on the leading axis."""

    derivs: gt.QuSzAny = eqx.field(converter=_Q)
    """d(`values`)/d(`ts`) at the knots; see `from_values`."""

    @classmethod
    def from_values(cls, ts: Any, values: Any, /) -> "TimeInterpolatedParameter":
        """Fit the knot derivatives and build the parameter.

        The normal entry point. The derivative fit is `fit_log_spline`'s
        not-a-knot cubic -- the same one the radial profiles use -- run once
        here rather than on every evaluation.
        """
        # Imported here: `galax.potential._src.harmonic` imports the potential
        # machinery, and params is below it.
        from galax.potential._src.harmonic import fit_log_spline

        ts_, values_ = _Q(ts), _Q(values)
        if ts_.ndim != 1:
            msg = f"ts must be 1-D (got shape {ts_.shape})"
            raise ValueError(msg)
        if ts_.shape[0] < 2:
            msg = (
                f"ts must have at least 2 entries (got {ts_.shape[0]}); "
                "there is nothing to interpolate between"
            )
            raise ValueError(msg)
        if values_.shape[0] != ts_.shape[0]:
            msg = (
                f"values must have time on the leading axis: got "
                f"{values_.shape[0]} values for {ts_.shape[0]} times"
            )
            raise ValueError(msg)

        derivs = fit_log_spline(ts_.value, values_.value)
        return cls(ts=ts_, values=values_, derivs=u.Q(derivs, values_.unit))

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **__: Any
    ) -> gt.QuSzAny:
        """Interpolate to ``t``, clamped to the tabulated range."""
        from galax.potential._src.harmonic import eval_log_spline

        tval = u.ustrip(AllowValue, self.ts.unit, t)
        grid = self.ts.value
        # Clamp rather than continue the edge cubic; see the class docstring.
        tq = jnp.clip(jnp.asarray(tval), grid[0], grid[-1])
        out = u.Q(
            eval_log_spline(grid, self.values.value, self.derivs.value, tq),
            self.values.unit,
        )
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)
