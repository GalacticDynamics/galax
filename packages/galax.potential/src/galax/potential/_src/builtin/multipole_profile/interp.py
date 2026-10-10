"""Parameters interpolated over a time grid."""

__all__ = [
    "time_interpolated_parameter",
]

from typing import Any

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.harmonic import eval_log_spline, fit_log_spline
from galax.potential._src.params.core import CustomParameter

_Q = Unless(u.AbstractQuantity, u.Q.from_)


def _interpolate(
    t: gt.BBtQuSz0,
    ts: gt.QuSzAny,
    values: gt.QuSzAny,
    derivs: gt.QuSzAny,
    /,
    **__: Any,
) -> gt.QuSzAny:
    """Interpolate ``values`` to ``t``, clamped to the tabulated range.

    A module-level function, not a closure: `CustomParameter.func` is a
    static field, so its identity is part of the `jax.jit` cache key. A
    closure built per parameter would be a new key every time; this is one
    key for every tabulated parameter in the process.

    This lives beside the expansion rather than in `params` because it is
    not the general way to tabulate a parameter -- it hardwires the cubic
    Hermite pair the radial profiles use, so the coefficients are
    interpolated in time by the same rule that interpolates them in radius.
    A caller wanting some other rule should build a `CustomParameter` over
    `interpax` directly; the parameters tutorial shows how.
    """
    tval = u.ustrip(AllowValue, ts.unit, t)
    grid = ts.value
    # Clamp rather than continue the edge cubic; see the factory's docstring.
    #
    # `where` rather than `clip`, for the derivative at the knots. JAX gives
    # `clip` the subgradient 0.5 at a tie, so `jax.grad` at `ts[0]` or
    # `ts[-1]` came back exactly *half* the interior slope -- and `t = 0` is
    # galax's default time, so a grid starting at 0 makes that boundary an
    # ordinary query, not an exotic one. Written this way the endpoints take
    # the interior branch and get the interior derivative, which is the
    # useful convention: the clamp guards extrapolation, it is not a claim
    # that the parameter goes flat at its last knot. Values are identical.
    tval = jnp.asarray(tval)
    tq = jnp.where(tval < grid[0], grid[0], jnp.where(tval > grid[-1], grid[-1], tval))
    return u.Q(eval_log_spline(grid, values.value, derivs.value, tq), values.unit)


def time_interpolated_parameter(ts: Any, values: Any, /) -> CustomParameter:
    r"""Build a parameter tabulated on a time grid, interpolated between knots.

    For a quantity known at a set of times rather than in closed form -- the
    coefficients of an expansion built at each of several epochs, say. The
    values are interpolated with the same cubic Hermite basis the radial
    profiles use, so the parameter is :math:`C^1` in time and the force an
    integrator sees has no kink at a knot.

    The knot derivatives are fitted once, here, rather than on every
    evaluation: the fit is a linear solve and this is called inside an
    integrator's scan body.

    **Outside** ``[ts[0], ts[-1]]`` the value is *clamped* to the boundary
    value rather than extrapolated. A tabulated parameter carries no
    information about what happens beyond its grid, and continuing the edge
    cubic there would invent some: for a coefficient of an expansion that
    diverges quickly and silently. Clamping saturates instead, which is the
    same choice `MultipoleProfilePotential` makes radially for its density.
    It also keeps one out-of-range time from poisoning a whole vmapped batch,
    which an error under `jax.jit` could not avoid.

    This is a factory, not a class. There is nothing about a tabulated
    parameter that `CustomParameter` cannot hold once the table travels in
    ``args`` instead of a closure -- which is where data belongs anyway, for
    reasons `CustomParameter` documents. What is specific to this one is the
    derivative fit, the shape checks, and the decision to clamp, and all
    three live here.

    Parameters
    ----------
    ts
        Strictly increasing times, shape ``(n_t,)``.
    values
        Values at ``ts``, with time on the leading axis.

    Examples
    --------
    >>> import unxt as u
    >>> import quaxed.numpy as jnp
    >>> from galax.potential._src.builtin.multipole_profile.interp import (
    ...     time_interpolated_parameter)

    >>> p = time_interpolated_parameter(
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

    # Named, because the converter's own failure for the common wrong input --
    # a bare array or list -- is `TypeError: from_() missing 1 required
    # keyword-only argument: 'unit'`, which names neither the argument nor
    # what it wanted.
    def _as_quantity(x: Any, name: str, example: str) -> Any:
        try:
            return _Q(x)
        except TypeError as exc:
            msg = (
                f"{name} must carry units (got {type(x).__name__}); "
                f"pass e.g. u.Q(..., {example!r}) rather than a bare array"
            )
            raise TypeError(msg) from exc

    ts_ = _as_quantity(ts, "ts", "Gyr")
    values_ = _as_quantity(values, "values", "Msun")
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

    # `values.unit / ts.unit`, because this is d(values)/d(ts). The array is
    # the same either way -- `_interpolate` reads raw `.value` for all three,
    # so the label never reaches the answer -- but labelling it `values.unit`
    # made `derivs.uconvert('Msun/Gyr')` raise and `derivs.uconvert('Msun')`
    # silently succeed, which is exactly backwards.
    # Strictly increasing, and finite. `eval_log_spline` brackets a query with
    # a sorted-grid search, so an out-of-order grid returns a wrong number
    # rather than failing -- this is the one way to get a quietly wrong answer
    # out of here, and it was reachable because only
    # `MultipoleProfilePotential` checked it, not this public factory.
    #
    # `eqx.error_if` rather than a Python `if`, because unlike the shape checks
    # above this compares *values*, which a Python `if` cannot do under trace.
    # Non-finite is checked separately: `inf` passes `diff > 0` (inf - 1 is
    # inf) and then makes every interpolated value `nan`.
    bad = jnp.any(jnp.diff(ts_.value) <= 0) | jnp.any(~jnp.isfinite(ts_.value))
    ts_ = u.Q(
        eqx.error_if(
            ts_.value,
            bad,
            "ts must be strictly increasing and finite; it is searched as a "
            "sorted grid, so an out-of-order one returns wrong values rather "
            "than failing",
        ),
        ts_.unit,
    )

    derivs = u.Q(fit_log_spline(ts_.value, values_.value), values_.unit / ts_.unit)
    return CustomParameter(func=_interpolate, args=(ts_, values_, derivs))
