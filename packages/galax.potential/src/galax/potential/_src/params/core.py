"""Parameters on a Potential."""

__all__ = [
    "LinearParameter",
    "CustomParameter",
]

import functools as ft

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.core

import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .base import AbstractParameter

t0 = u.Q(0, "Myr")


class LinearParameter(AbstractParameter):
    """Linear time dependence Parameter.

    This is in point-slope form, where the parameter is given by

    .. math::

        p(t) = m * (t - ti) + p(ti)

    Parameters
    ----------
    slope : Quantity[float, (), "[parameter]/[time]"]
        The slope of the linear parameter.
    point_time : Array[float, (), "time"]
        The time at which the parameter is equal to the intercept.
    point_value : Quantity[float, (), "[parameter]"]
        The value of the parameter at the ``point_time``.

    Examples
    --------
    >>> import galax.potential as gp
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> lp = gp.params.LinearParameter(slope=u.Q(-1e3, "Msun/yr"),
    ...     point_time=u.Q(0, "Myr"), point_value=u.Q(1e12, "Msun"))

    >>> lp(u.Q(0, "Gyr")).uconvert("Msun")
    Q(1.e+12, 'solMass')

    >>> jnp.round(lp(u.Q(1.0, "Gyr")), 3)
    Q(0., 'Gyr solMass / yr')

    The parameter can then be used as a potential's field:

    >>> pot = gp.KeplerPotential(m_tot=lp, units="galactic")

    """

    slope: gt.QuSzAny = eqx.field(converter=u.Q.from_)
    point_time: gt.BBtQuSz0 = eqx.field(converter=u.Quantity["time"].from_)
    point_value: gt.QuSzAny = eqx.field(converter=u.Q.from_)

    def __check_init__(self) -> None:
        """Check the initialization of the class."""
        # TODO: check point_value and slope * point_time have the same dimensions

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **_: Any
    ) -> gt.QuSzAny | gt.SzAny:
        """Return the parameter value.

        .. math::

            p(t) = m * (t - ti) + p(ti)

        Returns
        -------
        Array[float, "*shape"]
            The constant parameter value.

        Examples
        --------
        >>> from galax.potential.params import LinearParameter
        >>> import unxt as u
        >>> import quaxed.numpy as jnp

        >>> lp = LinearParameter(slope=u.Q(-1, "Msun/yr"),
        ...                      point_time=u.Q(0, "Myr"),
        ...                      point_value=u.Q(1e9, "Msun"))

        >>> lp(u.Q(0, "Gyr")).uconvert("Msun")
        Q(1.e+09, 'solMass')

        >>> jnp.round(lp(u.Q(1, "Gyr")), 3)
        Q(0., 'Gyr solMass / yr')

        """
        out = self.slope * (t - self.point_time) + self.point_value
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)


#####################################################################
# User-defined Parameter
# For passing a function as a parameter.


@final
class CustomParameter(AbstractParameter):
    """User-defined Parameter.

    Parameters
    ----------
    func : Callable[..., Array[float, (*shape,)]]
        The function to use to compute the parameter value. Called as
        ``func(t, *args, **kwargs)``, so it takes the time first and must
        accept whatever ``args`` and ``kwargs`` hold -- plus any keywords the
        call site adds, for which a trailing ``**_`` is the usual answer.
    args : tuple
        Extra positional arguments passed to ``func`` after ``t``. Put any
        *data* the function needs here rather than closing over it -- see
        below.
    kwargs : dict
        The same, by name. Use it once positional stops reading clearly.
        Keywords given at the call site override these. A stored ``ustrip``
        is *not* this method's own argument -- it is forwarded to ``func``
        like any other key -- so it is not a way to set a default unit.

    Examples
    --------
    >>> from galax.potential.params import CustomParameter
    >>> import unxt as u
    >>> from unxts.parametric import ParametricQuantity

    >>> def func(t: u.Quantity["time"]) -> ParametricQuantity["mass"]:
    ...     return u.Q(1e9, "Msun/Gyr") * t

    >>> up = CustomParameter(func=func)
    >>> up(u.Q(1e3, "Myr"))
    Q(1.e+12, 'Myr solMass / Gyr')

    Data the function needs goes in ``args``, not in a closure:

    >>> def scaled(t, m0):
    ...     return m0 * u.ustrip("Gyr", t)

    >>> up = CustomParameter(func=scaled, args=(u.Q(1e9, "Msun"),))
    >>> up(u.Q(2.0, "Gyr"))
    Q(2.e+09, 'solMass')

    By name, and overridable at the call site:

    >>> def ramp(t, *, m0, rate):
    ...     return m0 + rate * u.ustrip("Gyr", t)

    >>> up = CustomParameter(func=ramp,
    ...                      kwargs={"m0": u.Q(1e9, "Msun"), "rate": u.Q(1e9, "Msun")})
    >>> up(u.Q(2.0, "Gyr"))
    Q(3.e+09, 'solMass')

    >>> up(u.Q(2.0, "Gyr"), rate=u.Q(0.0, "Msun"))
    Q(1.e+09, 'solMass')

    ``func`` is a *static* field -- it is hashed, not traced -- so arrays
    captured in a closure are invisible to JAX. They are not pytree leaves,
    which costs two things that are easy to miss. A closure is hashed by
    identity, so rebuilding one over the *same* data is a fresh `jax.jit`
    cache entry and recompiles; and `equinox.tree_serialise_leaves` writes
    nothing for it, so saving a potential silently drops the data. ``args``
    is an ordinary field, so its arrays are leaves and both work.

    Being leaves is also the constraint: everything in ``args`` and
    ``kwargs`` passes through this method's `jax.jit`, so it must be a JAX
    type. Arrays, Quantities, ``None``, Python scalars and nested
    tuples/dicts of those are fine; a string, a function or a `unxt` unit
    object raises ``TypeError: Error interpreting argument ... as an
    abstract array``. Anything genuinely static belongs in a closure over
    ``func``, which is where the static field still earns its keep.

    """

    # `Callable[..., Any]`, not `ParameterCallable`: with `args` the function
    # takes whatever data it was given after `t`, so its signature is the
    # caller's business. `ParameterCallable` still describes `__call__` below,
    # which is what a *parameter* must look like and is unchanged.
    func: Callable[..., Any] = eqx.field(static=True)
    args: tuple[Any, ...] = eqx.field(default=())
    kwargs: dict[str, Any] = eqx.field(default_factory=dict)

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        # Call-site keywords win over stored ones, which is what makes the
        # stored ones defaults rather than a second, invisible call site.
        #
        # A *call-site* `ustrip` is consumed by this method and never reaches
        # `func`, being a named parameter. A *stored* one is not: `self.kwargs`
        # is merged into what gets forwarded, so `kwargs={"ustrip": ...}` goes
        # to `func` like any other key. Storing one is therefore not a way to
        # set this method's `ustrip`, and for the only value anyone would want
        # there -- a unit -- it does not even get that far, failing the leaf
        # constraint below first.
        out = self.func(t, *self.args, **{**self.kwargs, **kwargs})
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)
