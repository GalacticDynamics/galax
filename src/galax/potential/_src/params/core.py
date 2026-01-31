"""Parameters on a Potential."""

__all__ = [
    "LinearParameter",
    "CustomParameter",
]

import functools as ft
from typing import Any, final

import equinox as eqx
import jax
import jax.core

import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .base import AbstractParameter, ParameterCallable

t0 = u.Quantity(0, "Myr")


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
    >>> import coordinax.vecs as cxv
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> import unxt as u
    >>> import quaxed.numpy as jnp

    >>> lp = gp.params.LinearParameter(slope=u.Quantity(-1e3, "Msun/yr"),
    ...     point_time=u.Quantity(0, "Myr"), point_value=u.Quantity(1e12, "Msun"))

    >>> lp(u.Quantity(0, "Gyr")).uconvert("Msun")
    Quantity(Array(1.e+12, dtype=float64), unit='solMass')

    >>> jnp.round(lp(u.Quantity(1.0, "Gyr")), 3)
    Quantity(Array(0., dtype=float64, ...), unit='Gyr solMass / yr')

    Now let's show it in a potential:

    >>> pot = gp.KeplerPotential(m_tot=lp, units="galactic")

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([5.0, 0.0, 0.0], "kpc"),
    ...                              p=u.Quantity([0.0, 1_000.0, 0.0], "km/s"),
    ...                              t=u.Quantity(0.0, "Myr"))
    >>> savets = u.Quantity(jnp.linspace(0, 1, 10), "Gyr")
    >>> orbit = gd.compute_orbit(pot, w0, savets)
    >>> orbit.q.vconvert(cxv.CylindricalPos).rho.round(3)
    Distance(Array([ 5.   ,  7.159,  6.654, 10.295, 12.194, 11.337, 19.999,
                    24.589, 30.147, 40.37 ], dtype=float64),
        unit='kpc')

    """

    slope: gt.QuSzAny = eqx.field(converter=u.Quantity.from_)
    point_time: gt.BBtQuSz0 = eqx.field(converter=u.Quantity["time"].from_)
    point_value: gt.QuSzAny = eqx.field(converter=u.Quantity.from_)

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

        >>> lp = LinearParameter(slope=u.Quantity(-1, "Msun/yr"),
        ...                      point_time=u.Quantity(0, "Myr"),
        ...                      point_value=u.Quantity(1e9, "Msun"))

        >>> lp(u.Quantity(0, "Gyr")).uconvert("Msun")
        Quantity(Array(1.e+09, dtype=float64), unit='solMass')

        >>> jnp.round(lp(u.Quantity(1, "Gyr")), 3)
        Quantity(Array(0., dtype=float64), unit='Gyr solMass / yr')

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
    func : Callable[[BBtRealQuSz0], Array[float, (*shape,)]]
        The function to use to compute the parameter value.

    Examples
    --------
    >>> from galax.potential.params import CustomParameter
    >>> import unxt as u

    >>> def func(t: u.Quantity["time"]) -> u.Quantity["mass"]:
    ...     return u.Quantity(1e9, "Msun/Gyr") * t

    >>> up = CustomParameter(func=func)
    >>> up(u.Quantity(1e3, "Myr"))
    Quantity(Array(1.e+12, dtype=float64, ...), unit='Myr solMass / Gyr')

    """

    func: ParameterCallable = eqx.field(static=True)

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self, t: gt.BBtQuSz0, *, ustrip: u.AbstractUnit | None = None, **kwargs: Any
    ) -> gt.QuSzAny | gt.SzAny:
        out = self.func(t, **kwargs)
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)
