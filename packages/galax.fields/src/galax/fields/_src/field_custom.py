"""Custom field.

This is private API.

"""

__all__ = ["CustomField"]

from collections.abc import Callable
from jaxtyping import PyTree
from typing import Any, final

import equinox as eqx
import jax

import unxt as u

from .fields import AbstractField


@final
class CustomField(AbstractField):
    r"""A field built from a function, without subclassing.

    The field is :math:`dy/dt = f(t, y, args)` for any PyTree state ``y``.

    Parameters
    ----------
    func : Callable[[t, y, args], dy_dt]
        The function. It is called as ``func(t, y, args)`` and returns ``dy/dt``
        with the same PyTree structure as ``y``. It receives and returns plain
        arrays (no Quantities) in the field's ``units``, because ``integrate_field``
        strips units before calling it.
    units : `unxt.AbstractUnitSystem`, keyword-only
        The unit system the integration runs in. ``Quantity`` leaves of the
        initial conditions are converted to it, according to their dimension,
        and then stripped. Default "galactic".

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import galax.fields as gf

    Exponential decay, :math:`dy/dt = -k y`:

    >>> field = gf.CustomField(lambda t, y, args: -0.3 * y)
    >>> ts = jnp.linspace(0, 10, 3)
    >>> soln = gf.integrate_field(field, jnp.array([1.0]), ts)
    >>> soln.ys.round(4)
    Array([[1.    ],
           [0.2231],
           [0.0498]], dtype=float64)

    The state can be any PyTree, here a harmonic oscillator with state
    ``(x, v)``:

    >>> field = gf.CustomField(lambda t, y, args: (y[1], -y[0]))
    >>> y0 = (jnp.array(1.0), jnp.array(0.0))
    >>> soln = gf.integrate_field(field, y0, jnp.linspace(0, jnp.pi, 3))
    >>> soln.ys[0].round(4)
    Array([ 1.,  0., -1.], dtype=float64)

    """

    #: The function ``func(t, y, args) -> dy/dt``.
    func: Callable[[Any, PyTree, PyTree], PyTree]

    #: Unit system of the field.
    units: u.AbstractUnitSystem = eqx.field(
        default="galactic", converter=u.unitsystem, kw_only=True, static=True
    )

    def __call__(self, t: Any, y: PyTree, args: PyTree = None, /) -> PyTree:
        """Evaluate ``func(t, y, args)``."""
        return self.func(t, y, args)


@AbstractField.parse_inputs.dispatch
def parse_inputs(
    self: CustomField, t: Any, y0: PyTree, /, *, ustrip: bool
) -> tuple[Any, PyTree]:
    """Convert each Quantity leaf to the units of its own dimension.

    Array and float leaves pass through. If ``ustrip`` is true the units are
    stripped.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.fields as gf

    >>> field = gf.CustomField(lambda t, y, args: y, units="galactic")
    >>> field.parse_inputs(u.Q(1.0, "Gyr"), (u.Q(1.0, "kpc"), u.Q(1.0, "km/s")),
    ...                    ustrip=True)
    (Array(1000., dtype=float64, weak_type=True),
     (Array(1., dtype=float64, weak_type=True),
      Array(0.00102271, dtype=float64, weak_type=True)))

    """

    def convert(x: Any) -> Any:
        if not isinstance(x, u.AbstractQuantity):
            return x
        unit = self.units[u.dimension_of(x)]
        return u.ustrip(unit, x) if ustrip else u.uconvert(unit, x)

    is_q = lambda x: isinstance(x, u.AbstractQuantity)
    return (
        jax.tree.map(convert, t, is_leaf=is_q),
        jax.tree.map(convert, y0, is_leaf=is_q),
    )
