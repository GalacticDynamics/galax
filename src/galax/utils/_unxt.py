"""Compat waiting for `unxt` v1.3.0.

NOTE: this should be removed once `unxt` v1.3.0 is released.

"""

__all__ = ["AllowValue"]

from typing import Any, NoReturn

from jaxtyping import Array
from plum import dispatch

import unxt as u


class AllowValue:
    """A flag to allow a value to be passed through `unxt.ustrip`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array(1)
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    >>> u.ustrip(AllowValue, "km", u.Quantity(1000, "m"))
    Array(1., dtype=float64, ...)

    This is a flag, so it cannot be instantiated.

    >>> try: AllowValue()
    ... except TypeError as e: print(e)
    Cannot instantiate AllowValue

    """

    def __new__(cls) -> NoReturn:
        msg = "Cannot instantiate AllowValue"
        raise TypeError(msg)


@dispatch
def ustrip(
    flag: type[AllowValue],  # noqa: ARG001
    unit: Any,  # noqa: ARG001
    x: Array | float | int,
    /,
) -> Array | float | int:
    """Strip the units from a value. This is a no-op.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u

    >>> x = jnp.array(1)
    >>> y = u.ustrip(AllowValue, "km", x)
    >>> y is x
    True

    """
    return x


@dispatch  # TODO: type annotate by value
def ustrip(flag: type[AllowValue], unit: Any, x: u.AbstractQuantity, /) -> Any:  # noqa: ARG001
    """Strip the units from a quantity.

    Examples
    --------
    >>> import unxt as u
    >>> q = u.Quantity(1000, "m")
    >>> u.ustrip(AllowValue, "km", q)
    Array(1., dtype=float64, ...)

    """
    return u.ustrip(unit, x)
