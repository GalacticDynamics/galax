"""Parameters on a Potential."""

__all__ = [
    "ConstantParameter",
]

import functools as ft
from typing import Any, NoReturn, final

import equinox as eqx
import jax
import jax.core
import quax_blocks
from jaxtyping import ArrayLike
from quax import ArrayValue, register

import unxt as u
from dataclassish.converters import Unless
from unxt._src.units.api import AstropyUnits
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .base import AbstractParameter

t0 = u.Quantity(0, "Myr")


@final
class ConstantParameter(AbstractParameter, ArrayValue, quax_blocks.NumpyMathMixin):  # type: ignore[misc]
    """Time-independent potential parameter.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> p = gp.params.ConstantParameter(value=u.Quantity(1., "Msun"))
    >>> p
    ConstantParameter(Quantity(Array(1., dtype=float64, ...), unit='solMass'))

    The parameter value is constant:

    >>> p(u.Quantity(0, "Gyr"))
    Quantity(Array(1., dtype=float64, ...), unit='solMass')

    >>> p(u.Quantity(1, "Gyr")) - p(u.Quantity(2, "Gyr"))
    Quantity(Array(0., dtype=float64, ...), unit='solMass')

    ConstantParameter supports arithmetic operations with other
    ConstantParameter objects:

    >>> p + p
    ConstantParameter(Quantity(Array(2., dtype=float64, ...), unit='solMass'))

    >>> p - p
    ConstantParameter(Quantity(Array(0., dtype=float64, ...), unit='solMass'))

    Most arithmetic operations degrade it back to a `unxt.Quantity`:

    >>> p + u.Quantity(2, "Msun")
    Quantity(Array(3., dtype=float64, ...), unit='solMass')

    >>> u.Quantity(2, "Msun") + p
    Quantity(Array(3., dtype=float64, ...), unit='solMass')

    >>> p - u.Quantity(2, "Msun")
    Quantity(Array(-1., dtype=float64, ...), unit='solMass')

    >>> u.Quantity(2, "Msun") - p
    Quantity(Array(1., dtype=float64, ...), unit='solMass')

    >>> p * 2
    Quantity(Array(2., dtype=float64, ...), unit='solMass')

    >>> 2 * p
    Quantity(Array(2., dtype=float64, ...), unit='solMass')

    >>> p / 2
    Quantity(Array(0.5, dtype=float64, ...), unit='solMass')

    >>> 2 / p
    Quantity(Array(2., dtype=float64, ...), unit='1 / solMass')

    """

    # TODO: link this shape to the return shape from __call__
    value: gt.QuSzAny = eqx.field(
        converter=Unless(u.AbstractQuantity, u.Quantity.from_)
    )
    """The time-independent value of the parameter."""

    def aval(self) -> jax.core.ShapedArray:
        """Return the dtype and shape info.

        Examples
        --------
        >>> import galax.potential as gp
        >>> import unxt as u

        >>> p = gp.params.ConstantParameter(value=u.Quantity(1., "Msun"))
        >>> p.aval()
        ShapedArray(float64[], weak_type=True)

        """
        return self.value.aval()

    def materialise(self) -> NoReturn:
        """Return the dtype and shape info.

        Examples
        --------
        >>> import galax.potential as gp
        >>> import unxt as u

        >>> p = gp.params.ConstantParameter(value=u.Quantity(1., "Msun"))
        >>> try:
        ...     p.materialise()
        ... except NotImplementedError as e:
        ...     print(e)
        Cannot materialise a ConstantParameter.

        """
        msg = "Cannot materialise a ConstantParameter."
        raise NotImplementedError(msg)

    @ft.partial(jax.jit, static_argnames=("ustrip",))
    def __call__(
        self,
        t: gt.BBtQuSz0 = t0,  # noqa: ARG002
        *,
        ustrip: AstropyUnits | None = None,
        **__: Any,
    ) -> gt.QuSzAny:
        """Return the constant parameter value.

        Parameters
        ----------
        t : `~galax._custom_types.BBtQuSz0`, optional
            This is ignored and is thus optional. Note that for most
            :class:`~galax.potential.AbstractParameter` the time is required.
        ustrip : Unit | None
            The unit to strip from the parameter value. If None, the
            parameter value is returned with its original unit.
        **kwargs : Any
            This is ignored.

        """
        return (
            self.value if ustrip is None else u.ustrip(AllowValue, ustrip, self.value)
        )

    # -------------------------------------------
    # String representation

    def __repr__(self) -> str:
        """Return string representation.

        Examples
        --------
        >>> from galax.potential.params import ConstantParameter
        >>> import unxt as u

        >>> p = ConstantParameter(value=u.Quantity(1, "Msun"))
        >>> p
        ConstantParameter(Quantity(Array(1, dtype=int64, ...), unit='solMass'))

        """
        return f"{self.__class__.__name__}({self.value!r})"


# ------------------------------------------
# add_p


@register(jax.lax.add_p)  # type: ignore[misc]
def add_p_constantparams(
    x: ConstantParameter, y: ConstantParameter, /
) -> ConstantParameter:
    return ConstantParameter(x.value + y.value)


@register(jax.lax.add_p)
def add_p_constantparam_scalar(
    x: ConstantParameter, y: u.AbstractQuantity, /
) -> u.AbstractQuantity:
    return x.value + y


@register(jax.lax.add_p)
def add_p_constantparam_scalar(
    x: u.AbstractQuantity, y: ConstantParameter, /
) -> u.AbstractQuantity:
    return x + y.value


# ------------------------------------------
# sub_p


@register(jax.lax.sub_p)  # type: ignore[misc]
def sub_p_constantparams(
    x: ConstantParameter, y: ConstantParameter, /
) -> ConstantParameter:
    return ConstantParameter(x.value - y.value)


@register(jax.lax.sub_p)
def sub_p_constantparam_scalar(
    x: ConstantParameter, y: u.AbstractQuantity, /
) -> u.AbstractQuantity:
    return x.value - y


@register(jax.lax.sub_p)
def sub_p_constantparam_scalar(
    x: u.AbstractQuantity, y: ConstantParameter, /
) -> u.AbstractQuantity:
    return x - y.value


# ------------------------------------------
# mul_p


@register(jax.lax.mul_p)  # type: ignore[misc]
def mul_p_obj_constantparam(
    x: ConstantParameter, y: u.AbstractQuantity | ArrayLike, /
) -> u.AbstractQuantity:
    return x.value * y


@register(jax.lax.mul_p)  # type: ignore[misc]
def mul_p_constantparam_obj(
    x: u.AbstractQuantity | ArrayLike, y: ConstantParameter, /
) -> u.AbstractQuantity:
    return x * y.value


# ------------------------------------------
# div_p


@register(jax.lax.div_p)
def div_p_constantparams(
    x: ConstantParameter, y: u.AbstractQuantity | ArrayLike, /
) -> u.AbstractQuantity:
    return x.value / y


@register(jax.lax.div_p)
def div_p_constantparams(
    x: u.AbstractQuantity | ArrayLike, y: ConstantParameter, /
) -> u.AbstractQuantity:
    return x / y.value
