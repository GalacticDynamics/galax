"""Parameters on a Potential."""

__all__ = [
    "ConstantParameter",
]

import functools as ft

from jaxtyping import ArrayLike
from typing import Any, NoReturn, final

import equinox as eqx
import jax
import jax.core
import quax_blocks
from quax import ArrayValue, register

import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .base import AbstractParameter

t0 = u.Q(0, "Myr")


@final
class ConstantParameter(AbstractParameter, ArrayValue, quax_blocks.NumpyMathMixin):  # type: ignore[misc]
    """Time-independent potential parameter.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> p = gp.params.ConstantParameter(value=u.Q(1., "Msun"))
    >>> p
    ConstantParameter(Q(1., 'solMass'))

    The parameter value is constant:

    >>> p(u.Q(0, "Gyr"))
    Q(1., 'solMass')

    >>> p(u.Q(1, "Gyr")) - p(u.Q(2, "Gyr"))
    Q(0., 'solMass')

    ConstantParameter supports arithmetic operations with other
    ConstantParameter objects:

    >>> p + p
    ConstantParameter(Q(2., 'solMass'))

    >>> p - p
    ConstantParameter(Q(0., 'solMass'))

    Most arithmetic operations degrade it back to a `unxt.Quantity`:

    >>> p + u.Q(2, "Msun")
    Q(3., 'solMass')

    >>> u.Q(2, "Msun") + p
    Q(3., 'solMass')

    >>> p - u.Q(2, "Msun")
    Q(-1., 'solMass')

    >>> u.Q(2, "Msun") - p
    Q(1., 'solMass')

    >>> p * 2
    Q(2., 'solMass')

    >>> 2 * p
    Q(2., 'solMass')

    >>> p / 2
    Q(0.5, 'solMass')

    >>> 2 / p
    Q(2., '1 / solMass')

    """

    # TODO: link this shape to the return shape from __call__
    value: gt.QuSzAny = eqx.field(converter=Unless(u.AbstractQuantity, u.Q.from_))
    """The time-independent value of the parameter."""

    def aval(self) -> jax.core.ShapedArray:
        """Return the dtype and shape info.

        Examples
        --------
        >>> import galax.potential as gp
        >>> import unxt as u

        >>> p = gp.params.ConstantParameter(value=u.Q(1., "Msun"))
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

        >>> p = gp.params.ConstantParameter(value=u.Q(1., "Msun"))
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
        ustrip: u.AbstractUnit | None = None,
        **__: Any,
    ) -> gt.QuSzAny:
        """Return the constant parameter value.

        Parameters
        ----------
        t : `~galax.typing.BBtQuSz0`, optional
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

        >>> p = ConstantParameter(value=u.Q(1, "Msun"))
        >>> p
        ConstantParameter(Q(1, 'solMass'))

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
