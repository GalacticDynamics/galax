"""Descriptor for a Potential Parameter."""

from __future__ import annotations

__all__ = ["ParameterField"]

from dataclasses import KW_ONLY, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import astropy.units as u

import quaxed.array_api as xp
from jax_quantity import Quantity

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter
from galax.typing import Unit
from galax.utils.dataclasses import Sentinel, dataclass_with_converter, field

if TYPE_CHECKING:
    from galax.potential._potential.base import AbstractPotentialBase


def converter_parameter(value: Any) -> AbstractParameter:
    """Convert a value to a Parameter.

    Parameters
    ----------
    value : Any
        The value to convert to a Parameter.  If the value is a
        :class:`galax.potential.AbstractParameter`, it is returned as is.  If
        the value is a callable, it is converted to a
        :class:`galax.potential.UserParameter`. If the value is a
        :class:`jax_quantity.Quantity` (or :class`astropy.units.Quantity`), it
        is converted to a :class:`galax.potential.ConstantParameter`.
        If the value is none of the above, a :class:`TypeError`
        is raised.

    Returns
    -------
    AbstractParameter
        The value converted to a parameter.

    Raises
    ------
    TypeError
        If the value is not a valid input type described above.
    """
    if isinstance(value, AbstractParameter):
        out = value

    elif callable(value):
        unit = _get_unit_from_return_annotation(value)
        out = UserParameter(func=value, unit=unit)

    else:
        if isinstance(value, Quantity | u.Quantity):
            unit = u.Unit(value.unit)
        else:
            msg = "Parameter constant must be a Quantity"
            raise TypeError(msg)
        out = ConstantParameter(xp.asarray(value.value), unit=unit)

    return out


@final
@dataclass_with_converter(frozen=True, slots=True)
class ParameterField:
    """Descriptor for a Potential Parameter.

    Parameters
    ----------
    dimensions : PhysicalType
        Dimensions (unit-wise) of the parameter.
    equivalencies : Equivalency or tuple[Equivalency, ...], optional
        Equivalencies to use when converting the parameter value to the
        physical type. If not specified, the default equivalencies for the
        physical type will be used.
    """

    name: str = field(init=False)
    _: KW_ONLY
    default: AbstractParameter | Literal[Sentinel.MISSING] = field(
        default=Sentinel.MISSING,
        converter=lambda x: x if x is Sentinel.MISSING else converter_parameter(x),
    )
    dimensions: u.PhysicalType = field(converter=u.get_physical_type)
    equivalencies: u.Equivalency | tuple[u.Equivalency, ...] | None = None

    # ===========================================
    # Descriptor

    def __set_name__(self, owner: "type[AbstractPotentialBase]", name: str) -> None:
        object.__setattr__(self, "name", name)

    # -----------------------------
    # Getting

    @overload  # TODO: use `Self` when beartype is happy
    def __get__(
        self, instance: None, owner: "type[AbstractPotentialBase]"
    ) -> "ParameterField": ...

    @overload
    def __get__(
        self, instance: "AbstractPotentialBase", owner: None
    ) -> AbstractParameter: ...

    def __get__(  # TODO: use `Self` when beartype is happy
        self,
        instance: "AbstractPotentialBase | None",
        owner: "type[AbstractPotentialBase] | None",
    ) -> ParameterField | AbstractParameter:
        # Get from class
        if instance is None:
            # If the Parameter is being set as part of a dataclass constructor,
            # then we raise an AttributeError if there is no default value. This
            # is to prevent the Parameter from being set as the default value of
            # the dataclass field and erroneously included in the class'
            # ``__init__`` signature.
            if not is_dataclass(owner) or self.name not in owner.__dataclass_fields__:
                if self.default is Sentinel.MISSING:
                    raise AttributeError
                return self.default
            return self

        # Get from instance
        return cast(AbstractParameter, instance.__dict__[self.name])

    # -----------------------------

    def _check_unit(self, potential: "AbstractPotentialBase", unit: Unit) -> None:
        """Check that the given unit is compatible with the parameter's."""
        # When the potential is being constructed, the units may not have been
        # set yet, so we don't check the unit.
        if not hasattr(potential, "units"):
            return

        # Check the unit is compatible
        if not unit.is_equivalent(
            potential.units[self.dimensions],
            equivalencies=self.equivalencies,
        ):
            msg = (
                "Parameter function must return a value "
                f"with units equivalent to {self.dimensions}"
            )
            raise ValueError(msg)

    def __set__(
        self,
        potential: "AbstractPotentialBase",
        value: AbstractParameter | ParameterCallable | Any | Literal[Sentinel.MISSING],
    ) -> None:
        # TODO: use converter_parameter.
        # Convert
        if isinstance(value, AbstractParameter):
            # TODO: this doesn't handle the correct output unit, a. la.
            #       ``potential.units[self.dimensions]``
            # Check the unit is compatible
            self._check_unit(potential, value.unit)
        elif callable(value):
            # TODO: this only gets the existing unit, it doesn't handle the
            # correct output unit, a. la. potential.units[self.dimensions]
            unit = _get_unit_from_return_annotation(value)
            self._check_unit(potential, unit)  # Check the unit is compatible
            value = UserParameter(func=value, unit=unit)
        else:
            # TODO: the issue here is that ``units`` hasn't necessarily been set
            #       on the potential yet. What is needed is to possibly bail out
            #       here and defer the conversion until the units are set.
            #       AbstractPotentialBase has the ``_init_units`` method that
            #       can then call this method, hitting ``AbstractParameter``
            #       this time.
            unit = potential.units[self.dimensions]
            if isinstance(value, u.Quantity):
                value = value.to_value(unit, equivalencies=self.equivalencies)
            elif isinstance(value, Quantity):
                value = value.to_value(unit)
            value = ConstantParameter(xp.asarray(value), unit=unit)

        # Set
        potential.__dict__[self.name] = value


# -------------------------------------------


def _get_unit_from_return_annotation(func: ParameterCallable) -> Unit:
    """Get the unit from the return annotation of a Parameter function.

    Parameters
    ----------
    func : Callable[[Array[float, ()] | float | int], Array[float, (*shape,)]]
        The function to use to compute the parameter value.

    Returns
    -------
    Unit
        The unit from the return annotation of the function.
    """
    # Get the return annotation
    type_hints = get_type_hints(func, include_extras=True)
    if "return" not in type_hints:
        msg = "Parameter function must have a return annotation"
        raise TypeError(msg)

    # Check that the return annotation might contain a unit
    return_annotation = type_hints["return"]
    return_origin = get_origin(return_annotation)
    if return_origin is not Annotated:
        msg = "Parameter function return annotation must be annotated"
        raise TypeError(msg)

    # Get the unit from the return annotation
    return_args = get_args(return_annotation)
    has_unit = False
    for arg in return_args[1:]:
        # Try to convert the argument to a unit
        try:
            unit = u.Unit(arg)
        except ValueError:
            continue
        # Only one unit annotation is allowed
        if has_unit:
            msg = "function has more than one unit annotation"
            raise ValueError(msg)
        has_unit = True

    if not has_unit:
        msg = "function did not have a valid unit annotation"
        raise ValueError(msg)

    return unit
