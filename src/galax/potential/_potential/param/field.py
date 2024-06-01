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

from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem, galactic

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter
from galax.typing import Unit
from galax.utils.dataclasses import Sentinel, dataclass_with_converter, field

if TYPE_CHECKING:
    from galax.potential import AbstractPotentialBase


def converter_parameter(value: Any) -> AbstractParameter:
    """Convert a value to a Parameter.

    Parameters
    ----------
    value : Any
        The value to convert to a Parameter.  If the value is a
        :class:`galax.potential.AbstractParameter`, it is returned as is.  If
        the value is a callable, it is converted to a
        :class:`galax.potential.UserParameter`. If the value is a
        :class:`unxt.Quantity` (or :class`astropy.units.Quantity`), it
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
        # TODO: fix specifying a unit system. It should just extract the
        # dimensions.
        unit = _get_unit_from_return_annotation(value, galactic)
        out = UserParameter(func=value, unit=unit)

    else:
        # `Quantity.constructor`` handles errors if the value cannot be
        # converted to a Quantity.
        value = Quantity.constructor(value)
        out = ConstantParameter(value, unit=value.unit)

    return out


@final
@dataclass_with_converter(frozen=True, slots=True)
class ParameterField:
    """Descriptor for a Potential Parameter.

    Parameters
    ----------
    dimensions : PhysicalType
        Dimensions (unit-wise) of the parameter.

    Examples
    --------
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> class KeplerPotential(gp.AbstractPotential):
    ...     mass: gp.ParameterField = gp.ParameterField(dimensions="mass")
    ...     def _potential(self, q, t):
    ...         return -self.constants["G"] * self.mass(t) / xp.linalg.norm(q, axis=-1)

    The `mass` parameter is a `ParameterField` that has dimensions of mass.
    This can be a constant value or a function of time.

    The simplest example is a constant mass:

    >>> potential = KeplerPotential(mass=1e12 * u.Msun, units="galactic")
    >>> potential
    KeplerPotential(
      units=UnitSystem(kpc, Myr, solMass, rad),
      constants=ImmutableDict({'G': ...}),
      mass=ConstantParameter(
        unit=Unit("solMass"),
        value=Quantity[PhysicalType('mass')](value=f64[], unit=Unit("solMass"))
      )
    )

    """

    name: str = field(init=False)
    """The name of the parameter."""

    _: KW_ONLY
    default: AbstractParameter | Literal[Sentinel.MISSING] = field(
        default=Sentinel.MISSING,
        converter=lambda x: x if x is Sentinel.MISSING else converter_parameter(x),
    )
    """The default value of the parameter."""

    dimensions: u.PhysicalType = field(converter=u.get_physical_type)
    """The dimensions (unit-wise) of the parameter."""

    # ===========================================
    # Descriptor

    def __set_name__(self, owner: "type[AbstractPotentialBase]", name: str) -> None:
        """Set the name of the parameter."""
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
        if not unit.is_equivalent(potential.units[self.dimensions]):
            msg = (
                "Parameter function must return a value "
                f"with units consistent with {self.dimensions}."
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
            self._check_unit(potential, value.unit)
            v = value
        elif callable(value):
            unit = _get_unit_from_return_annotation(value, potential.units)
            self._check_unit(potential, unit)  # Check the unit is compatible
            v = UserParameter(func=value, unit=unit)
        else:
            unit = potential.units[self.dimensions]
            v = ConstantParameter(Quantity.constructor(value, unit), unit=unit)

        # Set
        potential.__dict__[self.name] = v


# -------------------------------------------


def _get_unit_from_return_annotation(
    the_callable: ParameterCallable, unitsystem: AbstractUnitSystem
) -> Unit:
    """Get the unit from the return annotation of a Parameter function.

    Parameters
    ----------
    the_callable : Callable[[Array[float, ()] | float | int], Array[float, (*shape,)]]
        The function to use to compute the parameter value.
    unitsystem: AbstractUnitSystem
        The unit system to use to convert the return annotation to a unit.

    Returns
    -------
    Unit
        The unit from the return annotation of the function.
    """
    func = the_callable.__call__ if hasattr(the_callable, "__call__") else the_callable  # noqa: B004

    # Get the return annotation
    type_hints = get_type_hints(func, include_extras=True)
    if "return" not in type_hints:
        msg = "Parameter function must have a return annotation"
        raise TypeError(msg)

    # Check that the return annotation might contain a unit
    return_annotation = type_hints["return"]

    # Astropy compatibility
    if return_annotation.__module__.startswith("astropy"):
        return _get_unit_from_astropy_return_annotation(return_annotation)

    return unitsystem[return_annotation.type_parameter]


def _get_unit_from_astropy_return_annotation(return_annotation: Any) -> Unit:
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
