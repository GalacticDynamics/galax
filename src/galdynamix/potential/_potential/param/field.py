__all__ = ["ParameterField"]

from dataclasses import KW_ONLY, dataclass, field, is_dataclass
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints, overload

import astropy.units as u
import jax.numpy as xp

from galdynamix.potential._potential.core import AbstractPotential
from galdynamix.typing import Unit

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter


@dataclass(frozen=True, slots=True)
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
    dimensions: u.PhysicalType
    equivalencies: u.Equivalency | tuple[u.Equivalency, ...] | None = None

    def __post_init__(self) -> None:
        # Process the physical type
        # TODO: move this to a ``converter`` argument for a custom
        # ``dataclass_transform``'s ``__init__`` method.
        if isinstance(self.dimensions, str):
            object.__setattr__(self, "dimensions", u.get_physical_type(self.dimensions))
        elif not isinstance(self.dimensions, u.PhysicalType):
            msg = f"Expected dimensions to be a PhysicalType, got {self.dimensions!r}"
            raise TypeError(msg)

    # ===========================================
    # Descriptor

    def __set_name__(self, owner: type[AbstractPotential], name: str) -> None:
        object.__setattr__(self, "name", name)

    # -----------------------------

    @overload  # TODO: use `Self` when beartype is happy
    def __get__(
        self, instance: None, owner: type[AbstractPotential]
    ) -> "ParameterField":
        ...

    @overload
    def __get__(self, instance: AbstractPotential, owner: None) -> AbstractParameter:
        ...

    def __get__(  # TODO: use `Self` when beartype is happy
        self, instance: AbstractPotential | None, owner: type[AbstractPotential] | None
    ) -> "ParameterField | AbstractParameter":
        # Get from class
        if instance is None:
            # If the Parameter is being set as part of a dataclass constructor,
            # then we raise an AttributeError. This is to prevent the Parameter
            # from being set as the default value of the dataclass field and
            # erroneously included in the class' ``__init__`` signature.
            if not is_dataclass(owner) or self.name not in owner.__dataclass_fields__:
                raise AttributeError
            return self

        # Get from instance
        return cast(AbstractParameter, instance.__dict__[self.name])

    # -----------------------------

    def _check_unit(self, potential: AbstractPotential, unit: Unit) -> None:
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
        potential: AbstractPotential,
        value: AbstractParameter | ParameterCallable | Any,
    ) -> None:
        # Convert
        if isinstance(value, AbstractParameter):
            # TODO: this doesn't handle the correct output unit, a. la.
            # potential.units[self.dimensions]
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
            value = ConstantParameter(xp.asarray(value), unit=unit)

        # Set
        potential.__dict__[self.name] = value


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
