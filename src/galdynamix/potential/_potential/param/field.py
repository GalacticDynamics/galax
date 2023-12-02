from __future__ import annotations

__all__ = ["ParameterField"]

from dataclasses import KW_ONLY, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, cast, overload

import astropy.units as u
import jax.numpy as xp

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter

if TYPE_CHECKING:
    from galdynamix.potential._potential.base import AbstractPotential


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
    dimensions: u.PhysicalType  # TODO: add a converter_argument
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

    @overload
    def __get__(self, instance: None, owner: type[AbstractPotential]) -> ParameterField:
        ...

    @overload
    def __get__(self, instance: AbstractPotential, owner: None) -> AbstractParameter:
        ...

    def __get__(
        self, instance: AbstractPotential | None, owner: type[AbstractPotential] | None
    ) -> ParameterField | AbstractParameter:
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

    def __set__(
        self,
        potential: AbstractPotential,
        value: AbstractParameter | ParameterCallable | Any,
    ) -> None:
        # Convert
        if isinstance(value, AbstractParameter):
            # TODO: use the dimensions & equivalencies info to check the parameters.
            # TODO: use the units on the `potential` to convert the parameter value.
            pass
        elif callable(value):
            # TODO: use the dimensions & equivalencies info to check the parameters.
            # TODO: use the units on the `potential` to convert the parameter value.
            value = UserParameter(func=value)
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
