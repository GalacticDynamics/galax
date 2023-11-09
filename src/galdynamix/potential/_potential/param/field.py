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
    physical_type : PhysicalType
        Physical type of the parameter.
    equivalencies : Equivalency or tuple[Equivalency, ...], optional
        Equivalencies to use when converting the parameter value to the
        physical type. If not specified, the default equivalencies for the
        physical type will be used.
    """

    name: str = field(init=False)
    _: KW_ONLY
    physical_type: u.PhysicalType  # TODO: add a converter_argument
    equivalencies: u.Equivalency | tuple[u.Equivalency, ...] | None = None

    def __post_init__(self) -> None:
        # TODO: move this to a ``converter`` argument for a custom
        # ``dataclass_transform``'s ``__init__`` method.
        if isinstance(self.physical_type, str):
            object.__setattr__(
                self, "physical_type", u.get_physical_type(self.physical_type)
            )
        elif not isinstance(self.physical_type, u.PhysicalType):
            msg = (
                "Expected physical_type to be a PhysicalType, "
                f"got {self.physical_type!r}"
            )
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
        value: AbstractParameter | ParameterCallable | Any,  # noqa: ANN401
    ) -> None:
        # Convert
        if isinstance(value, AbstractParameter):
            # TODO: use the physical_type information to check the parameters.
            # TODO: use the units on the `potential` to convert the parameter value.
            pass
        elif callable(value):
            # TODO: use the physical_type information to check the parameters.
            # TODO: use the units on the `potential` to convert the parameter value.
            value = UserParameter(func=value)
        else:
            unit = potential.units[self.physical_type]
            if isinstance(value, u.Quantity):
                value = value.to_value(unit, equivalencies=self.equivalencies)

            value = ConstantParameter(xp.asarray(value), unit=unit)
        # Set
        potential.__dict__[self.name] = value
