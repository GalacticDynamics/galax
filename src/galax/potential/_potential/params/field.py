"""Descriptor for a Potential Parameter."""

from __future__ import annotations

__all__ = ["ParameterField"]

from dataclasses import KW_ONLY, is_dataclass
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
    get_args,
    get_type_hints,
    overload,
)

import astropy.units as u
from astropy.units import PhysicalType as Dimensions
from is_annotated import isannotated

from unxt import AbstractQuantity, Quantity

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter
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
        # TODO: check dimensions ``_get_dimensions_from_return_annotation``
        out = UserParameter(func=value)

    else:
        # `Quantity.constructor`` handles errors if the value cannot be
        # converted to a Quantity.
        out = ConstantParameter(Quantity.constructor(value))

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
    ...     mass: gp.params.ParameterField = gp.params.ParameterField(dimensions="mass")
    ...     def _potential(self, q, t):
    ...         return -self.constants["G"] * self.mass(t) / xp.linalg.norm(q, axis=-1)

    The `mass` parameter is a `ParameterField` that has dimensions of mass.
    This can be a constant value or a function of time.

    The simplest example is a constant mass:

    >>> potential = KeplerPotential(mass=1e12 * u.Msun, units="galactic")
    >>> potential
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      mass=ConstantParameter(
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
            # The normal return is the descriptor itself
            return self

        # Get from instance
        return cast(AbstractParameter, instance.__dict__[self.name])

    # -----------------------------

    def _check_dimensions(
        self, potential: "AbstractPotentialBase", dims: Dimensions
    ) -> None:
        """Check that the given unit is compatible with the parameter's."""
        # When the potential is being constructed, the units may not have been
        # set yet, so we don't check the unit.
        if not hasattr(potential, "units"):
            return

        # Check the unit is compatible
        if not dims.is_equivalent(self.dimensions):
            msg = (
                "Parameter function must return a value with "
                f"dimensions consistent with {self.dimensions}."
            )
            raise ValueError(msg)

    def __set__(
        self,
        potential: "AbstractPotentialBase",
        value: AbstractParameter | ParameterCallable | Any | Literal[Sentinel.MISSING],
    ) -> None:
        # Convert
        if isinstance(value, AbstractParameter):
            v = value
        elif callable(value):
            dims = _get_dimensions_from_return_annotation(value)
            self._check_dimensions(potential, dims)  # Check the unit is compatible
            v = UserParameter(func=value)
        else:
            unit = potential.units[self.dimensions]
            v = ConstantParameter(Quantity.constructor(value, unit))

        # Set
        potential.__dict__[self.name] = v


# -------------------------------------------


def _get_dimensions_from_return_annotation(func: ParameterCallable, /) -> Dimensions:
    """Get the dimensions from the return annotation of a Parameter function.

    Parameters
    ----------
    func : Callable[[Array[float, ()] | float | int], Array[float, (*shape,)]]
        The function to use to compute the parameter value.

    Returns
    -------
    Dimensions
        The dimensions from the return annotation of the function.

    Examples
    --------
    >>> from unxt import Quantity
    >>> def func(t: Quantity["time"]) -> Quantity["mass"]: pass
    >>> _get_dimensions_from_return_annotation(func)
    PhysicalType('mass')

    >>> import astropy.units as u
    >>> def func(t: u.Quantity["time"]) -> u.Quantity["mass"]: pass
    >>> _get_dimensions_from_return_annotation(func)
    PhysicalType('mass')

    """
    # Get the return annotation
    type_hints = get_type_hints(func, include_extras=True)
    if "return" not in type_hints:
        msg = "Parameter function must have a return annotation"
        raise TypeError(msg)

    ann = type_hints["return"]

    # Get the dimensions from the return annotation
    dims: Dimensions | None = None

    # `unxt.Quantity`
    if isclass(ann) and issubclass(ann, AbstractQuantity):
        dims = ann.type_parameter

    # Astropy compatibility
    elif isannotated(ann):
        args = get_args(ann)

        if (
            len(args) == 2
            and issubclass(args[0], u.Quantity)
            and isinstance(args[1], Dimensions)
        ):
            dims = args[1]

    if dims is None:
        msg = "Parameter function return annotation must be a Quantity"
        raise TypeError(msg)

    return dims
