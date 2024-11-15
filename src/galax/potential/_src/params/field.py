"""Descriptor for a Potential Parameter."""

from __future__ import annotations

__all__ = ["ParameterField"]

from dataclasses import KW_ONLY, is_dataclass
from inspect import isclass, isfunction
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
from typing_extensions import Doc, override

from astropy.units import PhysicalType as Dimension, Quantity as AstropyQuantity

import unxt as u
from dataclassish.converters import Optional
from is_annotated import isannotated
from unxt.quantity import AbstractQuantity, Quantity

from .core import AbstractParameter, ConstantParameter, ParameterCallable, UserParameter
from galax.utils.dataclasses import (
    Sentinel,
    dataclass_with_converter,
    field,
    sentineled,
)

if TYPE_CHECKING:
    from galax.potential import AbstractBasePotential


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
        # `Quantity.from_`` handles errors if the value cannot be
        # converted to a Quantity.
        out = ConstantParameter(Quantity.from_(value))

    return out


@final
@dataclass_with_converter(frozen=True, slots=True)
class ParameterField:
    """Descriptor for a Potential Parameter.

    Parameters
    ----------
    dimensions : PhysicalType
        Dimension (unit-wise) of the parameter.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> class KeplerPotential(gp.AbstractPotential):
    ...     mass: gp.params.ParameterField = gp.params.ParameterField(dimensions="mass")
    ...     def _potential(self, q, t):
    ...         return -self.constants["G"] * self.mass(t) / jnp.linalg.norm(q, axis=-1)

    The `mass` parameter is a `ParameterField` that has dimensions of mass.
    This can be a constant value or a function of time.

    The simplest example is a constant mass:

    >>> potential = KeplerPotential(mass=u.Quantity(1e12, "Msun"), units="galactic")
    >>> potential
    KeplerPotential(
      units=LTMAUnitSystem( length=Unit("kpc"), ...),
      constants=ImmutableMap({'G': ...}),
      mass=ConstantParameter(
        value=Quantity[PhysicalType('mass')](value=...f64[], unit=Unit("solMass"))
      )
    )

    """

    name: str = field(init=False)
    """The name of the parameter."""

    _: KW_ONLY
    default: AbstractParameter | Literal[Sentinel.MISSING] = field(
        default=Sentinel.MISSING,
        converter=sentineled(converter_parameter, sentinel=Sentinel.MISSING),
    )
    """The default value of the parameter."""

    dimensions: Dimension = field(converter=u.dimension)
    """The dimensions (unit-wise) of the parameter."""

    doc: str | None = field(default=None, compare=False, converter=Optional(str))

    # ===========================================
    # Descriptor

    def __set_name__(self, owner: "type[AbstractBasePotential]", name: str) -> None:
        """Set the name of the parameter."""
        object.__setattr__(self, "name", name)

        # Try to get the documentation from the annotation
        ann = owner.__annotations__[name]  # Get the annotation from the class
        if isannotated(ann):
            for arg in get_args(ann)[1:]:
                if isinstance(arg, Doc):
                    object.__setattr__(self, "doc", arg.documentation)

    @property
    @override
    def __doc__(self) -> str | None:  # type: ignore[override]
        """The docstring of the parameter."""
        return self.__doc__ if self.doc is None else self.doc

    # -----------------------------
    # Getting

    @overload  # TODO: use `Self` when beartype is happy
    def __get__(
        self, instance: None, owner: "type[AbstractBasePotential]"
    ) -> "ParameterField": ...

    @overload
    def __get__(
        self, instance: "AbstractBasePotential", owner: None
    ) -> AbstractParameter: ...

    def __get__(  # TODO: use `Self` when beartype is happy
        self,
        instance: "AbstractBasePotential | None",
        owner: "type[AbstractBasePotential] | None",
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
        self, potential: "AbstractBasePotential", dims: Dimension
    ) -> None:
        """Check that the given unit is compatible with the parameter's."""
        # When the potential is being constructed, the units may not have been
        # set yet, so we don't check the unit.
        if not hasattr(potential, "units"):
            return

        # Check the dimensions are compatible
        if dims != self.dimensions:
            msg = (
                "Parameter function must return a value with "
                f"dimensions consistent with {self.dimensions}."
            )
            raise ValueError(msg)

    def __set__(
        self,
        potential: "AbstractBasePotential",
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
            v = ConstantParameter(Quantity.from_(value, unit))

        # Set
        potential.__dict__[self.name] = v


# -------------------------------------------


def _get_dimensions_from_return_annotation(func: ParameterCallable, /) -> Dimension:
    """Get the dimensions from the return annotation of a Parameter function.

    Parameters
    ----------
    func : Callable[[Array[float, ()] | float | int], Array[float, (*shape,)]]
        The function to use to compute the parameter value.

    Returns
    -------
    Dimension
        The dimensions from the return annotation of the function.

    Examples
    --------
    >>> import unxt as u
    >>> def func(t: u.Quantity["time"]) -> u.Quantity["mass"]: pass
    >>> _get_dimensions_from_return_annotation(func)
    PhysicalType('mass')

    >>> import astropy.units as u
    >>> def func(t: u.Quantity["time"]) -> u.Quantity["mass"]: pass
    >>> _get_dimensions_from_return_annotation(func)
    PhysicalType('mass')

    """
    # Get the function, unwarpping if necessary
    func = (
        func.__call__
        if hasattr(func, "__call__") and not isfunction(func)  # noqa: B004
        else func
    )

    # Get the return annotation
    type_hints = get_type_hints(func, include_extras=True)

    if "return" not in type_hints:
        msg = "Parameter function must have a return annotation"
        raise TypeError(msg)

    ann = type_hints["return"]

    # Get the dimensions from the return annotation
    dims: Dimension | None = None

    # `unxt.Quantity`
    if isclass(ann) and issubclass(ann, AbstractQuantity):
        dims = ann.type_parameter

    # Astropy compatibility
    elif isannotated(ann):
        args = get_args(ann)

        if (
            len(args) == 2
            and issubclass(args[0], AstropyQuantity)
            and isinstance(args[1], Dimension)
        ):
            dims = args[1]

    if dims is None:
        msg = "Parameter function return annotation must be a Quantity"
        raise TypeError(msg)

    return dims
