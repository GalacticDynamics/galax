"""Base class for composite potentials."""

__all__ = ["AbstractCompositePotential", "AbstractPreCompositedPotential"]


import functools as ft
import uuid
from collections.abc import Hashable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import MISSING, replace
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import equinox as eqx
import jax
import wadler_lindig as wl
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from dataclassish.flags import FilterRepr
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from .base import AbstractPotential, default_constants

if TYPE_CHECKING:
    import galax.potential  # noqa: ICN001


class AbstractCompositePotential(AbstractPotential):
    """Base class for composite potentials."""

    _data: eqx.AbstractVar[dict[str, AbstractPotential]]

    # === Potential ===

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._potential(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._gradient(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @ft.partial(jax.jit)
    def _laplacian(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._laplacian(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._density(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @ft.partial(jax.jit)
    def _hessian(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz33:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._hessian(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    # ===========================================
    # Collection Protocol

    def __contains__(self, key: Any) -> bool:
        """Check if the key is in the composite potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.CompositePotential(
        ...     disk=gp.KeplerPotential(m_tot=1e11, units="galactic")
        ... )

        >>> "disk" in pot
        True

        """
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ===========================================
    # Mapping Protocol

    def __getitem__(self, key: str) -> AbstractPotential:
        return cast(AbstractPotential, self._data[key])

    def keys(self) -> KeysView[str]:
        return cast(KeysView[str], self._data.keys())

    def values(self) -> ValuesView[AbstractPotential]:
        return cast(ValuesView[AbstractPotential], self._data.values())

    def items(self) -> ItemsView[str, AbstractPotential]:
        return cast(ItemsView[str, AbstractPotential], self._data.items())

    # ===========================================
    # Extending Mapping

    def __or__(self, other: Any) -> "galax.potential.CompositePotential":
        from .composite import CompositePotential

        if not isinstance(other, AbstractPotential):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            self._data
            | (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
        )

    def __ror__(self, other: Any) -> "galax.potential.CompositePotential":
        from .composite import CompositePotential

        if not isinstance(other, AbstractPotential):
            return NotImplemented

        return CompositePotential(  # combine the two dictionaries
            (  # make `other` into a compatible dictionary.
                other._data
                if isinstance(other, CompositePotential)
                else {str(uuid.uuid4()): other}
            )
            | self._data
        )

    # ===========================================
    # Convenience

    def __add__(self, other: AbstractPotential) -> "galax.potential.CompositePotential":
        return self | other


# =================


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePotential, /, **kwargs: Any
) -> AbstractCompositePotential:
    """Replace the parameters of a composite potential.

    Examples
    --------
    >>> from dataclassish import replace
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=1e11, a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=1e12, r_s=20, units="galactic"),
    ... )

    >>> new_pot = replace(pot, disk=gp.MiyamotoNagaiPotential(m_tot=u.Quantity(1e12, "Msun"), a=6.5, b=0.26, units="galactic"))
    >>> new_pot["disk"].m_tot.value
    Quantity(Array(1.e+12, dtype=float64,...), unit='solMass')

    """  # noqa: E501
    # TODO: directly call the Mapping implementation
    extra_keys = set(kwargs) - set(obj)
    kwargs = eqx.error_if(kwargs, any(extra_keys), f"invalid keys {extra_keys}.")

    return type(obj)(**{**obj, **kwargs})


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePotential, replacements: Mapping[str, Mapping[str, Any]], /
) -> AbstractCompositePotential:
    """Replace the parameters of a composite potential.

    Examples
    --------
    >>> from dataclassish import replace
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.CompositePotential(
    ...     disk=gp.MiyamotoNagaiPotential(m_tot=1e11, a=6.5, b=0.26, units="galactic"),
    ...     halo=gp.NFWPotential(m=1e12, r_s=20, units="galactic"),
    ... )

    >>> new_pot = replace(pot, {"disk": {"m_tot": u.Quantity(1e12, "Msun")}})
    >>> new_pot["disk"].m_tot.value
    Quantity(Array(1.e+12, dtype=float64,...), unit='solMass')

    """
    # AbstractCompositePhaseSpaceCoordinate is both a Mapping and a dataclass
    # so we need to disambiguate the method to call
    method = replace.invoke(Mapping[Hashable, Any], Mapping[str, Any])
    return method(obj, replacements)


##############################################################################


class AbstractPreCompositedPotential(AbstractCompositePotential):
    """Base class for pre-defined composite potentials.

    Here the component potentials are defined as fields of the class.

    For examples, see:

    - `galax.potential.BovyMWPotential2014`
    - `galax.potential.LM10Potential`
    - `galax.potential.MilkyWayPotential`
    - `galax.potential.MilkyWayPotential2022`

    """

    _keys: tuple[str, ...] = eqx.field(repr=False, static=True)

    def __init__(
        self,
        mapping: Mapping[str, AbstractPotential] | None = None,
        /,
        *,
        units: Any = MISSING,
        constants: Any = default_constants,
        **kwargs: Any,
    ) -> None:
        # Merge the mapping and kwargs
        kwargs = dict(mapping or {}, **kwargs)

        # Get the fields, for conversion and validation
        fields = self.__dataclass_fields__

        # Units
        self.units = fields["units"].metadata["converter"](
            units if units is not MISSING else fields["units"].default
        )

        # Constants
        # TODO: some similar check that the same constants are the same, e.g.
        #       `G` is the same for all potentials. Or use `constants` to update
        #       the `constants` of every potential (before `super().__init__`)
        constants = fields["constants"].metadata["converter"](constants)
        self.constants = ImmutableMap(
            {k: v.decompose(self.units) for k, v in constants.items()}
        )

        # Initialize the Parameter (potential) fields
        # TODO: more robust detection using the annotations: AbstractParameter
        # or Annotated[AbstractParameter, ...]
        # 1. Check the kwargs vs the fields
        self._keys = tuple(
            k for k, f in fields.items() if isinstance(f.default, AbstractPotential)
        )
        extra_keys = set(kwargs) - set(self._keys)
        if extra_keys:
            msg = f"invalid keys {extra_keys}"
            raise ValueError(msg)
        # 2. Iterate over the fields and set the values
        v: Any
        for k, v in kwargs.items():
            # Either update from the default or try more general conversion.
            pot = (
                replace(fields[k].default, **v)
                if isinstance(v, dict | ImmutableMap)  # type: ignore[redundant-expr]
                else fields[k].metadata["converter"](v)
            )
            setattr(self, k, pot)

    @property
    def _data(self) -> ImmutableMap[str, AbstractPotential]:
        """Return the parameters as an ImmutableMap."""
        return ImmutableMap({k: getattr(self, k) for k in self._keys})

    @override
    def values(self) -> tuple[AbstractPotential, ...]:  # type: ignore[override]
        return tuple(getattr(self, k) for k in self._keys)

    # ===========================================
    # Collection Protocol

    @override
    def __contains__(self, key: str) -> bool:
        """Check if the key is in the composite potential.

        Examples
        --------
        >>> import galax.potential as gp
        >>> pot = gp.MilkyWayPotential()
        >>> "disk" in pot
        True

        """
        return key in self._keys

    @override
    def __iter__(self) -> Iterator[str]:
        """Check if the key is in the composite potential.

        Examples
        --------
        >>> import galax.potential as gp

        >>> pot = gp.MilkyWayPotential()
        >>> tuple(iter(pot))
        ('disk', 'halo', 'bulge', 'nucleus')

        """
        return iter(self._keys)

    @override
    def __len__(self) -> int:
        """Check if the key is in the composite potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.MilkyWayPotential()
        >>> len(pot)
        4

        """
        return len(self._keys)

    # ===========================================
    # Mapping Protocol

    @override
    def __getitem__(self, key: str, /) -> AbstractPotential:
        """Check if the key is in the composite potential.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.MilkyWayPotential()
        >>> pot["disk"]
        MiyamotoNagaiPotential(
            units=...,
            constants=ImmutableMap({'G': ...}),
            m_tot=ConstantParameter(...),
            a=ConstantParameter(...),
            b=ConstantParameter(...)
        )

        """
        key = eqx.error_if(key, key not in self._keys, f"key {key} not found")
        return cast(AbstractPotential, getattr(self, key))

    # ===========================================
    # Wadler-Lindig API

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        """Return the Wadler-Lindig representation of this class.

        This is used for the `__repr__` and `__str__` methods or when using the
        `wadler_lindig` library.

        Examples
        --------
        >>> import galax.potential as gp
        >>> import wadler_lindig as wl

        >>> pot = gp.MilkyWayPotential()
        >>> wl.pprint(pot)
        MilkyWayPotential(
            disk=MiyamotoNagaiPotential(...),
            halo=NFWPotential(...),
            bulge=HernquistPotential(...),
            nucleus=HernquistPotential(...),
            units=...,
            constants=ImmutableMap({'G': ...})
        )

        """
        return wl.bracketed(
            begin=wl.TextDoc(f"{self.__class__.__name__}("),
            docs=wl.named_objs(list(field_items(FilterRepr, self)), **kwargs),
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs.get("indent", 4),
        )
