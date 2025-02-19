"""Base class for composite potentials."""

__all__ = ["AbstractCompositePotential"]


import uuid
from collections.abc import Hashable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
from .base import AbstractPotential
from galax.utils._unxt import AllowValue

if TYPE_CHECKING:
    import galax.potential  # noqa: ICN001


# Note: cannot have `strict=True` because of inheriting from ImmutableMap.
class AbstractCompositePotential(AbstractPotential):
    """Base class for composite potentials."""

    _data: eqx.AbstractVar[dict[str, AbstractPotential]]

    # === Potential ===

    @partial(jax.jit)
    def _potential(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._potential(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @partial(jax.jit)
    def _gradient(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz3:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._gradient(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @partial(jax.jit)
    def _laplacian(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._laplacian(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @partial(jax.jit)
    def _density(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        return jnp.sum(
            jnp.array([p._density(xyz, t) for p in self.values()]),  # noqa: SLF001
            axis=0,
        )

    @partial(jax.jit)
    def _hessian(
        self, xyz: gt.BBtQuSz3 | gt.BBtSz3, t: gt.BBtRealQuSz0 | gt.BBtRealSz0, /
    ) -> gt.BBtSz33:
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
    Quantity['mass'](Array(1.e+12, dtype=float64,...), unit='solMass')

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
    Quantity['mass'](Array(1.e+12, dtype=float64,...), unit='solMass')

    """
    # AbstractCompositePhaseSpaceCoordinate is both a Mapping and a dataclass
    # so we need to disambiguate the method to call
    method = replace.invoke(Mapping[Hashable, Any], Mapping[str, Any])
    return method(obj, replacements)
