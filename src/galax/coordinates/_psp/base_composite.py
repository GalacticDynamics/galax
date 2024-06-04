"""ABC for composite phase-space positions."""

__all__ = ["AbstractCompositePhaseSpacePosition"]

from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from jaxtyping import Shaped
from plum import dispatch

import coordinax as cx
import quaxed.numpy as qnp
from unxt import Quantity

import galax.typing as gt
from .base import AbstractBasePhaseSpacePosition, ComponentShapeTuple
from galax.utils import ImmutableDict
from galax.utils._misc import zeroth
from galax.utils.dataclasses import dataclass_items

if TYPE_CHECKING:
    from typing import Self


# Note: cannot have `strict=True` because of inheriting from ImmutableDict.
class AbstractCompositePhaseSpacePosition(
    ImmutableDict[AbstractBasePhaseSpacePosition],  # TODO: as a TypeVar
    AbstractBasePhaseSpacePosition,
    strict=False,  # type: ignore[call-arg]
):
    r"""Abstract base class of composite phase-space positions.

    The composite phase-space position is a point in the 3 spatial + 3 kinematic
    + 1 time -dimensional phase space :math:`\mathbb{R}^7` of a dynamical
    system. It is composed of multiple phase-space positions, each of which
    represents a component of the system.

    The input signature matches that of :class:`dict` (and
    :class:`~galax.utils.ImmutableDict`), so you can pass in the components as
    keyword arguments or as a dictionary.

    The components are stored as a dictionary and can be key accessed. However,
    the composite phase-space position itself acts as a single
    `AbstractBasePhaseSpacePosition` object, so you can access the composite
    positions, velocities, and times as if they were a single object. In this
    base class the composition of the components is abstract and must be
    implemented in the subclasses.

    Examples
    --------
    >>> from dataclasses import replace
    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> def stack(vs: list[cx.AbstractPosition]) -> cx.AbstractPosition:
    ...    comps = {k: xp.stack([getattr(v, k) for v in vs], axis=-1)
    ...             for k in vs[0].components}
    ...    return replace(vs[0], **comps)

    >>> class CompositePhaseSpacePosition(gc.AbstractCompositePhaseSpacePosition):
    ...     @property
    ...     def q(self) -> cx.AbstractPosition3D:
    ...         return stack([psp.q for psp in self.values()])
    ...
    ...     @property
    ...     def p(self) -> cx.AbstractPosition3D:
    ...         return stack([psp.p for psp in self.values()])
    ...
    ...     @property
    ...     def t(self) -> Shaped[Quantity["time"], "..."]:
    ...         return stack([psp.t for psp in self.values()])

    >>> psp1 = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
    ...                              p=Quantity([4, 5, 6], "km/s"),
    ...                              t=Quantity(7, "Myr"))
    >>> psp2 = gc.PhaseSpacePosition(q=Quantity([10, 20, 30], "kpc"),
    ...                              p=Quantity([40, 50, 60], "km/s"),
    ...                              t=Quantity(7, "Myr"))

    >>> c_psp = CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)
    >>> c_psp["psp1"] is psp1
    True

    >>> c_psp.q
    CartesianPosition3D(
      x=Quantity[...](value=f64[2], unit=Unit("kpc")),
      y=Quantity[...](value=f64[2], unit=Unit("kpc")),
      z=Quantity[...](value=f64[2], unit=Unit("kpc"))
    )

    >>> c_psp.p.d_x
    Quantity['speed'](Array([ 4., 40.], dtype=float64), unit='km / s')
    """

    _data: dict[str, AbstractBasePhaseSpacePosition]

    def __init__(
        self,
        psps: (
            dict[str, AbstractBasePhaseSpacePosition]
            | tuple[tuple[str, AbstractBasePhaseSpacePosition], ...]
        ) = (),
        /,
        **kwargs: AbstractBasePhaseSpacePosition,
    ) -> None:
        super().__init__(psps, **kwargs)  # <- ImmutableDict.__init__

    @property
    @abstractmethod
    def q(self) -> cx.AbstractPosition3D:
        """Positions."""

    @property
    @abstractmethod
    def p(self) -> cx.AbstractPosition3D:
        """Conjugate momenta."""

    @property
    @abstractmethod
    def t(self) -> Shaped[Quantity["time"], "..."]:
        """Times."""

    # ==========================================================================
    # Array properties

    def __getitem__(self, key: Any) -> "Self":
        """Get item from the key."""
        # Get specific item
        if isinstance(key, str):
            return self._data[key]

        # Get from each value, e.g. a slice
        return type(self)(**{k: v[key] for k, v in self.items()})

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch and component shapes."""
        # TODO: speed up
        batch_shape = qnp.broadcast_shapes(*[psp.shape for psp in self.values()])
        batch_shape = (*batch_shape[:-1], len(self) * batch_shape[-1])
        shape = zeroth(self.values())._shape_tuple[1]  # noqa: SLF001
        return batch_shape, shape

    # ==========================================================================
    # Convenience methods

    def to_units(self, units: Any) -> "Self":
        return type(self)(**{k: v.to_units(units) for k, v in self.items()})

    # ===============================================================
    # Collection methods

    @property
    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Get the shapes of the components."""
        return MappingProxyType({k: v.shape for k, v in dataclass_items(self)})


# =============================================================================
# helper functions


# Register AbstractCompositePhaseSpacePosition with `coordinax.represent_as`
@dispatch  # type: ignore[misc]
def represent_as(
    psp: AbstractCompositePhaseSpacePosition,
    position_cls: type[cx.AbstractPosition],
    /,
    differential: type[cx.AbstractVelocity] | None = None,
) -> AbstractCompositePhaseSpacePosition:
    """Return with the components transformed.

    Parameters
    ----------
    psp : :class:`~galax.coordinates.AbstractCompositePhaseSpacePosition`
        The phase-space position.
    position_cls : type[:class:`~vector.AbstractPosition`]
        The target position class.
    differential : type[:class:`~vector.AbstractVelocity`], optional
        The target differential class. If `None` (default), the differential
        class of the target position class is used.

    Examples
    --------
    TODO

    """
    differential_cls = (
        position_cls.differential_cls if differential is None else differential
    )
    return type(psp)(
        **{k: represent_as(v, position_cls, differential_cls) for k, v in psp.items()}
    )
