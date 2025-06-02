"""ABC for composite phase-space positions."""

__all__ = ["AbstractCompositePhaseSpaceCoordinate"]

from abc import abstractmethod
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, cast

import equinox as eqx
from jaxtyping import Shaped
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from xmmutablemap import ImmutableMap
from zeroth import zeroth

import galax._custom_types as gt
from .base import AbstractPhaseSpaceCoordinate, ComponentShapeTuple
from galax.coordinates._src.base import AbstractPhaseSpaceObject
from galax.coordinates._src.utils import PSPVConvertOptions


# Note: cannot have `strict=True` because of inheriting from ImmutableMap.
class AbstractCompositePhaseSpaceCoordinate(  # type: ignore[misc,unused-ignore]
    AbstractPhaseSpaceCoordinate,
    ImmutableMap[str, AbstractPhaseSpaceCoordinate],  # type: ignore[misc]
    strict=False,  # type: ignore[call-arg]
):
    r"""Abstract base class of composite phase-space coordinates.

    The composite phase-space coordinate is a point in the 3 spatial + 3
    kinematic + 1 time -dimensional phase space :math:`\mathbb{R}^7` of a
    dynamical system. It is composed of multiple phase-space coordinates, each
    of which represents a component of the system.

    The input signature matches that of :class:`dict` (and
    :class:`~xmmutablemap.ImmutableMap`), so you can pass in the components as
    keyword arguments or as a dictionary.

    The components are stored as a dictionary and can be key accessed. However,
    the composite phase-space coordinate itself acts as a single
    `AbstractPhaseSpaceCoordinate` object, so you can access the composite
    positions, velocities, and times as if they were a single object. In this
    base class the composition of the components is abstract and must be
    implemented in the subclasses.

    Examples
    --------
    For this example we will use
    `galax.coordinates.CompositePhaseSpaceCoordinate`.

    >>> from dataclasses import replace
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> def stack(vs: list[cx.vecs.AbstractPos]) -> cx.vecs.AbstractPos:
    ...    comps = {k: jnp.stack([getattr(v, k) for v in vs], axis=-1)
    ...             for k in vs[0].components}
    ...    return replace(vs[0], **comps)

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                               p=u.Quantity([4, 5, 6], "km/s"),
    ...                               t=u.Quantity(7, "Myr"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([10, 20, 30], "kpc"),
    ...                               p=u.Quantity([40, 50, 60], "km/s"),
    ...                               t=u.Quantity(7, "Myr"))

    >>> cwt = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)
    >>> cwt["wt1"] is wt1
    True

    >>> print(cwt.q)
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [[ 1  2  3]
         [10 20 30]]>

    Note that the length of the individual components are 0, but the length of
    the composite is the sum of the lengths of the components.

    >>> len(wt1)
    0

    >>> len(cwt)
    2

    """

    _data: dict[str, AbstractPhaseSpaceCoordinate]

    _GETITEM_DYNAMIC_FILTER_SPEC: ClassVar = None  # TODO: use this in getitem
    _GETITEM_TIME_FILTER_SPEC: ClassVar = None

    def __init__(
        self,
        psps: (
            dict[str, AbstractPhaseSpaceCoordinate]
            | tuple[tuple[str, AbstractPhaseSpaceCoordinate], ...]
        ) = (),
        /,
        **kwargs: AbstractPhaseSpaceCoordinate,
    ) -> None:
        ImmutableMap.__init__(self, psps, **kwargs)  # <- ImmutableMap.__init__

    # ==========================================================================
    # PSP API

    @property
    @abstractmethod
    def q(self) -> cx.vecs.AbstractPos3D:
        """Positions."""

    @property
    @abstractmethod
    def p(self) -> cx.vecs.AbstractVel3D:
        """Conjugate momenta."""

    @property
    @abstractmethod
    def t(self) -> Shaped[u.Quantity["time"], "..."]:
        """Times."""

    # ==========================================================================
    # Array API

    @property
    def _shape_tuple(self) -> tuple[gt.Shape, ComponentShapeTuple]:
        """Batch and component shapes.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.coordinates as gc

        >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
        ...                               p=u.Quantity([4, 5, 6], "m/s"),
        ...                               t=u.Quantity(7.0, "s"))
        >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
        ...                               p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
        ...                               t=u.Quantity(6.0, "s"))

        >>> cwt = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)
        >>> cwt._shape_tuple
        ((2,), ComponentShapeTuple(q=3, p=3, t=1))
        """
        # TODO: speed up
        batch_shape = jnp.broadcast_shapes(*[psp.shape for psp in self.values()])
        if not batch_shape:
            batch_shape = (len(self),)
        else:
            batch_shape = (*batch_shape[:-1], len(self._data) * batch_shape[-1])
        shape = zeroth(self.values())._shape_tuple[1]  # noqa: SLF001
        return batch_shape, shape

    def __len__(self) -> int:
        # Length is the sum of the lengths of the components.
        # For length-0 components, we assume a length of 1.
        return sum([len(w) or 1 for w in self.values()])

    # ===============================================================
    # Python API

    def __repr__(self) -> str:  # TODO: not need this hack
        return cast(str, ImmutableMap.__repr__(self))

    # ===============================================================
    # Collection methods

    @property
    def shapes(self) -> Mapping[str, gt.Shape]:
        """Get the shapes of the components."""
        return MappingProxyType({k: v.shape for k, v in field_items(self)})


#####################################################################
# Dispatches

# ===============================================================
# `__getitem__`


@AbstractPhaseSpaceObject.__getitem__.dispatch
def getitem(
    self: AbstractCompositePhaseSpaceCoordinate, key: Any
) -> AbstractCompositePhaseSpaceCoordinate:
    """Get item from the key.

    This is the default dispatch for composite PSC objects, passing the key to
    each component.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7, "s"))
    >>> w2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6, "s"))
    >>> cw = gc.CompositePhaseSpaceCoordinate(w1=w1, w2=w2)

    >>> print(cw[...])
    CompositePhaseSpaceCoordinate(
        w1=PhaseSpaceCoordinate(
            q=<CartesianPos3D (x[m], y[m], z[m])
                [1 2 3]>,
            p=<CartesianVel3D (x[m / s], y[m / s], z[m / s])
                [4 5 6]>,
            t=Quantity(Array(7, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()),
        w2=PhaseSpaceCoordinate(
            q=<CartesianPos3D (x[m], y[m], z[m])
                [1.5 2.5 3.5]>,
            p=<CartesianVel3D (x[m / s], y[m / s], z[m / s])
                [4.5 5.5 6.5]>,
            t=Quantity(Array(6, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()))

    """
    # Get from each value, e.g. a slice
    return type(self)(**{k: v[key] for k, v in self.items()})


@AbstractPhaseSpaceObject.__getitem__.dispatch
def getitem(
    self: AbstractCompositePhaseSpaceCoordinate, key: str
) -> AbstractPhaseSpaceCoordinate:
    """Get item from the key.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> w1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7.0, "s"))
    >>> w2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6.0, "s"))
    >>> cw = gc.CompositePhaseSpaceCoordinate(w1=w1, w2=w2)

    >>> cw["w1"] is w1
    True

    """
    return self._data[key]


# ===============================================================
# `unxt.uconvert`


@dispatch(precedence=1)  # type: ignore[call-overload,misc]  # TODO: make precedence=0
def uconvert(
    usys: u.AbstractUnitSystem | str, cwt: AbstractCompositePhaseSpaceCoordinate, /
) -> AbstractCompositePhaseSpaceCoordinate:
    """Convert the components to the given units.

    Examples
    --------
    For this example we will use
    `galax.coordinates.CompositePhaseSpaceCoordinate`.

    >>> import unxt as u
    >>> import galax.coordinates as gc

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))

    >>> cwt = gc.CompositePhaseSpaceCoordinate(wt1=wt1)
    >>> cwt.uconvert(u.unitsystems.solarsystem)
    CompositePhaseSpaceCoordinate({'wt1': PhaseSpaceCoordinate(
        q=CartesianPos3D(
            x=Quantity(2.06264806e+08, unit='AU'),
            ...

    """
    return type(cwt)(**{k: v.uconvert(usys) for k, v in cwt.items()})


# ===============================================================
# `coordinax.vconvert`


@dispatch
def vconvert(
    target: PSPVConvertOptions,
    cwt: AbstractCompositePhaseSpaceCoordinate,
    /,
    **kwargs: Any,
) -> AbstractCompositePhaseSpaceCoordinate:
    """Return with the components transformed.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7, "s"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6, "s"))
    >>> cpsp = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)

    We can transform the composite phase-space position to a new position class.

    >>> cx.vconvert({"q": cx.vecs.CylindricalPos, "p": cx.vecs.SphericalVel}, cpsp)
    CompositePhaseSpaceCoordinate({'wt1': PhaseSpaceCoordinate(
            q=CylindricalPos( ... ),
            p=SphericalVel( ... ),
            t=Quantity(7, unit='s'),
            frame=SimulationFrame()
        ),
        'wt2': PhaseSpaceCoordinate(
            q=CylindricalPos( ... ),
            p=SphericalVel( ... ),
            t=Quantity(6, unit='s'),
            frame=SimulationFrame()
    )})

    """
    q_cls = target["q"]
    target = {
        "q": q_cls,
        "p": q_cls.time_derivative_cls if (p_cls := target.get("p")) is None else p_cls,
    }

    # TODO: use `dataclassish.replace`
    return type(cwt)(**{k: cx.vconvert(target, wt, **kwargs) for k, wt in cwt.items()})


@dispatch
def vconvert(
    target_position_cls: type[cx.vecs.AbstractPos],
    cwt: AbstractCompositePhaseSpaceCoordinate,
    /,
    **kwargs: Any,
) -> AbstractCompositePhaseSpaceCoordinate:
    """Return with the components transformed.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                               p=u.Quantity([4, 5, 6], "m/s"),
    ...                               t=u.Quantity(7, "s"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                               p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                               t=u.Quantity(6, "s"))
    >>> cpsp = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)

    We can transform the composite phase-space position to a new position class.

    >>> cx.vconvert(cx.vecs.CylindricalPos, cpsp)
    CompositePhaseSpaceCoordinate({'wt1': PhaseSpaceCoordinate(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=Quantity...
      ),
      'wt2': PhaseSpaceCoordinate(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=...
    )})

    """
    target = {"q": target_position_cls, "p": target_position_cls.time_derivative_cls}
    return vconvert(target, cwt, **kwargs)


# ===============================================================
# `dataclassish.replace`


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePhaseSpaceCoordinate, /, **kwargs: Any
) -> AbstractCompositePhaseSpaceCoordinate:
    """Replace the components of the composite phase-space position.

    Examples
    --------
    >>> import galax.coordinates as gc
    >>> from dataclassish import replace

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7.0, "s"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6.0, "s"))
    >>> cpsp = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)

    We can replace the components of the composite phase-space position.

    >>> cwt2 = replace(cpsp, wt1=wt2, wt2=wt1)

    >>> cwt2["wt1"] != wt1
    True

    >>> cwt2["wt1"] == wt2
    Array(True, dtype=bool)

    >>> cwt2["wt2"] == wt1
    Array(True, dtype=bool)

    """
    # TODO: directly call the Mapping implementation
    extra_keys = set(kwargs) - set(obj)
    kwargs = eqx.error_if(kwargs, any(extra_keys), f"invalid keys {extra_keys}.")

    return type(obj)(**{**obj, **kwargs})


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePhaseSpaceCoordinate,
    replacements: Mapping[str, Any],
    /,
) -> AbstractCompositePhaseSpaceCoordinate:
    """Replace the components of the composite phase-space position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> from dataclassish import replace

    We define a composite phase-space position with two components. Every
    component is a phase-space position in Cartesian coordinates.

    >>> wt1 = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7.0, "s"))
    >>> wt2 = gc.PhaseSpaceCoordinate(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6.0, "s"))
    >>> cpsp = gc.CompositePhaseSpaceCoordinate(wt1=wt1, wt2=wt2)

    We can selectively replace the ``t`` component of each constituent
    phase-space position.

    >>> cwt2 = replace(cpsp, {"wt1": {"t": u.Quantity(10.0, "s")},
    ...                        "wt2": {"t": u.Quantity(11.0, "s")}})

    """
    # AbstractCompositePhaseSpaceCoordinate is both a Mapping and a dataclass
    # so we need to disambiguate the method to call
    method = replace.invoke(Mapping[Hashable, Any], Mapping[str, Any])
    return method(obj, replacements)
