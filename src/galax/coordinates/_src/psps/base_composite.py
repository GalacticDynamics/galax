"""ABC for composite phase-space positions."""

__all__ = ["AbstractCompositePhaseSpacePosition"]

from abc import abstractmethod
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any, cast

import equinox as eqx
from jaxtyping import Shaped
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_items
from xmmutablemap import ImmutableMap
from zeroth import zeroth

import galax.typing as gt
from .base import AbstractPhaseSpacePosition, ComponentShapeTuple
from .utils import PSPVConvertOptions


# Note: cannot have `strict=True` because of inheriting from ImmutableMap.
class AbstractCompositePhaseSpacePosition(  # type: ignore[misc,unused-ignore]
    AbstractPhaseSpacePosition,
    ImmutableMap[str, AbstractPhaseSpacePosition],  # type: ignore[misc]
    strict=False,  # type: ignore[call-arg]
):
    r"""Abstract base class of composite phase-space positions.

    The composite phase-space position is a point in the 3 spatial + 3 kinematic
    + 1 time -dimensional phase space :math:`\mathbb{R}^7` of a dynamical
    system. It is composed of multiple phase-space positions, each of which
    represents a component of the system.

    The input signature matches that of :class:`dict` (and
    :class:`~xmmutablemap.ImmutableMap`), so you can pass in the components
    as keyword arguments or as a dictionary.

    The components are stored as a dictionary and can be key accessed. However,
    the composite phase-space position itself acts as a single
    `AbstractPhaseSpacePosition` object, so you can access the composite
    positions, velocities, and times as if they were a single object. In this
    base class the composition of the components is abstract and must be
    implemented in the subclasses.

    Examples
    --------
    For this example we will use
    `galax.coordinates.CompositePhaseSpacePosition`.

    >>> from dataclasses import replace
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    >>> def stack(vs: list[cx.vecs.AbstractPos]) -> cx.vecs.AbstractPos:
    ...    comps = {k: jnp.stack([getattr(v, k) for v in vs], axis=-1)
    ...             for k in vs[0].components}
    ...    return replace(vs[0], **comps)

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([10, 20, 30], "kpc"),
    ...                              p=u.Quantity([40, 50, 60], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))

    >>> c_psp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)
    >>> c_psp["psp1"] is psp1
    True

    >>> print(c_psp.q)
    <CartesianPos3D (x[kpc], y[kpc], z[kpc])
        [[ 1  2  3]
         [10 20 30]]>

    Note that the length of the individual components are 0, but the length of
    the composite is the sum of the lengths of the components.

    >>> len(psp1)
    0

    >>> len(c_psp)
    2

    """

    _data: dict[str, AbstractPhaseSpacePosition]

    def __init__(
        self,
        psps: (
            dict[str, AbstractPhaseSpacePosition]
            | tuple[tuple[str, AbstractPhaseSpacePosition], ...]
        ) = (),
        /,
        **kwargs: AbstractPhaseSpacePosition,
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

        >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
        ...                            p=u.Quantity([4, 5, 6], "m/s"),
        ...                            t=u.Quantity(7.0, "s"))
        >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
        ...                            p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
        ...                            t=u.Quantity(6.0, "s"))

        >>> cw = gc.CompositePhaseSpacePosition(w1=w1, w2=w2)
        >>> cw._shape_tuple
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

    # ---------------------------------------------------------------
    # Getitem

    @AbstractPhaseSpacePosition.__getitem__.dispatch
    def __getitem__(
        self: "AbstractCompositePhaseSpacePosition", key: Any
    ) -> "AbstractCompositePhaseSpacePosition":
        """Get item from the key.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
        ...                            p=u.Quantity([4, 5, 6], "m/s"),
        ...                            t=u.Quantity(7, "s"))
        >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
        ...                            p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
        ...                            t=u.Quantity(6, "s"))
        >>> cw = gc.CompositePhaseSpacePosition(w1=w1, w2=w2)

        >>> cw[...]
        CompositePhaseSpacePosition({'w1': PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()
          ), 'w2': PhaseSpacePosition(
            q=CartesianPos3D( ... ),
            p=CartesianVel3D( ... ),
            t=Quantity['time'](Array(6, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()
        )})

        """
        # Get from each value, e.g. a slice
        return type(self)(**{k: v[key] for k, v in self.items()})

    @AbstractPhaseSpacePosition.__getitem__.dispatch
    def __getitem__(
        self: "AbstractCompositePhaseSpacePosition", key: str
    ) -> AbstractPhaseSpacePosition:
        """Get item from the key.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc

        >>> w1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
        ...                            p=u.Quantity([4, 5, 6], "m/s"),
        ...                            t=u.Quantity(7.0, "s"))
        >>> w2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
        ...                            p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
        ...                            t=u.Quantity(6.0, "s"))
        >>> cw = gc.CompositePhaseSpacePosition(w1=w1, w2=w2)

        >>> cw["w1"] is w1
        True

        """
        return self._data[key]

    # ===============================================================
    # Python API

    def __repr__(self) -> str:  # TODO: not need this hack
        return cast(str, ImmutableMap.__repr__(self))

    # ===============================================================
    # Collection methods

    @property
    def shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Get the shapes of the components."""
        return MappingProxyType({k: v.shape for k, v in field_items(self)})


# =============================================================================
# Dispatches

# =================
# `unxt.uconvert` dispatches


@dispatch(precedence=1)  # type: ignore[call-overload,misc]  # TODO: make precedence=0
def uconvert(
    usys: u.AbstractUnitSystem | str, cpsp: AbstractCompositePhaseSpacePosition
) -> AbstractCompositePhaseSpacePosition:
    """Convert the components to the given units.

    Examples
    --------
    For this example we will use
    `galax.coordinates.CompositePhaseSpacePosition`.

    >>> import unxt as u
    >>> from unxt.unitsystems import solarsystem
    >>> import galax.coordinates as gc

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                              p=u.Quantity([4, 5, 6], "km/s"),
    ...                              t=u.Quantity(7, "Myr"))

    >>> c_psp = gc.CompositePhaseSpacePosition(psp1=psp1)
    >>> c_psp.uconvert(solarsystem)
    CompositePhaseSpacePosition({'psp1': PhaseSpacePosition(
        q=CartesianPos3D(
            x=Quantity[...](value=...f64[], unit=Unit("AU")),
            ...

    """
    return type(cpsp)(**{k: v.uconvert(usys) for k, v in cpsp.items()})


# =================
# `coordinax.vconvert` dispatches


@dispatch
def vconvert(
    target: PSPVConvertOptions,
    psps: AbstractCompositePhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractCompositePhaseSpacePosition:
    """Return with the components transformed.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7, "s"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6, "s"))
    >>> cpsp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)

    We can transform the composite phase-space position to a new position class.

    >>> cx.vconvert({"q": cx.vecs.CylindricalPos, "p": cx.vecs.SphericalVel}, cpsp)
    CompositePhaseSpacePosition({'psp1': PhaseSpacePosition(
            q=CylindricalPos( ... ),
            p=SphericalVel( ... ),
            t=Quantity['time'](Array(7, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()
        ),
        'psp2': PhaseSpacePosition(
            q=CylindricalPos( ... ),
            p=SphericalVel( ... ),
            t=Quantity['time'](Array(6, dtype=int64, ...), unit='s'),
            frame=SimulationFrame()
    )})

    """
    q_cls = target["q"]
    target = {
        "q": q_cls,
        "p": q_cls.time_derivative_cls if (p_cls := target.get("p")) is None else p_cls,
    }

    # TODO: use `dataclassish.replace`
    return type(psps)(
        **{k: cx.vconvert(target, psp, **kwargs) for k, psp in psps.items()}
    )


@dispatch
def vconvert(
    target_position_cls: type[cx.vecs.AbstractPos],
    psps: AbstractCompositePhaseSpacePosition,
    /,
    **kwargs: Any,
) -> AbstractCompositePhaseSpacePosition:
    """Return with the components transformed.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.coordinates as gc

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7, "s"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6, "s"))
    >>> cpsp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)

    We can transform the composite phase-space position to a new position class.

    >>> cx.vconvert(cx.vecs.CylindricalPos, cpsp)
    CompositePhaseSpacePosition({'psp1': PhaseSpacePosition(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=Quantity...
      ),
      'psp2': PhaseSpacePosition(
        q=CylindricalPos( ... ),
        p=CylindricalVel( ... ),
        t=...
    )})

    """
    target = {"q": target_position_cls, "p": target_position_cls.time_derivative_cls}
    return vconvert(target, psps, **kwargs)


# =================


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePhaseSpacePosition, /, **kwargs: Any
) -> AbstractCompositePhaseSpacePosition:
    """Replace the components of the composite phase-space position.

    Examples
    --------
    >>> import galax.coordinates as gc
    >>> from dataclassish import replace

    We define a composite phase-space position with two components.
    Every component is a phase-space position in Cartesian coordinates.

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7.0, "s"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6.0, "s"))
    >>> cpsp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)

    We can replace the components of the composite phase-space position.

    >>> cpsp2 = replace(cpsp, psp1=psp2, psp2=psp1)

    >>> cpsp2["psp1"] != psp1
    True

    >>> cpsp2["psp1"] == psp2
    Array(True, dtype=bool)

    >>> cpsp2["psp2"] == psp1
    Array(True, dtype=bool)

    """
    # TODO: directly call the Mapping implementation
    extra_keys = set(kwargs) - set(obj)
    kwargs = eqx.error_if(kwargs, any(extra_keys), f"invalid keys {extra_keys}.")

    return type(obj)(**{**obj, **kwargs})


@dispatch(precedence=1)
def replace(
    obj: AbstractCompositePhaseSpacePosition,
    replacements: Mapping[str, Any],
    /,
) -> AbstractCompositePhaseSpacePosition:
    """Replace the components of the composite phase-space position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> from dataclassish import replace

    We define a composite phase-space position with two components. Every
    component is a phase-space position in Cartesian coordinates.

    >>> psp1 = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "m"),
    ...                              p=u.Quantity([4, 5, 6], "m/s"),
    ...                              t=u.Quantity(7.0, "s"))
    >>> psp2 = gc.PhaseSpacePosition(q=u.Quantity([1.5, 2.5, 3.5], "m"),
    ...                              p=u.Quantity([4.5, 5.5, 6.5], "m/s"),
    ...                              t=u.Quantity(6.0, "s"))
    >>> cpsp = gc.CompositePhaseSpacePosition(psp1=psp1, psp2=psp2)

    We can selectively replace the ``t`` component of each constituent
    phase-space position.

    >>> cpsp2 = replace(cpsp, {"psp1": {"t": u.Quantity(10.0, "s")},
    ...                        "psp2": {"t": u.Quantity(11.0, "s")}})

    """
    # AbstractCompositePhaseSpacePosition is both a Mapping and a dataclass
    # so we need to disambiguate the method to call
    method = replace.invoke(Mapping[Hashable, Any], Mapping[str, Any])
    return method(obj, replacements)
