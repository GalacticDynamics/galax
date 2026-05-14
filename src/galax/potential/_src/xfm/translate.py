"""Triaxial transformation on a potential."""

__all__ = ["TranslatedPotential", "TimeDependentTranslationParameter"]

import functools as ft
from dataclasses import KW_ONLY

from collections.abc import Callable
from jaxtyping import Array, Real
from typing import Any, final

import equinox as eqx
import interpax
import jax
from plum import dispatch

import unxt as u
from unxt.quantity import AllowValue

import galax._custom_types as gt
from .base import AbstractTransformedPotential
from galax.dynamics._src.utils import cond_reverse
from galax.potential._src.base import AbstractPotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class TranslatedPotential(AbstractTransformedPotential):
    r"""Translate a potential along the x, y, and z axes.

    See Also
    --------
    galax.potential.TransformedPotential : use a
        `coordinax.ops.AbstractOperator` to perform an arbitrary transformation.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    The base potential is a Kepler potential:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    We will compare the potential at a specific position before and after
    translation.

    >>> xyz = u.Q([10.0, 0.0, 0.0], "kpc")
    >>> t = 0.0

    For reference, the potential at the specific position is:

    >>> pot.potential(xyz, t)
    Q(-0.44985022, 'kpc2 / Myr2')

    Now we will translate the potential by 1 kpc in the y-direction:

    >>> delta = u.Q([0.0, 1.0, 0.0], "kpc")
    >>> pot_delta = gp.TranslatedPotential(pot, translation=delta)
    >>> pot_delta.potential(xyz, t)
    Q(-0.44761769, 'kpc2 / Myr2')

    To show that the translation is along the y-axis, we can translate the
    position by the same amount:

    >>> pot.potential(xyz - delta, t)
    Q(-0.44761769, 'kpc2 / Myr2')

    We can make a time-dependent translation. This one is very simple, but of
    course it could be more complicated, like the trajectory of an orbit.

    >>> path_t = u.Q(jnp.linspace(0, 1, 200), "Gyr")
    >>> trajectory = u.Q(
    ...     jnp.stack([path_t.ustrip("Myr"),
    ...                jnp.zeros(path_t.shape),
    ...                jnp.zeros(path_t.shape)], axis=-1),
    ...     "kpc")

    >>> delta = gp.params.TimeDependentTranslationParameter.from_(
    ...     path_t, trajectory, units=pot.units)
    >>> pot_tdelta = gp.TranslatedPotential(base_potential=pot,
    ...                                     translation=delta)

    >>> pot_tdelta.potential(xyz, t)
    Q(-0.44985022, 'kpc2 / Myr2')

    >>> pot_tdelta.potential(xyz, u.Q(100, "Myr"))
    Q(-0.04998336, 'kpc2 / Myr2')

    """

    #: the original potential
    base_potential: AbstractPotential

    translation: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        ul = self.units["length"]
        xyz = u.ustrip(AllowValue, ul, xyz.astype(float))

        # Translate the position.
        delta = self.translation(t)
        delta = u.ustrip(AllowValue, ul, delta.astype(float))
        xyz = xyz - delta

        # Evaluate the potential energy at the transformed position, time.
        return self.base_potential._potential(xyz, t)  # noqa: SLF001


# ============================================================================


@final
class TimeDependentTranslationParameter(AbstractParameter):
    r"""Translate a potential along the x, y, and z axes.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    The base potential is a Kepler potential:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    We will compare the potential at a specific position before and after
    translation.

    >>> xyz = u.Q([10.0, 0.0, 0.0], "kpc")
    >>> t = 0.0

    For reference, the potential at the specific position is:

    >>> pot.potential(xyz, t)
    Q(-0.44985022, 'kpc2 / Myr2')

    We can make a time-dependent translation. This one is very simple, but of
    course it could be more complicated, like the trajectory of an orbit.

    >>> path_t = u.Q(jnp.linspace(0, 1, 200), "Gyr")
    >>> trajectory = u.Q(
    ...     jnp.stack([path_t.ustrip("Myr"),
    ...                jnp.zeros(path_t.shape),
    ...                jnp.zeros(path_t.shape)], axis=-1),
    ...     "kpc")

    >>> delta = gp.params.TimeDependentTranslationParameter.from_(
    ...     path_t, trajectory, units=pot.units)
    >>> pot_tdelta = gp.TranslatedPotential(base_potential=pot,
    ...                                     translation=delta)

    >>> pot_tdelta.potential(xyz, t)
    Q(-0.44985022, 'kpc2 / Myr2')

    >>> pot_tdelta.potential(xyz, u.Q(100, "Myr"))
    Q(-0.04998336, 'kpc2 / Myr2')

    """

    translation: Callable[[gt.BBtSz0], gt.BBtSz3]
    """The translation function (t) -> (x, y, z)."""

    _: KW_ONLY

    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)

    def __call__(
        self,
        t: gt.BBtQorVSz0,
        *,
        ustrip: u.AbstractUnit | None = None,
        **_: Any,
    ) -> gt.BBtQuSz3:
        t = u.ustrip(u.quantity.AllowValue, self.units["time"], t)
        out = u.Q.from_(self.translation(t), self.units["length"])
        return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)

    # ---------------------------------
    # Constructors

    @classmethod
    @dispatch
    def from_(
        cls: "type[TimeDependentTranslationParameter]",
        t: Real[Array, "time"],
        xyz: Real[Array, "time 3"],
        /,
        *,
        units: u.AbstractUnitSystem,
    ) -> "TimeDependentTranslationParameter":
        # Parse inputs
        pred = t[1] < t[0]
        t, xyz = cond_reverse(pred, t), cond_reverse(pred, xyz)
        # Interpolate the translation
        spl = interpax.CubicSpline(t, xyz, extrapolate=False)
        # Return the translation parameter
        return cls(translation=spl, units=units)

    @classmethod
    @dispatch
    def from_(
        cls: "type[TimeDependentTranslationParameter]",
        t: Real[u.AbstractQuantity, "time"],
        xyz: Real[u.AbstractQuantity, "time 3"],
        /,
        *,
        units: u.AbstractUnitSystem | None = None,
    ) -> "TimeDependentTranslationParameter":
        usys = units if units is not None else u.unitsystem(t.unit, xyz.unit)
        return cls.from_(
            t.ustrip(usys["time"]), xyz.ustrip(usys["length"]), units=units
        )
