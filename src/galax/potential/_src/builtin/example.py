"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class HarmonicOscillatorPotential(AbstractSinglePotential):
    r"""Harmonic Oscillator Potential.

    Represents an N-dimensional harmonic oscillator.

    .. math::

        \Phi(\mathbf{q}, t) = \frac{1}{2} |\omega(t) \cdot \mathbf{q}|^2

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HarmonicOscillatorPotential(omega=u.Quantity(1, "1 / Myr"),
    ...                                      units="galactic")
    >>> pot
    HarmonicOscillatorPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      omega=ConstantParameter(value=Quantity(weak_i64[], unit='1 / Myr'))
    )

    >>> q = u.Quantity([1.0, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")

    >>> pot.potential(q, t)
    Quantity(Array(0.5, dtype=float64), unit='kpc2 / Myr2')

    >>> pot.density(q, t)
    Quantity(Array(1.76897707e+10, dtype=float64), unit='solMass / kpc3')

    """

    # TODO: enable omega to be a 3D vector
    omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="frequency", doc="The frequency."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        # \Phi(\mathbf{q}, t) = \frac{1}{2} |\omega(t) \cdot \mathbf{q}|^2
        omega = self.omega(t, ustrip=self.units["frequency"])

        return 0.5 * jnp.sum(jnp.square(jnp.atleast_1d(omega) * xyz), axis=-1)

    @ft.partial(jax.jit)
    def _density(self, _: gt.BBtQorVSz3, t: gt.BtQuSz0 | gt.BtSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        omega = jnp.atleast_1d(self.omega(t, ustrip=self.units["frequency"]))

        # \rho(\mathbf{q}, t) = \frac{1}{4 \pi G} \sum_i \omega_i^2
        denom = 4 * jnp.pi * self.constants["G"].value
        return jnp.sum(omega**2, axis=-1) / denom


# -------------------------------------------------------------------


@final
class HenonHeilesPotential(AbstractSinglePotential):
    r"""Henon-Heiles Potential.

    This is a modified version of the [classical Henon-Heiles
    potential](https://en.wikipedia.org/wiki/Hénon-Heiles_system).

    .. math::

        \Phi * t_s^2 = \frac{1}{2} (x^2 + y^2) + \lambda (x^2 y - y^3 / 3)

    Note the timescale :math:`t_s` is introduced to convert the potential to
    specific energy, from the classical area units.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HenonHeilesPotential(coeff=u.Quantity(1, "1 / kpc"),
    ...                               timescale=u.Quantity(1, "Myr"),
    ...                               units="galactic")
    >>> pot
    HenonHeilesPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      coeff=ConstantParameter(...),
      timescale=ConstantParameter(...)
    )

    >>> q = u.Quantity([1.0, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity(Array(0.5, dtype=float64), unit='kpc2 / Myr2')

    """

    coeff: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="wavenumber", doc="Coefficient for the cubic terms."
    )

    timescale: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="time",
        doc="""Timescale of the potential.

    The [classical Henon-Heiles
    potential](https://en.wikipedia.org/wiki/Hénon-Heiles_system) has a
    potential with units of area. `galax` requires the potential to have units
    of specific energy, so we introduce a timescale parameter to convert the
    potential to specific energy.

    """,
    )

    @ft.partial(jax.jit)
    def _potential(
        self,
        xyz: gt.BBtQorVSz3,
        t: gt.BBtQorVSz0,
        /,
    ) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Quantity.from_(t, self.units["time"])

        # Compute parameters
        ts2 = self.timescale(t, ustrip=self.units["time"]) ** 2
        coeff = self.coeff(t, ustrip=self.units["wavenumber"])

        x2, y = xyz[..., 0] ** 2, xyz[..., 1]
        R2 = x2 + y**2
        return (R2 / 2 + coeff * (x2 * y - y**3 / 3.0)) / ts2
