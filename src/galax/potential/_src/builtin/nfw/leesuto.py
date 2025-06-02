"""galax: Galactic Dynamix in Jax."""

__all__ = ["LeeSutoTriaxialNFWPotential"]

import functools as ft
from dataclasses import KW_ONLY
from typing import Final, final

import equinox as eqx
import jax

import quaxed.lax as qlax
import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

LOG2: Final = jnp.log(jnp.asarray(2.0))


@final
class LeeSutoTriaxialNFWPotential(AbstractSinglePotential):
    """Approximate triaxial (in the density) NFW potential.

    Approximation of a Triaxial NFW Potential with the flattening in the
    density, not the potential. See Lee & Suto (2003) for details.

    .. warning::

        This potential is only physical for `a1 >= a2 >= a3`.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.LeeSutoTriaxialNFWPotential(
    ...     m=1e11, r_s=15, a1=1, a2=0.9, a3=0.8, units="galactic")

    >>> q = u.Quantity([1, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity(Array(-0.14620419, dtype=float64), unit='kpc2 / Myr2')

    >>> q = u.Quantity([0, 1, 0], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity(Array(-0.14593972, dtype=float64), unit='kpc2 / Myr2')

    >>> q = u.Quantity([0, 0, 1], "kpc")
    >>> pot.potential(q, t).decompose(pot.units)
    Quantity(Array(-0.14570309, dtype=float64), unit='kpc2 / Myr2')
    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc=r"""Scale mass.

    This is the mass corresponding to the circular velocity at the scale radius.
    :math:`v_c = \sqrt{G M / r_s}`
    """,
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    a1: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", default=u.Quantity(1.0, ""), doc="Major axis."
    )

    a2: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        default=u.Quantity(1.0, ""),
        doc="Intermediate axis.",
    )

    a3: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless", default=u.Quantity(1.0, ""), doc="Minor axis."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __check_init__(self) -> None:
        t = u.Quantity(0.0, "Myr")
        _ = eqx.error_if(
            t,
            jnp.logical_or(self.a1(t) < self.a2(t), self.a2(t) < self.a3(t)),
            f"a1 {self.a1(t)} >= a2 {self.a2(t)} >= a3 {self.a3(t)} is required",
        )

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /
    ) -> gt.BBtSz0:
        t = u.Quantity.from_(t, self.units["time"])
        ul = self.units["dimensionless"]
        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "a1": self.a1(t, ustrip=ul),
            "a2": self.a2(t, ustrip=ul),
            "a3": self.a3(t, ustrip=ul),
        }
        return potential(params, xyz)


# =============================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BtFloatSz0:
    r"""Specific potential energy."""
    # Compute parameters
    # https://github.com/adrn/gala/blob/2067009de41518a71c674d0252bc74a7b2d78a36/gala/potential/potential/builtin/builtin_potentials.c#L1472

    # Parse inputs
    r = jnp.linalg.vector_norm(xyz, axis=-1)

    v_c2 = p["G"] * p["m"] / p["r_s"]

    # 1- eccentricities
    e_b2 = 1 - jnp.square(p["a2"] / p["a1"])
    e_c2 = 1 - jnp.square(p["a3"] / p["a1"])

    # The potential at the origin
    phi0 = v_c2 / (LOG2 - 0.5 + (LOG2 - 0.75) * (e_b2 + e_c2))

    # The potential at the given position
    s = r / p["r_s"]

    # The functions F1, F2, and F3 and some useful quantities
    log1pu = jnp.log(1 + s)
    u2 = s**2
    um3 = s ** (-3)
    costh2 = xyz[..., 2] ** 2 / r**2  # z^2 / r^2
    sinthsinphi2 = xyz[..., 1] ** 2 / r**2  # (sin(theta) * sin(phi))^2
    # Note that ꜛ is safer than computing the separate pieces, as it avoids
    # x=y=0, z!=0, which would result in a NaN.

    F1 = -log1pu / s
    F2 = -1.0 / 3 + (2 * u2 - 3 * s + 6) / (6 * u2) + (1 / s - um3) * log1pu
    F3 = (u2 - 3 * s - 6) / (2 * u2 * (1 + s)) + 3 * um3 * log1pu

    # Select the output, r=0 is a special case.
    out: gt.BtFloatSz0 = phi0 * qlax.select(
        s == 0,
        jnp.ones_like(s),
        (F1 + (e_b2 + e_c2) / 2 * F2 + (e_b2 * sinthsinphi2 + e_c2 * costh2) / 2 * F3),
    )
    return out
