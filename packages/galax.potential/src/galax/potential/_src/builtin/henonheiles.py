"""Private module that defines `HenonHeilesPotential` and its functions.

The public API is `galax.potential.henonheiles`.
"""

__all__ = [
    # class
    "HenonHeilesPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
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

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class HenonHeilesPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Henon-Heiles potential.

    This is a modified version of the [classical Henon-Heiles
    potential](https://en.wikipedia.org/wiki/Hénon-Heiles_system).

    .. math::

        \Phi\, t_s^2 = \frac{1}{2} (x^2 + y^2) + \lambda (x^2 y - y^3 / 3)

    where :math:`\lambda` is ``coeff`` and :math:`t_s` is ``timescale``. The
    timescale is introduced to convert the potential to specific energy, from
    the classical area units. The potential does not depend on :math:`z`.

    The cubic terms are harmonic, so the Laplacian is constant,
    :math:`\nabla^2 \Phi = 2 / t_s^2`, and so is the density,

    .. math::

        \rho = \frac{1}{2 \pi G\, t_s^2}.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.HenonHeilesPotential(coeff=u.Q(1, "1 / kpc"),
    ...                               timescale=u.Q(1, "Myr"),
    ...                               units="galactic")
    >>> pot
    HenonHeilesPotential(
      units=LTMAUnitSystem( ... ),
      constants=ImmutableMap({'G': ...}),
      coeff=ConstantParameter(...),
      timescale=ConstantParameter(...)
    )

    >>> q = u.Q([1.0, 0, 0], "kpc")
    >>> t = u.Q(0, "Gyr")
    >>> pot.potential(q, t)
    Q(0.5, 'kpc2 / Myr2')

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

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz0:
    r"""Potential of the Henon-Heiles model.

    .. math::

        \Phi = \frac{1}{t_s^2} \left[ \frac{1}{2} (x^2 + y^2)
               + \lambda (x^2 y - y^3 / 3) \right]

    where :math:`\lambda` is ``p["coeff"]`` and :math:`t_s` is
    ``p["timescale"]``.
    """
    x2, y = xyz[..., 0] ** 2, xyz[..., 1]
    R2 = x2 + y**2
    _result = (R2 / 2 + p["coeff"] * (x2 * y - y**3 / 3.0)) / p["timescale"] ** 2
    return _result


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Density of the Henon-Heiles model, uniform in space.

    .. math::

        \rho = \frac{1}{2 \pi G\, t_s^2}

    which is :math:`\nabla^2 \Phi / (4 \pi G)` with :math:`\nabla^2 \Phi =
    2 / t_s^2`.
    """
    _result = 2 / (4 * jnp.pi * p["G"] * p["timescale"] ** 2)
    return jnp.broadcast_to(_result, xyz.shape[:-1])  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
