"""Private module that defines `MiyamotoNagaiPotential` and its functions.

The public API is `galax.potential.miyamotonagai`.
"""

__all__ = [
    # class
    "MiyamotoNagaiPotential",
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
from galax.potential._src.symmetry import Symmetry


@final
class MiyamotoNagaiPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Miyamoto-Nagai potential.

    The Miyamoto-Nagai model is a flattened, axisymmetric mass distribution of
    finite total mass, symmetric about the plane :math:`z = 0`. It generalizes
    the Kuzmin disk: the potential reduces to the Kuzmin disk for :math:`b = 0`
    and to the Plummer sphere for :math:`a = 0`, so one family covers shapes from
    a razor-thin disk to a sphere [MiyamotoNagai1975]_. The parameter ``m_tot``
    is the total mass, ``a`` the radial scale length and ``b`` the vertical
    scale height. The density is finite and positive everywhere.

    The density is

    .. math::

        \rho(R, z) = \frac{b^2 m_{\mathrm{tot}}}{4\pi}
            \frac{a R^2 + (a + 3\zeta)(a + \zeta)^2}
                 {\left[R^2 + (a + \zeta)^2\right]^{5/2} \zeta^3},

    and the potential is

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{R^2 + (a + \zeta)^2}},

    where :math:`R^2 = x^2 + y^2`, :math:`\zeta = \sqrt{z^2 + b^2}` and :math:`G`
    is the gravitational constant. The two satisfy Poisson's equation,
    :math:`\nabla^2 \Phi = 4 \pi G \rho`, and the total mass is
    :math:`m_{\mathrm{tot}}`. The density at the centre is

    .. math::

        \rho(0, 0) = \frac{m_{\mathrm{tot}} (a + 3b)}{4 \pi b (a + b)^3}.

    In the plane :math:`z = 0` the circular velocity is

    .. math::

        v_c^2(R) = \frac{G m_{\mathrm{tot}} R^2}{\left[R^2 + (a + b)^2\right]^{3/2}},

    which depends on ``a`` and ``b`` only through :math:`a + b`.

    Notes
    -----
    Far from the centre the potential tends to :math:`-G m_{\mathrm{tot}} / r`.
    For :math:`a > 0` the density falls as :math:`R^{-3}` in the plane, and as
    :math:`|z|^{-5}` on the symmetry axis. The :math:`a \to 0` limit is the
    Plummer sphere of scale length ``b`` (see
    :class:`~galax.potential.PlummerPotential`) and the :math:`b \to 0` limit is
    the Kuzmin disk of scale length ``a`` (see
    :class:`~galax.potential.KuzminPotential`). The Satoh potential
    (:class:`~galax.potential.SatohPotential`) is a related flattened family with
    the same two parameters and a different vertical structure. The gradient and
    Hessian of this class come from automatic differentiation of the potential.

    References
    ----------
    .. [MiyamotoNagai1975] Miyamoto, M., & Nagai, R. (1975). Three-Dimensional
       Models for the Distribution of Mass in Galaxies. Publications of the
       Astronomical Society of Japan, 27, 533-543.
       `ADS <https://ui.adsabs.harvard.edu/abs/1975PASJ...27..533M>`__.
       The potential is their eq. 4 and the density their eq. 5.

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    # TODO: rename
    a: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length in the major-axis (x-y) plane."
    )

    b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length in the minor-axis (x-y) plane."
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.ZROTATION_ZREFLECTION
    """`Symmetry.ZROTATION_ZREFLECTION`: the density depends on $R$ and $|z|$."""

    @ft.partial(jax.jit, inline=True)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)
        return potential(p, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute parameters
        p = self.params(t)
        return density(p, xyz)  # type: ignore[no-any-return]


# ===================================================================
# Functions


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3) -> gt.Sz0:
    r"""Miyamoto-Nagai potential function.

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{R^2 + (a + \zeta)^2}},
        \qquad \zeta = \sqrt{z^2 + b^2},

    with :math:`R^2 = x^2 + y^2` (Miyamoto & Nagai 1975, eq. 4).

    """
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    zp2 = (jnp.sqrt(xyz[..., 2] ** 2 + p["b"] ** 2) + p["a"]) ** 2
    return -p["G"] * p["m_tot"] / jnp.sqrt(R2 + zp2)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3) -> gt.Sz0:
    r"""Miyamoto-Nagai density function (Miyamoto & Nagai 1975, eq. 5).

    .. math::

        \rho(R, z) = \frac{b^2 m_{\mathrm{tot}}}{4\pi}
            \frac{a R^2 + (a + 3\zeta)(a + \zeta)^2}
                 {\left[R^2 + (a + \zeta)^2\right]^{5/2} \zeta^3},
        \qquad \zeta = \sqrt{z^2 + b^2}.

    """
    a, b, m_tot = p["a"], p["b"], p["m_tot"]
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    zeta = jnp.sqrt(xyz[..., 2] ** 2 + b**2)
    numer = a * R2 + (a + 3 * zeta) * (a + zeta) ** 2
    denom = (R2 + (a + zeta) ** 2) ** 2.5 * zeta**3
    return b**2 * m_tot / (4 * jnp.pi) * numer / denom  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
