"""Private module that defines `KuzminPotential` and its functions.

The public API is `galax.potential.kuzmin`.
"""

__all__ = [
    # class
    "KuzminPotential",
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
class KuzminPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Kuzmin disk potential.

    The potential of an axisymmetric razor-thin disk: all of the mass lies in
    the plane :math:`z = 0`, extends to infinite radius, and has a finite total
    mass because the surface density falls as :math:`R^{-3}`. The potential is
    symmetric under :math:`z \to -z`. The parameters are the total mass
    ``m_tot`` and the scale length ``r_s``, the radius at which the surface
    density has fallen to :math:`2^{-3/2}` of its central value. The potential
    is closed-form, and it is the :math:`c \to 0` limit of
    :class:`~galax.potential.KuzminKutuzovPotential`.

    The potential, in cylindrical coordinates :math:`(R, z)`, is

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}
                           {\sqrt{R^2 + \left(r_s + |z|\right)^2}},

    where :math:`G` is the gravitational constant. Above the disk
    (:math:`z > 0`) this is the potential of a point mass
    :math:`m_{\mathrm{tot}}` at :math:`(R, z) = (0, -r_s)`, and below the disk
    that of a point mass at :math:`(0, +r_s)` [DeZeeuwVanDeVen2011Kuzmin]_. The
    disk is due to Kuzmin [Kuzmin1953Disk]_ and was rediscovered by Toomre
    [Toomre1963Disk]_, and a textbook treatment is in [BT08Kuzmin]_.

    The jump in :math:`\partial \Phi / \partial z` across the plane gives the
    surface density, :math:`\Sigma = (2 \pi G)^{-1}\, \partial_z \Phi |_{z = 0^+}`,

    .. math::

        \Sigma(R) = \frac{m_{\mathrm{tot}}\, r_s}
                         {2 \pi \left(R^2 + r_s^2\right)^{3/2}},

    and the mass within cylindrical radius :math:`R` is

    .. math::

        M(<R) = m_{\mathrm{tot}}
                \left[ 1 - \frac{r_s}{\sqrt{R^2 + r_s^2}} \right],

    which tends to :math:`m_{\mathrm{tot}}` as :math:`R \to \infty`. In the
    plane the circular speed is given by

    .. math::

        v_c^2(R) = R \left.\frac{\partial \Phi}{\partial R}\right|_{z = 0}
                 = \frac{G m_{\mathrm{tot}} R^2}{\left(R^2 + r_s^2\right)^{3/2}},

    which peaks at :math:`R = \sqrt{2}\, r_s`, and the central potential is
    :math:`\Phi(0) = -G m_{\mathrm{tot}} / r_s`.

    Notes
    -----
    The volume density is a Dirac delta function in :math:`z` times
    :math:`\Sigma(R)`, and exactly zero off the plane (:math:`\nabla^2 \Phi = 0`
    for :math:`z \ne 0`). The function :func:`density` and the resulting
    :func:`~galax.potential.laplacian` return exact zeros, so the delta function
    on the plane is not represented. The vertical force jumps in sign across
    the plane, because :math:`\Phi` depends on :math:`|z|`.

    References
    ----------
    .. [Kuzmin1953Disk] Kuzmin, G. G. (1953). Tartu Astr. Obs. Teated, 1.
       This is the citation given for the disk by [DeZeeuwVanDeVen2011Kuzmin]_
       and by Dejonghe & de Zeeuw (1988, ApJ, 333, 90).
    .. [Toomre1963Disk] Toomre, A. (1963). On the Distribution of Matter Within
       Highly Flattened Galaxies. The Astrophysical Journal, 138, 385.
       `ADS <https://ui.adsabs.harvard.edu/abs/1963ApJ...138..385T>`__;
       `doi:10.1086/147653 <https://doi.org/10.1086/147653>`__
    .. [BT08Kuzmin] Binney, J., & Tremaine, S. (2008). Galactic Dynamics, 2nd
       ed. Princeton University Press.
    .. [DeZeeuwVanDeVen2011Kuzmin] de Zeeuw, P. T., & van de Ven, G. (2011).
       Grigori Kuzmin and Stellar Dynamics. Open Astronomy (Baltic Astronomy),
       20, 211.
       `ADS <https://ui.adsabs.harvard.edu/abs/2011BaltA..20..211D>`__;
       `doi:10.1515/astro-2017-0285 <https://doi.org/10.1515/astro-2017-0285>`__;
       `arXiv:1108.0382 <https://arxiv.org/abs/1108.0382>`__

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale length of the 'disk'."
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

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        # Compute
        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:
    r"""Potential for the Kuzmin potential.

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}
                           {\sqrt{R^2 + \left(r_s + |z|\right)^2}}

    where :math:`R^2 = x^2 + y^2`, :math:`m_{\mathrm{tot}}` is the total mass
    and :math:`r_s` is the scale length of the disk.
    """
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    _result = -p["G"] * p["m_tot"] / jnp.sqrt(R2 + (jnp.abs(z) + p["r_s"]) ** 2)
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3) -> gt.BBtSz0:  # noqa: ARG001
    r"""Density for the Kuzmin potential: exactly zero.

    Razor-thin disk: zero volume density off the :math:`z = 0` plane. All of the
    mass is in the surface density
    :math:`\Sigma(R) = m_{\mathrm{tot}} r_s / [2\pi (R^2 + r_s^2)^{3/2}]` on
    the plane, which cannot be represented as a finite volume density.
    """
    return jnp.zeros(xyz.shape[:-1], dtype=xyz.dtype)  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
