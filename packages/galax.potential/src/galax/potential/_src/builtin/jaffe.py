"""Jaffe potential."""

__all__ = [
    # class
    "JaffePotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

import functools as ft

from typing import final

import jax

import quaxed.numpy as jnp

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.kernels import (
    mass_from_dpotential,
    poisson_laplacian,
    radial_gradient,
    radial_second_derivative,
    spherical_gradient,
    spherical_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical


@final
class JaffePotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Jaffe potential.

    The model is spherical, with a central density cusp and a finite total
    mass. The density goes as :math:`r^{-2}` well inside the scale radius
    ``r_s`` and as :math:`r^{-4}` well outside it. The parameter ``m_tot`` is
    the total mass, and half of it lies inside ``r_s``. Jaffe [Jaffe1983]_
    introduced the model as a
    simple description of the distribution of light in spherical galaxies whose
    projected profile is close to the de Vaucouleurs :math:`R^{1/4}` law.

    The density is

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s}{4 \pi\, r^2 (r + r_s)^2}

    and the potential is

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r_s}
                  \ln\left(1 + \frac{r_s}{r}\right)

    where :math:`r` is the distance from the centre, :math:`m_{\mathrm{tot}}`
    is the total mass, :math:`r_s` is the scale radius and :math:`G` is the
    gravitational constant. The potential satisfies Poisson's equation
    :math:`\nabla^2 \Phi = 4 \pi G \rho`.

    The mass enclosed within radius :math:`r` is

    .. math::

        M(<r) = m_{\mathrm{tot}} \frac{r}{r + r_s}

    which tends to :math:`m_{\mathrm{tot}}` as :math:`r \to \infty` and equals
    :math:`m_{\mathrm{tot}} / 2` at the half-mass radius :math:`r = r_s`. The
    circular speed is :math:`v_c^2 = G M(<r)/r = G m_{\mathrm{tot}} / (r +
    r_s)`, and the escape speed is :math:`v_{\mathrm{esc}}^2 = -2 \Phi`.

    Notes
    -----
    The density and the potential both diverge at the origin, the potential
    logarithmically, so ``r`` must be positive. The enclosed mass and the
    circular speed stay finite. At large radius the potential approaches the
    point-mass potential :math:`-G m_{\mathrm{tot}} / r`. The inner slope of
    the density is steeper than the :math:`r^{-1}` of
    :class:`~galax.potential.HernquistPotential`, and the outer slope is the
    same.

    References
    ----------
    .. [Jaffe1983] Jaffe, W. (1983). A simple model for the distribution of
       light in spherical galaxies. Monthly Notices of the Royal Astronomical
       Society, 202, 995-999.
       `ADS <https://ui.adsabs.harvard.edu/abs/1983MNRAS.202..995J>`__;
       `doi:10.1093/mnras/202.4.995 <https://doi.org/10.1093/mnras/202.4.995>`__.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale length.")  # type: ignore[assignment]

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return potential(params, r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        r = r_spherical(xyz, self.units["length"])

        params = self.params(t)
        return density(params, r)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Potential function for the Jaffe potential.

    .. math::

        \Phi(r) = -\frac{G m_{\mathrm{tot}}}{r_s}
                  \ln\left(1 + \frac{r_s}{r}\right)

    where :math:`G` is the gravitational constant, :math:`m_{\mathrm{tot}}` is
    the total mass and :math:`r_s` is the scale radius.
    """
    _result = -p["G"] * p["m_tot"] / p["r_s"] * jnp.log(1 + p["r_s"] / r)
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:
    r"""Density function for the Jaffe potential.

    .. math::

        \rho(r) = \frac{m_{\mathrm{tot}}\, r_s}{4 \pi\, r^2 (r + r_s)^2}

    where :math:`m_{\mathrm{tot}}` is the total mass and :math:`r_s` is the
    scale radius.
    """
    _result = p["m_tot"] * p["r_s"] / (4 * jnp.pi * r**2 * (r + p["r_s"]) ** 2)
    return _result  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
dpotential_dr = radial_gradient(potential)
d2potential_dr2 = radial_second_derivative(dpotential_dr)
mass_enclosed = mass_from_dpotential(dpotential_dr)
laplacian = poisson_laplacian(density)
gradient = spherical_gradient(dpotential_dr)
hessian = spherical_hessian(dpotential_dr, d2potential_dr2, laplacian)
