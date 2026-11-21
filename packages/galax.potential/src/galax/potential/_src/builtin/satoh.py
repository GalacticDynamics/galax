"""galax: Galactic Dynamix in Jax."""

__all__ = [
    # class
    "SatohPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
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
class SatohPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Satoh potential.

    The Satoh model is a flattened, axisymmetric mass distribution of finite
    total mass, symmetric about the plane :math:`z = 0`. It is the limiting
    member of Satoh's series of flattened higher-order Plummer models
    [Satoh1980]_. The potential tends to the Kuzmin disk for :math:`b \to 0`
    and to the point-mass potential for :math:`a \to 0`. The parameter ``m_tot``
    is the total mass, ``a`` the scale length and ``b`` the scale height. The
    density is finite and positive everywhere. Satoh proposed the model for
    axisymmetric galaxies from spirals to ellipticals, and used it to model
    NGC 4697.

    The density is

    .. math::

        \rho(R, z) = \frac{a b^2 m_{\mathrm{tot}}}{4\pi}
            \frac{R^2 + z^2 + (a + 3\zeta)(a + 2\zeta)}
                 {\left[R^2 + z^2 + a(a + 2\zeta)\right]^{5/2} \zeta^3},

    and the potential is

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}
                          {\sqrt{R^2 + z^2 + a(a + 2\zeta)}},

    where :math:`R^2 = x^2 + y^2`, :math:`\zeta = \sqrt{z^2 + b^2}` and :math:`G`
    is the gravitational constant. The two satisfy Poisson's equation,
    :math:`\nabla^2 \Phi = 4 \pi G \rho`, and the total mass is
    :math:`m_{\mathrm{tot}}`. The density at the centre is

    .. math::

        \rho(0, 0) = \frac{m_{\mathrm{tot}} (a + 3b)}
                          {4 \pi b \, a^{3/2} (a + 2b)^{3/2}}.

    In the plane :math:`z = 0` the circular velocity is

    .. math::

        v_c^2(R) = \frac{G m_{\mathrm{tot}} R^2}
                        {\left[R^2 + a(a + 2b)\right]^{3/2}},

    which is the circular velocity of a Plummer sphere of scale length
    :math:`\sqrt{a(a + 2b)}`.

    Notes
    -----
    Far from the centre the potential tends to :math:`-G m_{\mathrm{tot}} / r`.
    For :math:`a > 0` the density falls as :math:`R^{-3}` in the plane. At
    :math:`b = 0` the potential is
    :math:`-G m_{\mathrm{tot}} / \sqrt{R^2 + (a + |z|)^2}`, the Kuzmin disk of scale
    length ``a`` (see :class:`~galax.potential.KuzminPotential`). The
    Miyamoto-Nagai potential (:class:`~galax.potential.MiyamotoNagaiPotential`)
    is the zeroth member of the same series of models, with the same two
    parameters and a different vertical structure: its density has an
    equatorial brim around a central sphere, which the Satoh model reduces
    [Satoh1980]_. The gradient and Hessian of this class come from automatic
    differentiation of the potential.

    References
    ----------
    .. [Satoh1980] Satoh, C. (1980). Dynamical Models of Axisymmetric Galaxies
       and Their Application to the Elliptical Galaxy NGC 4697. Publications of
       the Astronomical Society of Japan, 32, 41-62.
       `ADS <https://ui.adsabs.harvard.edu/abs/1980PASJ...32...41S>`__.
       The potential and density are the pair of his eq. 8.

    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass", doc="Total mass.")  # type: ignore[assignment]

    a: AbstractParameter = ParameterField(dimensions="length", doc="Scale length")  # type: ignore[assignment]

    b: AbstractParameter = ParameterField(dimensions="length", doc="Scale height.")  # type: ignore[assignment]

    symmetry = Symmetry.ZROTATION_ZREFLECTION
    """`Symmetry.ZROTATION_ZREFLECTION`: the density depends on $R$ and $|z|$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)

        params = self.params(t)
        return density(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Specific potential energy of the Satoh model (Satoh 1980, eq. 8).

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}
                          {\sqrt{R^2 + z^2 + a(a + 2\zeta)}},
        \qquad \zeta = \sqrt{z^2 + b^2},

    with :math:`R^2 = x^2 + y^2`.

    """
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    term = R2 + z**2 + p["a"] * (p["a"] + 2 * jnp.sqrt(z**2 + p["b"] ** 2))
    return -p["G"] * p["m_tot"] / jnp.sqrt(term)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz0:
    r"""Density function for the Satoh potential (Satoh 1980, eq. 8).

    .. math::

        \rho(R, z) = \frac{a b^2 m_{\mathrm{tot}}}{4\pi}
            \frac{R^2 + z^2 + (a + 3\zeta)(a + 2\zeta)}
                 {\left[R^2 + z^2 + a(a + 2\zeta)\right]^{5/2} \zeta^3},
        \qquad \zeta = \sqrt{z^2 + b^2},

    which is the same as the expanded form
    :math:`\zeta (R^2 + a^2 + 6 b^2 + 7 z^2) + 5 a \zeta^2` over :math:`\zeta^4`
    used in the code.

    """
    a, b, m_tot = p["a"], p["b"], p["m_tot"]
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = xyz[..., 2]
    zeta = jnp.sqrt(z**2 + b**2)
    term = R2 + z**2 + a * (a + 2 * zeta)
    numer = (
        m_tot * a * b**2 * (zeta * (R2 + a**2 + 6 * b**2 + 7 * z**2) + 5 * a * zeta**2)
    )
    denom = 4 * jnp.pi * zeta**4 * term**2.5
    return numer / denom  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
