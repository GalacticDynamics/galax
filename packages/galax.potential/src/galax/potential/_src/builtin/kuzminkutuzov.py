"""Kuzmin-Kutuzov Staeckel potential."""

__all__ = [
    # class
    "KuzminKutuzovPotential",
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
class KuzminKutuzovPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Kuzmin-Kutuzov Staeckel potential.

    An axisymmetric potential, symmetric under :math:`z \to -z`, that is of
    Staeckel form in prolate spheroidal coordinates, so that every orbit in it
    has three exact integrals of motion. The mass distribution is stratified
    approximately on spheroids, oblate for :math:`c < a` and prolate for
    :math:`c > a`, with a density that falls as :math:`r^{-4}` at large radii
    and a finite total mass. The spherical limit :math:`c = a` is Henon's
    isochrone. The parameters are the total mass ``m_tot`` and the two scale
    lengths ``a`` (in the plane) and ``c`` (along :math:`z`). The model was
    introduced by [Kuzmin1956KK]_ and studied by [KuzminKutuzov1962]_ and
    [DejongheDeZeeuw1988]_, and the potential is also given by
    [BatsleerDejonghe1994]_. This is galpy's ``KuzminKutuzovStaeckelPotential``
    with ``ac = a / c`` and ``Delta = sqrt(a^2 - c^2)``.

    The potential is

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{\lambda} + \sqrt{\nu}},

    where :math:`G` is the gravitational constant and
    :math:`c^2 \le \nu \le a^2 \le \lambda` are the roots in :math:`\tau` of
    :math:`R^2 / (\tau - a^2) + z^2 / (\tau - c^2) = 1`: the prolate spheroidal
    coordinates with focal distance :math:`\Delta = \sqrt{a^2 - c^2}`.

    Because :math:`\Phi` is symmetric in :math:`\lambda` and :math:`\nu`, it
    depends only on :math:`\lambda + \nu = R^2 + z^2 + a^2 + c^2` and
    :math:`\lambda\nu = c^2 R^2 + a^2 z^2 + a^2 c^2`, which gives the form
    evaluated here:

    .. math::

        \Phi(R, z) = -\frac{G m_{\mathrm{tot}}}{\sqrt{R^2 + z^2 + a^2 + c^2
            + 2\sqrt{c^2 R^2 + a^2 z^2 + a^2 c^2}}}

    The density (Eq. 4.3 of [DejongheDeZeeuw1988]_) has the same property:

    .. math::

        \rho = \frac{m_{\mathrm{tot}}\, c^2}{4\pi} \,
            \frac{\lambda\nu + a^2 (\lambda + \nu + 3\sqrt{\lambda\nu})}
                 {(\lambda\nu)^{3/2} (\sqrt{\lambda} + \sqrt{\nu})^3}

    The central values are :math:`\Phi(0) = -G m_{\mathrm{tot}} / (a + c)` and
    (Eq. 4.5 of [DejongheDeZeeuw1988]_)

    .. math::

        \rho(0) = \frac{m_{\mathrm{tot}} (a + 2c)}{4 \pi a c (a + c)^2},

    and in the plane :math:`z = 0` the potential reduces to
    :math:`\Phi(R, 0) = -G m_{\mathrm{tot}} / (c + \sqrt{R^2 + a^2})`. At large
    radii :math:`\Phi \to -G m_{\mathrm{tot}} / r`, so ``m_tot`` is the total
    mass.

    Notes
    -----
    The quadratic for :math:`\lambda, \nu` is never solved, and every term is
    positive (only square roots of positive quantities are taken), so both are
    smooth and free of cancellation everywhere for :math:`a, c > 0`,
    including the origin, the plane, the axis and the focal points
    :math:`(R, z) = (0, \pm\Delta)`. :func:`~galax.potential.gradient` and
    :func:`~galax.potential.hessian` (by automatic differentiation) are
    finite there too. The closed-form density matters for flat models: off
    the plane :math:`\rho \propto c^2`, so a Laplacian of :math:`\Phi`
    would lose :math:`\sim (a / c)^2` to cancellation.

    The parameters are not validated (they may depend on time):

    - :math:`0 < c < a` is the oblate Kuzmin-Kutuzov model above.
    - :math:`c = a`: the focal points meet at the origin and this is exactly
      the isochrone, :math:`-G m_{\mathrm{tot}} / (a + \sqrt{r^2 + a^2})`.
    - :math:`c > a` is a valid prolate model: :math:`\Phi` and :math:`\rho`
      are finite and :math:`\rho > 0`. But :math:`\Delta^2 = a^2 - c^2 < 0`,
      so it is not of Staeckel form in prolate spheroidal coordinates, and a
      Staeckel-fudge focal distance does not apply.
    - :math:`c \to 0` tends to the razor-thin `~galax.potential.KuzminPotential`
      with ``r_s = a``. At :math:`c = 0` exactly :math:`\Phi` is that disk,
      but the gradient is NaN on the plane :math:`z = 0`; use
      `~galax.potential.KuzminPotential` instead.

    References
    ----------
    .. [Kuzmin1956KK] Kuzmin, G. G. (1956). Astronomicheskii Zhurnal, 33, 27.
    .. [KuzminKutuzov1962] Kuzmin, G. G., & Kutuzov, S. A. (1962). Bulletin of
       the Abastumani Astrophysical Observatory, 27, 82.
    .. [DejongheDeZeeuw1988] Dejonghe, H., & de Zeeuw, T. (1988). Analytic
       Axisymmetric Galaxy Models with Three Integrals of Motion. The
       Astrophysical Journal, 333, 90.
       `ADS <https://ui.adsabs.harvard.edu/abs/1988ApJ...333...90D>`__;
       `doi:10.1086/166727 <https://doi.org/10.1086/166727>`_
    .. [BatsleerDejonghe1994] Batsleer, P., & Dejonghe, H. (1994). The
       generation of Staeckel potentials for galactic dynamical modeling.
       Astronomy & Astrophysics, 287, 43.
       `ADS <https://ui.adsabs.harvard.edu/abs/1994A&A...287...43B>`__

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KuzminKutuzovPotential(m_tot=1e11, a=5, c=1, units="galactic")
    >>> pot.potential(u.Q([0.0, 0.0, 0.0], "kpc"), t=0)
    Q(-0.07497504, 'kpc2 / Myr2')

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass of the potential."
    )

    a: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc=r"""Scale length in the plane; de Zeeuw's :math:`\alpha = -a^2`.

    :math:`a / c = \sqrt{\alpha / \gamma}` is galpy's ``ac``. The inner
    isodensity surfaces have axis ratio roughly, not exactly, :math:`c / a`.
    """,
    )

    c: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length",
        doc=r"""Scale length along :math:`z`; de Zeeuw's :math:`\gamma = -c^2`.

    :math:`c < a` is oblate, :math:`c = a` the isochrone, :math:`c > a`
    prolate. Use `~galax.potential.KuzminPotential` for :math:`c = 0`.
    """,
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.ZROTATION_ZREFLECTION
    """`Symmetry.ZROTATION_ZREFLECTION`: the density depends on $(R, |z|)$."""

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
    r"""Kuzmin-Kutuzov potential function.

    .. math::

        \Phi = -\frac{G m_{\mathrm{tot}}}{\sqrt{\lambda + \nu
                + 2\sqrt{\lambda\nu}}},
        \quad \lambda + \nu = R^2 + z^2 + a^2 + c^2,
        \quad \lambda\nu = c^2 R^2 + a^2 z^2 + a^2 c^2

    where :math:`\sqrt{\lambda\nu}` is the geometric mean of the two prolate
    spheroidal coordinates, so that
    :math:`\sqrt{\lambda} + \sqrt{\nu} = \sqrt{\lambda + \nu
    + 2\sqrt{\lambda\nu}}`.

    """
    a2, c2 = p["a"] ** 2, p["c"] ** 2
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z2 = xyz[..., 2] ** 2
    lam_plus_nu = R2 + z2 + a2 + c2
    lam_nu = c2 * R2 + a2 * z2 + a2 * c2  # >= a^2 c^2 > 0: sqrt is smooth
    denom = jnp.sqrt(lam_plus_nu + 2 * jnp.sqrt(lam_nu))
    return -p["G"] * p["m_tot"] / denom  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3) -> gt.Sz0:
    r"""Kuzmin-Kutuzov density function (Dejonghe & de Zeeuw 1988, Eq. 4.3).

    .. math::

        \rho = \frac{m_{\mathrm{tot}}\, c^2}{4\pi} \,
            \frac{\lambda\nu + a^2 (\lambda + \nu + 3\sqrt{\lambda\nu})}
                 {(\lambda\nu)^{3/2} (\sqrt{\lambda} + \sqrt{\nu})^3}

    with :math:`\lambda + \nu` and :math:`\lambda\nu` as in `potential`.

    """
    a2, c2 = p["a"] ** 2, p["c"] ** 2
    R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z2 = xyz[..., 2] ** 2
    lam_plus_nu = R2 + z2 + a2 + c2
    lam_nu = c2 * R2 + a2 * z2 + a2 * c2
    sqrt_lam_nu = jnp.sqrt(lam_nu)
    sum_sqrt = jnp.sqrt(lam_plus_nu + 2 * sqrt_lam_nu)  # sqrt(lam) + sqrt(nu)
    numer = lam_nu + a2 * (lam_plus_nu + 3 * sqrt_lam_nu)
    denom = lam_nu * sqrt_lam_nu * sum_sqrt**3
    return p["m_tot"] * c2 / (4 * jnp.pi) * numer / denom  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
