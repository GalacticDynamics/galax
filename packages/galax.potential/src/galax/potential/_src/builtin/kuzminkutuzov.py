"""Kuzmin-Kutuzov Staeckel potential."""

__all__ = (
    # class
    "KuzminKutuzovPotential",
    # functions
    "potential",
    "density",
)

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
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry


@final
class KuzminKutuzovPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Kuzmin-Kutuzov Staeckel potential.

    Kuzmin, G. G., & Kutuzov, S. A. 1962, Bull. Abastumani Astrophys. Obs.,
    27, 82. See also Dejonghe, H., & de Zeeuw, T. 1988, ApJ, 333, 90, and
    Batsleer, P., & Dejonghe, H. 1994, A&A, 287, 43. This is galpy's
    ``KuzminKutuzovStaeckelPotential`` with ``ac = a / c`` and
    ``Delta = sqrt(a^2 - c^2)``.

    .. math::

        \Phi(R, z) = -\frac{G M}{\sqrt{\lambda} + \sqrt{\nu}}

    where :math:`c^2 \le \nu \le a^2 \le \lambda` are the roots in
    :math:`\tau` of :math:`R^2 / (\tau - a^2) + z^2 / (\tau - c^2) = 1`: the
    prolate spheroidal coordinates with focal distance
    :math:`\Delta = \sqrt{a^2 - c^2}`. The potential is exactly of Staeckel
    form in these coordinates, so orbits in it have three exact integrals.

    Because :math:`\Phi` is symmetric in :math:`\lambda` and :math:`\nu`, it
    depends only on :math:`\lambda + \nu = R^2 + z^2 + a^2 + c^2` and
    :math:`\lambda\nu = c^2 R^2 + a^2 z^2 + a^2 c^2`, which gives the form
    evaluated here:

    .. math::

        \Phi(R, z) = -\frac{G M}{\sqrt{R^2 + z^2 + a^2 + c^2
            + 2\sqrt{c^2 R^2 + a^2 z^2 + a^2 c^2}}}

    The density (Dejonghe & de Zeeuw 1988) has the same property:

    .. math::

        \rho = \frac{M c^2}{4\pi} \,
            \frac{\lambda\nu + a^2 (\lambda + \nu + 3\sqrt{\lambda\nu})}
                 {(\lambda\nu)^{3/2} (\sqrt{\lambda} + \sqrt{\nu})^3}

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
      the isochrone, :math:`-G M / (a + \sqrt{r^2 + a^2})`.
    - :math:`c > a` is a valid prolate model: :math:`\Phi` and :math:`\rho`
      are finite and :math:`\rho > 0`. But :math:`\Delta^2 = a^2 - c^2 < 0`,
      so it is not of Staeckel form in prolate spheroidal coordinates, and a
      Staeckel-fudge focal distance does not apply.
    - :math:`c \to 0` tends to the razor-thin `~galax.potential.KuzminPotential`
      with ``r_s = a``. At :math:`c = 0` exactly :math:`\Phi` is that disk,
      but the gradient is NaN on the plane :math:`z = 0`; use
      `~galax.potential.KuzminPotential` instead.

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
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        ul = self.units["length"]
        p = {
            "G": self.constants["G"].value,
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "c": self.c(t, ustrip=ul),
        }
        return potential(p, xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        ul = self.units["length"]
        p = {
            "m_tot": self.m_tot(t, ustrip=self.units["mass"]),
            "a": self.a(t, ustrip=ul),
            "c": self.c(t, ustrip=ul),
        }
        return density(p, xyz)  # type: ignore[no-any-return]


# ===================================================================
# Functions


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3) -> gt.Sz0:
    r"""Kuzmin-Kutuzov potential function.

    $$
    \Phi = -\frac{G M}{\sqrt{\lambda + \nu + 2\sqrt{\lambda\nu}}},
    \quad \lambda + \nu = R^2 + z^2 + a^2 + c^2,
    \quad \lambda\nu = c^2 R^2 + a^2 z^2 + a^2 c^2
    $$

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
    r"""Kuzmin-Kutuzov density function (Dejonghe & de Zeeuw 1988).

    $$
    \rho = \frac{M c^2}{4\pi} \,
        \frac{\lambda\nu + a^2 (\lambda + \nu + 3\sqrt{\lambda\nu})}
             {(\lambda\nu)^{3/2} (\sqrt{\lambda} + \sqrt{\nu})^3}
    $$

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
