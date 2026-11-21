r"""Zhao (1996) double power-law potential.

Zhao, H. 1996, MNRAS 278, 488 ("Analytical Models For Galactic Nuclei"),
https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z. Equation numbers in
this module refer to that paper.

Everything this model needs -- potential (Eqs. 6-7), enclosed mass (Eq. 15) and
density (Eq. 1) -- is closed-form in terms of the incomplete beta function
(Eq. 43, `spexial.incomplete_beta`), so *no* quantity here
is obtained by automatic differentiation: `gradient`, `hessian` and `laplacian`
are all written analytically below. That matters for speed, because the
alternative is differentiating through that function on every force evaluation.
"""

__all__ = [
    # class
    "ZhaoPotential",
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

import equinox as eqx
import jax
import jax.scipy.special as jsp

import quaxed.numpy as jnp
import unxt as u
from spexial import incomplete_beta
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.kernels import spherical_gradient
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import (
    incomplete_beta_zw,
    r_spherical,
    radial_derivatives,
    safe_vector_norm,
    shell_d2phi_dr2,
    with_r_derivative,
)

DimL = u.dimension("length")


@final
class ZhaoPotential(AbstractSinglePotential):
    r"""Zhao (1996) double power-law potential.

    A spherical model whose density is a double power law: it falls as
    :math:`r^{-\gamma}` well inside the scale radius and as :math:`r^{-\beta}`
    well outside it, with a transition between the two whose width is set by
    :math:`\alpha`. The parameters are the scale radius ``r_s``, the three
    shape parameters ``alpha``, ``beta`` and ``gamma``, and the scale mass
    ``m``, which is the mass enclosed within :math:`r_s`. The total mass is
    finite for :math:`\beta > 3`, and the mass within a finite radius is finite
    for :math:`\gamma < 3`. The family contains several common profiles, and
    [Zhao1996Potential]_ gives the potential of all of them in terms of the
    incomplete beta function.

    With :math:`u = r / r_s` the density (Eq. 1 of [Zhao1996Potential]_) is

    .. math::

        \rho(r) = \frac{\rho_s}{u^{\gamma}
                  \left(1 + u^{1/\alpha}\right)^{(\beta - \gamma)\alpha}},
        \qquad
        \rho_s = \frac{m}{4 \pi \alpha\, r_s^3\,
                 B\left(\alpha(3 - \gamma), \alpha(\beta - 3); \tfrac{1}{2}\right)},

    where :math:`B(a, b; x) = \int_0^x t^{a-1} (1 - t)^{b-1}\, dt` is the
    incomplete beta function (Eq. 43). The constant :math:`\rho_s r_s^3` is
    Zhao's :math:`C`, normalized here to the mass within :math:`r_s` instead of
    to unit total mass (Eq. 44), because the total mass is infinite for
    :math:`\beta \le 3`. With the variable

    .. math::

        \chi = \frac{u^{1/\alpha}}{1 + u^{1/\alpha}}

    (Eq. 5), which equals :math:`1/2` at :math:`r = r_s` for every
    :math:`\alpha`, the enclosed mass (Eq. 15, without a central black hole) is

    .. math::

        M(<r) = m\,
            \frac{B\left(\alpha(3 - \gamma), \alpha(\beta - 3); \chi\right)}
                 {B\left(\alpha(3 - \gamma), \alpha(\beta - 3); \tfrac{1}{2}\right)}.

    For :math:`\beta > 3` the total mass is

    .. math::

        m_{\mathrm{tot}} = 4 \pi \alpha \rho_s r_s^3\,
            \frac{\Gamma\big(\alpha(3 - \gamma)\big)\,
                  \Gamma\big(\alpha(\beta - 3)\big)}
                 {\Gamma\big(\alpha(\beta - \gamma)\big)},

    and `from_m_tot` builds the potential from it. The potential (Eqs. 6 and 7)
    is

    .. math::

        \Phi(r) = -4 \pi G\, \rho(r)\, r^2 f_{0,0}(r),

    .. math::

        f_{0,0}(r) = \frac{\alpha B(c_0 - q_0, q_0; \chi)}
                          {\chi^{c_0 - q_0} (1 - \chi)^{q_0}}
                   + \frac{\alpha B(c_0 - p_0, p_0; 1 - \chi)}
                          {(1 - \chi)^{c_0 - p_0} \chi^{p_0}},

    where :math:`G` is the gravitational constant and, from Eq. 42,
    :math:`p_0 = \alpha(2 - \gamma)`, :math:`q_0 = \alpha(\beta - 3)` and
    :math:`c_0 = \alpha(\beta - \gamma)`. The first term is :math:`G M(<r)/r`
    and the second is the contribution of the mass outside :math:`r`. The
    force follows from Newton's shell theorem,
    :math:`d\Phi/dr = G M(<r) / r^2`, and :math:`\nabla^2 \Phi = 4 \pi G \rho`.

    Several named profiles are special cases. With :math:`(\alpha, \beta,
    \gamma)` as in Table 1 of [Zhao1996Potential]_, the Hernquist model is
    :math:`(1, 4, 1)`, the Jaffe model :math:`(1, 4, 2)` and the Plummer model
    :math:`(1/2, 5, 0)`. In each of these :math:`r_s` is the scale length of the
    named model, and for example :math:`\Phi = -G m_{\mathrm{tot}} / (r + r_s)`
    for Hernquist and :math:`\Phi = -G m_{\mathrm{tot}} / \sqrt{r^2 + r_s^2}`
    for Plummer. The Navarro-Frenk-White density is :math:`(1, 3, 1)`, and
    :math:`(1, 3, \gamma)` is its generalization with inner slope
    :math:`\gamma`; both follow from Eq. 1 but are not listed in the table.

    Notes
    -----
    The parameters are not validated. The model needs :math:`\alpha > 0` and
    :math:`0 \le \gamma < 3`. For :math:`\beta \le 3` the total mass is infinite
    but ``m`` is still defined. The potential vanishes at infinity for
    :math:`\beta > 2`. For :math:`\beta \le 2` the potential cannot be
    referenced to zero at infinity, the class still returns finite values for
    :math:`\beta < 2` with the force :math:`G M(<r) / r^2`, and :math:`\beta =
    2` exactly returns NaN. The class evaluates the gradient, Hessian and
    Laplacian analytically from :math:`M(<r)` and :math:`\rho`, without
    automatic differentiation.

    References
    ----------
    .. [Zhao1996Potential] Zhao, H. (1996). Analytical models for galactic
       nuclei. Monthly Notices of the Royal Astronomical Society, 278, 488.
       `ADS <https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z>`_;
       `doi:10.1093/mnras/278.2.488 <https://doi.org/10.1093/mnras/278.2.488>`_;
       `arXiv:astro-ph/9509122 <https://arxiv.org/abs/astro-ph/9509122>`_

    """

    m: AbstractParameter = ParameterField(
        dimensions="mass",
        doc=(
            "Scale mass parameter. This is equivalent to the mass enclosed within the "
            "scale radius. When beta > 3, the model has finite mass, but when beta <= 3"
            " the total mass is infinite."
        ),
    )  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    alpha: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Transition width (alpha > 0)."
    )  # type: ignore[assignment]
    beta: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Outer slope (finite mass when beta > 3)."
    )  # type: ignore[assignment]
    gamma: AbstractParameter = ParameterField(
        dimensions="dimensionless", doc="Inner slope (0 <= gamma < 3)."
    )  # type: ignore[assignment]

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r = r_spherical(xyz, self.units["length"])
        return potential(self.params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])
        return density(self.params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        """Analytic, from the enclosed mass; see `gradient`."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        return gradient(self.params(t), xyz)

    @ft.partial(jax.jit)
    def _laplacian(self, xyz: gt.BBtQorVSz3, /, t: gt.BBtQorVSz0) -> gt.BBtFloatSz0:
        """Analytic, from Poisson's equation; see `laplacian`."""
        r = r_spherical(xyz, self.units["length"])
        return laplacian(self.params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _hessian(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz33:
        """Analytic, from the radial derivatives; see `hessian`.

        No `vectorize_method`: unlike an autodiff hessian, `hessian` below is
        already batch-native, and wrapping it in the per-point vectorize costs
        ~15%.
        """
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        return hessian(self.params(t), xyz)  # type: ignore[no-any-return]

    # ===========================================
    # Constructors

    @classmethod
    def from_m_tot(
        cls,
        m_tot: gt.Sz0 | u.Quantity["mass"],
        r_s: gt.Sz0 | u.Quantity["length"],
        alpha: gt.Sz0 | u.Quantity["dimensionless"],
        beta: gt.Sz0 | u.Quantity["dimensionless"],
        gamma: gt.Sz0 | u.Quantity["dimensionless"],
        *,
        units: u.AbstractUnitSystem | str = "galactic",
        constants: ImmutableMap[str, u.AbstractQuantity] = default_constants,
    ) -> "ZhaoPotential":
        """Create a Zhao potential from a total mass and scale radius.

        Note: This is only possible when beta > 3, when the model has finite mass.

        Parameters
        ----------
        m_tot
            Total mass of the halo.
        r_s
            Scale radius of the halo.
        alpha
            Transition width between the two power laws.
        beta
            Outer slope of the halo density profile.
        gamma
            Inner slope of the halo density profile.
        units (optional)
            Unit system to use for the potential.
        constants (optional)
            Physical constants to use for the potential.
        """
        beta = eqx.error_if(beta, beta <= 3.0, "Beta must be >3 to have finite mass.")
        usys = u.unitsystem(units)
        params = {
            "r_s": u.ustrip(usys["length"], r_s) if hasattr(r_s, "unit") else r_s,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
        return cls(
            m=m_tot * _mass_fraction_within_r_s(params),
            **params,
            units=units,
            constants=constants,
        )


# ===================================================================
# Model internals


@ft.partial(jax.jit)
def _cpq(a: gt.Sz0, b: gt.Sz0, g: gt.Sz0) -> tuple[gt.Sz0, gt.Sz0, gt.Sz0]:
    r"""Zhao's $c_0$, $p_0$, $q_0$ (Eq. 42, at $l = 0$).

    .. math::

        p_0 = \alpha(2 - \gamma), \quad
        q_0 = \alpha(\beta - 3), \quad
        c_0 = \alpha(\beta - \gamma)

    """
    c0 = a * (b - g)
    p0 = a * (2.0 - g)
    q0 = a * (b - 3.0)
    return c0, p0, q0


@ft.partial(jax.jit)
def _r_to_chi(p: gt.Params, r: gt.BBtSz0, /) -> gt.BBtFloatSz0:
    r"""Zhao's radial variable $\chi$ (Eq. 5).

    .. math::

        \chi = \frac{u^{1/\alpha}}{u^{1/\alpha} + 1}, \quad u = r / r_s

    Zhao works in units of the break radius, so $u = r / r_s$ plays the role of
    his $r$. Note that $\chi(r_s) = 1/2$ for any $\alpha$.
    """
    ua = (r / p["r_s"]) ** (1.0 / p["alpha"])
    return ua / (1.0 + ua)


@ft.partial(jax.jit)
def _r_to_1mchi(p: gt.Params, r: gt.BBtSz0, /) -> gt.BBtFloatSz0:
    r"""$1 - \chi = 1 / (u^{1/\alpha} + 1)$, without forming $1 - \chi$.

    Far out $\chi$ rounds towards 1 and keeps only some of the digits of
    $1 - \chi$; this has them all.
    """
    return 1.0 / (1.0 + (r / p["r_s"]) ** (1.0 / p["alpha"]))


@ft.partial(jax.jit)
def _mass_fraction_within_r_s(p: gt.Params, /) -> gt.FloatSz0:
    r"""$M(r_s) / M_{tot}$, i.e. Eq. 15 at $\chi = 1/2$ over Eq. 15 at $\chi = 1$.

    Only used by `ZhaoPotential.from_m_tot`, which requires $\beta > 3$, i.e.
    $q_0 > 0$: both that $M_{tot} = M(\infty)$ is finite, and that the total
    $B(a, q_0, 1)$ below is the ordinary complete beta function.
    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    a = c0 - q0
    total = jnp.exp(jsp.betaln(a, q0))  # B(a, q0, 1) = B(a, q0), for q0 > 0
    return incomplete_beta(a, q0, jnp.asarray(0.5)) / total  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def _norm(p: gt.Params, /) -> gt.FloatSz0:
    r"""Zhao's normalization constant $C$ (his Eq. 44), renormalized.

    Zhao normalizes to unit total mass (Eq. 44), which only exists for
    $\beta > 3$. The free parameter here is instead ``m``, the mass enclosed
    within the scale radius, which is well defined for every valid
    $(\alpha, \beta, \gamma)$. Inverting Eq. 15 at $r = r_s$, where
    $\chi = 1/2$ (Eq. 5) for any $\alpha$,

    .. math::

        C = \frac{m}{4\pi\alpha B(\alpha(3-\gamma), \alpha(\beta-3), 1/2)}

    The value returned carries an extra factor of $r_s^3$ relative to Zhao's
    $C$ (his radii are in units of the break radius), i.e. it is a mass rather
    than a density.
    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    bz_half = incomplete_beta(c0 - q0, q0, jnp.asarray(0.5))
    return p["m"] / (4.0 * jnp.pi * p["alpha"] * bz_half)  # type: ignore[no-any-return]


# ===================================================================
# Model functions


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile of the Zhao model (Eq. 1).

    .. math::

        \rho(r) = \frac{\rho_s}{u^\gamma (1 + u^{1/\alpha})^{(\beta-\gamma)\alpha}},
        \quad u = r / r_s

    with :math:`\rho_s = C / r_s^3` and

    .. math::

        \rho_s = \frac{m}{4\pi\alpha r_s^3\,
            B(\alpha(3-\gamma), \alpha(\beta-3); 1/2)}

    where :math:`m` is the mass within :math:`r_s` and :math:`B(a, b; x)` is
    the incomplete beta function (Eq. 43).

    """
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
    uu = r / p["r_s"]
    b = (beta - gamma) * alpha
    _result = _norm(p) / p["r_s"] ** 3 / uu**gamma / (1.0 + uu ** (1.0 / alpha)) ** b
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Mass enclosed within radius ``r`` (Eq. 15, without the black hole term).

    .. math::

        M(<r) = 4\pi\alpha C B(\alpha(3-\gamma), \alpha(\beta-3); \chi)
              = m \frac{B(\alpha(3-\gamma), \alpha(\beta-3); \chi)}
                       {B(\alpha(3-\gamma), \alpha(\beta-3); 1/2)}

    where :math:`C = \rho_s r_s^3` (see `_norm`), :math:`m` is the mass within
    :math:`r_s` and :math:`\chi` is Eq. 5.

    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    chi, chi1 = _r_to_chi(p, r), _r_to_1mchi(p, r)
    bz = incomplete_beta_zw(c0 - q0, q0, chi, chi1)
    return 4.0 * jnp.pi * p["alpha"] * _norm(p) * bz  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dmass_dr(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""$dM/dr = 4\pi r^2 \rho(r)$, without forming $r_s^3$ (see `density`)."""
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
    uu = r / p["r_s"]
    b = (beta - gamma) * alpha
    shape = uu ** (2.0 - gamma) / (1.0 + uu ** (1.0 / alpha)) ** b
    return 4.0 * jnp.pi * _norm(p) / p["r_s"] * shape  # type: ignore[no-any-return]


@ft.partial(jax.jit)
@radial_derivatives(mass_enclosed, dmass_dr)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Specific potential energy of the Zhao model (Eqs. 6 and 7).

    .. math::

        \Phi(r) = -4\pi G \rho(r) f_{0,0}(r) r^2

    .. math::

        f_{0,0}(r) = \frac{\alpha B(c_0 - q_0, q_0, \chi)}
                          {\chi^{c_0-q_0} (1-\chi)^{q_0}}
                   + \frac{\alpha B(c_0 - p_0, p_0, 1-\chi)}
                          {(1-\chi)^{c_0-p_0} \chi^{p_0}}

    with $c_0$, $p_0$, $q_0$ from Eq. 42 and $\chi$ from Eq. 5. The first term
    is the contribution of the mass interior to $r$ (it is $M(r)/r$, Eq. 15),
    the second that of the shell exterior to it.

    Written with the general incomplete beta function of Eq. 43, both terms
    stay finite across the whole valid parameter range: the first for
    $\beta \leq 3$ (where $q_0 \leq 0$), the second for $\gamma \geq 2$ (where
    $p_0 \leq 0$, e.g. the Jaffe model, $(\alpha,\beta,\gamma) = (1,4,2)$).
    """
    c0, p0, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    chi, chi1 = _r_to_chi(p, r), _r_to_1mchi(p, r)

    # Eq. 7, interior term: B(c0-q0, q0, chi) / (chi^(c0-q0) (1-chi)^q0)
    bz_in = incomplete_beta_zw(c0 - q0, q0, chi, chi1)
    interior = bz_in / (chi ** (c0 - q0) * chi1**q0)
    # Eq. 7, exterior term: B(c0-p0, p0, 1-chi) / ((1-chi)^(c0-p0) chi^p0)
    bz_ex = incomplete_beta_zw(c0 - p0, p0, chi1, chi)
    exterior = bz_ex / (chi1 ** (c0 - p0) * chi**p0)

    f00 = p["alpha"] * (interior + exterior)
    # Eq. 6, with G restored (Zhao sets G = 1).
    _result = -4.0 * jnp.pi * p["G"] * density(p, r) * f00 * r**2
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dpotential_dr(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Radial derivative of the potential.

    The model is spherical, so by Newton's shell theorem only the mass interior
    to $r$ (Eq. 15) contributes:

    .. math::

        \frac{d\Phi}{dr} = \frac{G M(r)}{r^2}

    """
    return p["G"] * mass_enclosed(p, r) / r**2  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def d2potential_dr2(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Second radial derivative of the potential.

    Differentiating `dpotential_dr` and substituting $M'(r) = 4\pi r^2 \rho(r)$
    (Eqs. 15 and 1):

    .. math::

        \frac{d^2\Phi}{dr^2} = 4\pi G \rho(r) - \frac{2 G M(r)}{r^3}
            = \frac{G M'(r)}{r^2} - \frac{2 G M(r)}{r^3}

    Evaluated in the second form, by `shell_d2phi_dr2`: `density` forms
    $r_s^3$, and that and $r^3$ overflow float32 in SI units.
    """
    return _d2phi(p, r)


_d2phi = shell_d2phi_dr2(mass_enclosed, dmass_dr)

# The jvp of `gradient` -- `jax.linearize` of it, as the Staeckel fudge takes --
# would otherwise be autodiff of `mass_enclosed`, i.e. of the incomplete beta
# function's series.
dpotential_dr = with_r_derivative(dpotential_dr, d2potential_dr2)  # type: ignore[assignment]


@ft.partial(jax.jit)
def laplacian(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Laplacian of the potential, i.e. Poisson's equation with Eq. 1.

    .. math::

        \nabla^2 \Phi = 4 \pi G \rho(r)

    """
    return 4.0 * jnp.pi * p["G"] * density(p, r)  # type: ignore[no-any-return]


# $\nabla\Phi = \Phi'(r)\,\hat{r}$, exactly zero at the origin.
gradient = spherical_gradient(dpotential_dr)


@ft.partial(jax.jit)
def hessian(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz33:
    r"""Hessian of the potential.

    For any spherical potential,

    .. math::

        \partial_i \partial_j \Phi
            = \frac{\Phi'(r)}{r} \delta_{ij}
            + \left(\Phi''(r) - \frac{\Phi'(r)}{r}\right) \frac{x_i x_j}{r^2}

    with $\Phi'(r)/r$ from `dpotential_dr` and, from `d2potential_dr2`,

    .. math::

        \Phi''(r) - \frac{\Phi'(r)}{r} = \frac{G M'(r)}{r^2} - 3 \frac{\Phi'(r)}{r}

    Only $\Phi'/r$ reads $M(r)$. Had both coefficients read it, XLA on CPU would
    fuse a copy of the incomplete beta series into each and evaluate it twice
    per point (an `optimization_barrier` does not stop it: the CPU backend
    removes barriers before fusion). Each division by $r$ is kept separate, as
    in `shell_d2phi_dr2`, for float32 in SI units.

    At the origin the Hessian is $\nabla^2\Phi(0)/3\,\delta_{ij}$ (`laplacian`
    at $r = 0$), which is $+\infty$ on the diagonal for a cusp.
    """
    origin = jnp.sum(jnp.square(xyz), axis=-1, keepdims=True) == 0
    r1 = jnp.where(origin, 1.0, safe_vector_norm(xyz)[..., None])
    # `dpotential_dr`, not `mass_enclosed` directly: its custom jvp keeps a
    # derivative of this Hessian off autodiff of the series.
    # At the origin the Hessian is isotropic with trace `laplacian(p, 0)`
    # (as in `spherical_hessian`); the outer term is dropped, and the inner
    # `where` keeps every radius off 0 so that no inf * 0 = NaN arises.
    r0 = jnp.where(origin, 0.0, r1)
    # `* (1 / r1)`: `/ r1` would fuse with the `/ r**2` in `dpotential_dr` into
    # an r**3 that overflows float32 in SI (as in `spherical_hessian`).
    radial = jnp.where(origin, laplacian(p, r0) / 3, dpotential_dr(p, r1) * (1 / r1))
    coef_outer = p["G"] * dmass_dr(p, r1) / r1 / r1 - 3 * radial
    coef_outer = jnp.where(origin, 0.0, coef_outer)

    rhat = xyz / r1
    outer = rhat[..., :, None] * rhat[..., None, :]
    # `where`, not `radial * eye`: at a cusp's origin `radial` is inf, and inf * 0
    # is NaN off the diagonal when nothing rewrites the product (`disable_jit`).
    eye = jnp.eye(3, dtype=bool)
    _result = jnp.where(eye, radial[..., None], 0.0) + coef_outer[..., None] * outer
    return _result  # type: ignore[no-any-return]
