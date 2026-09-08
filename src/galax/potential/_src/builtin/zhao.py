r"""Zhao (1996) double power-law potential.

Zhao, H. 1996, MNRAS 278, 488 ("Analytical Models For Galactic Nuclei"),
https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z. Equation numbers in
this module refer to that paper.

Everything this model needs -- potential (Eqs. 6-7), enclosed mass (Eq. 15) and
density (Eq. 1) -- is closed-form in terms of the incomplete beta function
(Eq. 43, `galax.potential._src.special.incomplete_beta`), so *no* quantity here
is obtained by automatic differentiation: `gradient`, `hessian` and `laplacian`
are all written analytically below. That matters for speed, because the
alternative is differentiating through that function on every force evaluation.
"""

__all__ = ["ZhaoPotential"]

import functools as ft

from typing import final

import equinox as eqx
import jax
import jax.scipy.special as jsp

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.special import incomplete_beta
from galax.potential._src.utils import r_spherical

DimL = u.dimension("length")


@final
class ZhaoPotential(AbstractSinglePotential):
    r"""Zhao (1996) double power-law potential.

    This model represents a double power law in the density, with an inner slope
    :math:`\gamma` and an outer slope :math:`\beta`, but with a third parameter
    :math:`\alpha` that controls the width of the transition region between the two
    power laws (Zhao 1996, Eq. 1):

    .. math::

        \rho(r) = \frac{C}{u^\gamma (1 + u^{1/\alpha})^{(\beta - \gamma)\alpha}},
        \quad u = r / r_s

    This model has a finite total mass for :math:`\beta > 3`. The other power-law
    parameters should satisfy :math:`\alpha > 0` and :math:`0 \leq \gamma < 3`.

    This model also reduces to a number of well-known analytic forms for certain values
    of the parameters (reproduced from Table 1 of Zhao 1996):
    - :math:`(\alpha, \beta, \gamma) = (1, 4, 1)`: Hernquist model
    - :math:`(\alpha, \beta, \gamma) = (1, 4, 2)`: Jaffe model
    - :math:`(\alpha, \beta, \gamma) = (1/2, 5, 0)`: Plummer model
    - :math:`(\alpha, \beta, \gamma) = (1, 3, 1)`: NFW model
    - :math:`(\alpha, \beta, \gamma) = (1, 3, \gamma)`: Generalized NFW model
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

    def _params(self, t: gt.BBtQorVSz0, /) -> gt.Params:
        """Evaluate the parameters at ``t``, stripped to this unit system."""
        t = u.Q.from_(t, self.units["time"])
        udim = self.units["dimensionless"]
        return {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "alpha": self.alpha(t, ustrip=udim),
            "beta": self.beta(t, ustrip=udim),
            "gamma": self.gamma(t, ustrip=udim),
        }

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        r = r_spherical(xyz, self.units["length"])
        return potential(self._params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        r = r_spherical(xyz, self.units["length"])
        return density(self._params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        """Analytic, from the enclosed mass; see `gradient`."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        return gradient(self._params(t), xyz)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _laplacian(self, xyz: gt.BBtQorVSz3, /, t: gt.BBtQorVSz0) -> gt.BBtFloatSz0:
        """Analytic, from Poisson's equation; see `laplacian`."""
        r = r_spherical(xyz, self.units["length"])
        return laplacian(self._params(t), r)  # type: ignore[no-any-return]

    @vectorize_method(signature="(3),()->(3,3)")
    @ft.partial(jax.jit)
    def _hessian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.Sz33:
        """Analytic, from the radial derivatives; see `hessian`."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        return hessian(self._params(t), xyz)  # type: ignore[no-any-return]

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
            Inner slope of the halo density profile.
        beta
            Outer slope of the halo density profile.
        gamma
            Transition slope of the halo density profile.
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
    return norm_from_bz_half(p, bz_half)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def norm_from_bz_half(p: gt.Params, bz_half: gt.FloatSz0, /) -> gt.FloatSz0:
    """`_norm` given an already-evaluated ``B(alpha(3-gamma), alpha(beta-3), 1/2)``.

    Split out so that a variant with fixed power-law indices can supply that
    value as a constant. See `_norm` for what it means.
    """
    return p["m"] / (4.0 * jnp.pi * p["alpha"] * bz_half)  # type: ignore[no-any-return]


# ===================================================================
# Model functions


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile of the Zhao model (Eq. 1).

    .. math::

        \rho(r) = \frac{C}{u^\gamma (1 + u^{1/\alpha})^{(\beta-\gamma)\alpha}},
        \quad u = r / r_s

    """
    return density_from_norm(p, r, _norm(p))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density_from_norm(
    p: gt.Params, r: gt.BBtSz0, norm: gt.FloatSz0, /
) -> gt.BtFloatSz0:
    """`density` given an already-evaluated normalization. See `density`."""
    alpha, beta, gamma = p["alpha"], p["beta"], p["gamma"]
    uu = r / p["r_s"]
    b = (beta - gamma) * alpha
    _result = norm / p["r_s"] ** 3 / uu**gamma / (1.0 + uu ** (1.0 / alpha)) ** b
    return _result


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Mass enclosed within radius ``r`` (Eq. 15, without the black hole term).

    .. math::

        M(r) = 4\pi\alpha C B(\alpha(3-\gamma), \alpha(\beta-3), \chi)

    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    chi = _r_to_chi(p, r)
    return mass_enclosed_from_bz(p, _norm(p), incomplete_beta(c0 - q0, q0, chi))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed_from_bz(
    p: gt.Params, norm: gt.FloatSz0, bz_interior: gt.BBtSz0, /
) -> gt.BtFloatSz0:
    """`mass_enclosed` given already-evaluated pieces. See `mass_enclosed`."""
    _result = 4.0 * jnp.pi * p["alpha"] * norm * bz_interior
    return _result  # type: ignore[no-any-return]


@ft.partial(jax.jit)
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
    chi = _r_to_chi(p, r)
    return potential_from_bz(  # type: ignore[no-any-return]
        p,
        r,
        chi,
        incomplete_beta(c0 - q0, q0, chi),
        incomplete_beta(c0 - p0, p0, 1.0 - chi),
        density(p, r),
    )


@ft.partial(jax.jit)
def potential_from_bz(
    p: gt.Params,
    r: gt.BBtSz0,
    chi: gt.BBtSz0,
    bz_interior: gt.BBtSz0,
    bz_exterior: gt.BBtSz0,
    rho: gt.BBtSz0,
    /,
) -> gt.BtFloatSz0:
    """Zhao Eqs. 6-7 from already-evaluated pieces. See `potential`."""
    c0, p0, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    chi1 = 1.0 - chi

    # Eq. 7, interior term: B(c0-q0, q0, chi) / (chi^(c0-q0) (1-chi)^q0)
    interior = bz_interior / (chi ** (c0 - q0) * chi1**q0)
    # Eq. 7, exterior term: B(c0-p0, p0, 1-chi) / ((1-chi)^(c0-p0) chi^p0)
    exterior = bz_exterior / (chi1 ** (c0 - p0) * chi**p0)

    f00 = p["alpha"] * (interior + exterior)
    # Eq. 6, with G restored (Zhao sets G = 1).
    _result = -4.0 * jnp.pi * p["G"] * rho * f00 * r**2
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

    """
    return d2potential_dr2_from(p, r, mass_enclosed(p, r), density(p, r))  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def d2potential_dr2_from(
    p: gt.Params, r: gt.BBtSz0, m_enc: gt.BBtSz0, rho: gt.BBtSz0, /
) -> gt.BtFloatSz0:
    """`d2potential_dr2` from already-evaluated pieces. See `d2potential_dr2`."""
    return 4.0 * jnp.pi * p["G"] * rho - 2.0 * p["G"] * m_enc / r**3  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def laplacian(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Laplacian of the potential, i.e. Poisson's equation with Eq. 1.

    .. math::

        \nabla^2 \Phi = 4 \pi G \rho(r)

    """
    return 4.0 * jnp.pi * p["G"] * density(p, r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def gradient(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz3:
    r"""Gradient of the potential.

    .. math::

        \nabla\Phi = \frac{d\Phi}{dr} \hat{r} = \frac{G M(r)}{r^2} \hat{r}

    """
    r = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    return gradient_from_radial(dpotential_dr(p, r), xyz, r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def gradient_from_radial(
    dphi_dr: gt.BBtSz0, xyz: gt.BBtSz3, r: gt.BBtSz0, /
) -> gt.BBtSz3:
    """`gradient` given dPhi/dr. See `gradient`."""
    return dphi_dr * (xyz / r)


@ft.partial(jax.jit)
def hessian(p: gt.Params, xyz: gt.Sz3, /) -> gt.Sz33:
    r"""Hessian of the potential.

    For any spherical potential,

    .. math::

        \partial_i \partial_j \Phi
            = \frac{\Phi'(r)}{r} \delta_{ij}
            + \left(\Phi''(r) - \frac{\Phi'(r)}{r}\right) \frac{x_i x_j}{r^2}

    with $\Phi'$ and $\Phi''$ from `dpotential_dr` and `d2potential_dr2`.
    """
    r = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    return hessian_from_radial(dpotential_dr(p, r), d2potential_dr2(p, r), xyz, r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def hessian_from_radial(
    dphi_dr: gt.BBtSz0, d2phi_dr2: gt.BBtSz0, xyz: gt.Sz3, r: gt.BBtSz0, /
) -> gt.Sz33:
    """`hessian` given the two radial derivatives. See `hessian`."""
    radial = dphi_dr / r
    rhat = xyz / r
    outer = rhat[..., :, None] * rhat[..., None, :]
    eye = jnp.eye(3, dtype=outer.dtype)
    _result = radial[..., None] * eye + (d2phi_dr2 - radial)[..., None] * outer
    return _result  # type: ignore[no-any-return]
