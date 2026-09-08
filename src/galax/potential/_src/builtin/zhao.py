r"""Zhao (1996) double power-law potential.

Zhao, H. 1996, MNRAS 278, 488 ("Analytical Models For Galactic Nuclei"),
https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z. Equation numbers in
this module refer to that paper.

Everything this model needs -- potential (Eqs. 6-7), enclosed mass (Eq. 15) and
density (Eq. 1) -- is closed-form in terms of the incomplete beta function
(Eq. 43), so *no* quantity here is obtained by automatic differentiation:
`gradient`, `hessian` and `laplacian` are all written analytically below. That
matters for speed, because the alternative is differentiating through the
incomplete beta function's series expansion on every force evaluation.
"""

__all__ = ["ZhaoPotential"]

import functools as ft

from typing import final

import equinox as eqx
import jax
from jax.custom_derivatives import SymbolicZero

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
# The incomplete beta function (Zhao Eq. 43)
#
# Zhao writes every dynamical quantity in terms of
#
#     B(a, b, x) = int_0^x t^(a-1) (1-t)^(b-1) dt              (Eq. 43)
#
# the *unregularized* incomplete beta function. `jax.scipy.special` provides
# only the regularized `betainc`, and reconstructing Eq. 43 as
# `beta(a, b) * betainc(a, b, x)` is nan whenever `b <= 0` -- exactly the
# beta <= 3 (infinite total mass) regime -- because the complete beta function
# diverges there even though the product does not.
#
# `jax.scipy.special.hyp2f1` can express it (DLMF 8.17.7) for any `b`, but it
# is a `lax.while_loop` whose trip count is data-dependent: under `vmap` every
# lane pays the worst lane's iteration count, and its derivative runs a second
# such loop. The two fixed-length, geometrically convergent series below cost a
# predictable ~64 fused multiply-adds instead, and carry an exact O(1)
# derivative rule.
#
# (`nfw/hyp2f1.py` solves the same problem for `gNFWPotential`, but only for
# the two parameter patterns that model needs -- `a == 1`, or `b == 0`. Zhao's
# Eqs. 7 and 15 need general `(a, b)`.)

_BZ_NTERMS = 64
"""Series length. Both series converge like 2^-k, so this is ~1e-19."""


def _bz_small_z(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ for $z \leq 1/2$, by the defining Taylor series.

    Expanding $(1-t)^{b-1}$ binomially in Eq. 43 and integrating term by term,

    .. math::

        B(a, b, z) = z^a \sum_{k=0}^\infty \frac{(1-b)_k}{k!\,(a+k)} z^k

    The terms fall off like $z^k \leq 2^{-k}$, hence the fixed term count.

    Summed with `jax.lax.scan`, accumulating into the carry: the terms are
    batched over `z`, so materializing them all at once would cost an
    ``(*batch, 64)`` temporary and make this memory- rather than flop-bound.
    """

    def step(
        carry: tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0], k: gt.Sz0
    ) -> tuple[tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0], None]:
        total, z_pow, coeff = carry  # coeff = (1-b)_k / k!
        total = total + coeff / (a + k) * z_pow
        return (total, z_pow * z, coeff * (k + 1.0 - b) / (k + 1.0)), None

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones_like(a))
    (total, _, _), _ = jax.lax.scan(step, init, jnp.arange(_BZ_NTERMS))
    return z**a * total  # type: ignore[no-any-return]


def _bz_large_z(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ for $z > 1/2$, by reflecting about $t = 1/2$.

    Substituting $t = 1-u$ in Eq. 43 and splitting the range at $u = 1/2$,

    .. math::

        B(a, b, z) = \int_{w}^{1} u^{b-1}(1-u)^{a-1} du
                   = B(a, b, 1/2) + \int_w^{1/2} u^{b-1}(1-u)^{a-1} du

    with $w = 1 - z$. Expanding $(1-u)^{a-1}$ binomially (legitimate since
    $u \leq 1/2$ on the remaining range) and integrating term by term,

    .. math::

        B(a, b, z) = B(a, b, 1/2)
            + \sum_{m=0}^\infty \frac{(1-a)_m}{m!}
              \frac{(1/2)^{b+m} - w^{b+m}}{b+m}

    where the $b + m \to 0$ term is $\ln(1/(2w))$. Both pieces converge like
    $2^{-m}$.

    This branch is what makes $b \leq 0$ work: it never forms the complete beta
    function $B(a, b)$, which is what diverges there, while keeping the genuine
    $z \to 1$ divergence of $B(a, b, z)$ itself exact (as $w^b$, or $-\ln w$
    when $b = 0$).

    Summed with `jax.lax.scan` for the same reason as `_bz_small_z`.
    """
    w = 1.0 - z
    log_half_over_w = jnp.log(0.5 / w)

    def step(
        carry: tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0, gt.Sz0], m: gt.Sz0
    ) -> tuple[tuple[gt.BBtFloatSz0, gt.BBtFloatSz0, gt.Sz0, gt.Sz0], None]:
        total, w_pow, half_pow, coeff = carry  # coeff = (1-a)_m / m!
        # ((1/2)^(b+m) - w^(b+m)) / (b+m), whose b+m -> 0 limit is log(1/(2w)).
        bm = b + m
        is_pole = jnp.abs(bm) < 1e-12
        bm_safe = jnp.where(is_pole, 1.0, bm)
        term = jnp.where(is_pole, log_half_over_w, (half_pow - w_pow) / bm_safe)
        total = total + coeff * term
        carry = (total, w_pow * w, half_pow * 0.5, coeff * (m + 1.0 - a) / (m + 1.0))
        return carry, None

    init = (jnp.zeros_like(w), w**b, 0.5**b, jnp.ones_like(a))
    (total, _, _, _), _ = jax.lax.scan(step, init, jnp.arange(_BZ_NTERMS))
    return _bz_small_z(a, b, jnp.full_like(w, 0.5)) + total  # type: ignore[no-any-return]


def _incomplete_beta_impl(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    r"""$B(a, b, z)$ (Zhao Eq. 43) for $a > 0$, any real $b$, $z \in [0, 1)$.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp
    >>> from galax.potential._src.builtin.zhao import incomplete_beta

    It agrees with the regularized incomplete beta function wherever that is
    defined, including close to the $z \to 1$ endpoint:

    >>> a, b = 2.0, 1.5
    >>> z = jnp.asarray([0.3, 0.999])
    >>> bool(jnp.allclose(incomplete_beta(a, b, z),
    ...                   jsp.beta(a, b) * jsp.betainc(a, b, z)))
    True

    But unlike that product it stays finite for ``b <= 0``, where the complete
    beta function diverges:

    >>> incomplete_beta(2.0, 0.0, jnp.asarray(0.5))
    Array(0.19314718, dtype=float64)

    >>> jsp.beta(2.0, 0.0) * jsp.betainc(2.0, 0.0, jnp.asarray(0.5))
    Array(nan, dtype=float64)

    """
    z = jnp.asarray(z)
    # Both branches are evaluated, so clamp each one's input into the range
    # where it is well behaved; `where` then discards the unused value.
    return jnp.where(  # type: ignore[no-any-return]
        z <= 0.5,
        _bz_small_z(a, b, jnp.minimum(z, 0.5)),
        _bz_large_z(a, b, jnp.maximum(z, 0.5)),
    )


@jax.custom_jvp
def incomplete_beta(a: gt.Sz0, b: gt.Sz0, z: gt.BBtSz0) -> gt.BBtFloatSz0:
    """See `_incomplete_beta_impl` for the definition and examples.

    The `jax.custom_jvp` gives the `z`-derivative in O(1) -- it is just the
    Eq. 43 integrand evaluated at the endpoint (Leibniz) -- instead of
    differentiating through a 64-term series. It is a `custom_jvp`, not a
    `custom_vjp`, so that `jax.hessian`'s `jacfwd(jacrev(...))` still works for
    anything downstream that has not been given an analytic form.
    """
    return _incomplete_beta_impl(a, b, z)


@ft.partial(incomplete_beta.defjvp, symbolic_zeros=True)
def _incomplete_beta_jvp(
    primals: tuple[gt.Sz0, gt.Sz0, gt.BBtSz0],
    tangents: tuple[
        gt.Sz0 | SymbolicZero, gt.Sz0 | SymbolicZero, gt.BBtSz0 | SymbolicZero
    ],
) -> tuple[gt.BBtFloatSz0, gt.BBtFloatSz0]:
    a, b, z = primals
    a_dot, b_dot, z_dot = tangents

    primal_out = _incomplete_beta_impl(a, b, z)
    tangent_out = jnp.zeros_like(primal_out)

    # d/dz B(a, b, z) = z^(a-1) (1-z)^(b-1): the Eq. 43 integrand at z.
    if not isinstance(z_dot, SymbolicZero):
        tangent_out = tangent_out + z ** (a - 1.0) * (1.0 - z) ** (b - 1.0) * z_dot

    # The a/b tangents are only needed when the power-law indices are
    # themselves differentiated; there is no cheap closed form, so fall back to
    # autodiff of the series. Skipped entirely in the common case, where the
    # differentiation is with respect to position at fixed alpha/beta/gamma.
    a_zero, b_zero = isinstance(a_dot, SymbolicZero), isinstance(b_dot, SymbolicZero)
    if not (a_zero and b_zero):
        _, ab_tangent = jax.jvp(
            lambda aa, bb: _incomplete_beta_impl(aa, bb, z),
            (a, b),
            (
                jnp.zeros_like(a) if a_zero else a_dot,
                jnp.zeros_like(b) if b_zero else b_dot,
            ),
        )
        tangent_out = tangent_out + ab_tangent

    return primal_out, tangent_out


incomplete_beta = jax.jit(incomplete_beta)  # type: ignore[assignment]


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
    r"""$M(r_s) / M_{tot}$, from Eq. 15 evaluated at $\chi = 1/2$ and $\chi = 1$.

    Only used by `ZhaoPotential.from_m_tot`, which requires $\beta > 3$ so that
    $M_{tot} = M(\infty)$ is finite.
    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    a = c0 - q0
    total = incomplete_beta(a, q0, jnp.asarray(1.0))  # B(a, q0), q0 > 0 here
    return incomplete_beta(a, q0, jnp.asarray(0.5)) / total


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

        \rho(r) = \frac{C}{u^\gamma (1 + u^{1/\alpha})^{(\beta-\gamma)\alpha}},
        \quad u = r / r_s

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

        M(r) = 4\pi\alpha C B(\alpha(3-\gamma), \alpha(\beta-3), \chi)

    """
    c0, _, q0 = _cpq(p["alpha"], p["beta"], p["gamma"])
    chi = _r_to_chi(p, r)
    _result = 4.0 * jnp.pi * p["alpha"] * _norm(p) * incomplete_beta(c0 - q0, q0, chi)
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
    chi1 = 1.0 - chi

    # Eq. 7, interior term: B(c0-q0, q0, chi) / (chi^(c0-q0) (1-chi)^q0)
    interior = incomplete_beta(c0 - q0, q0, chi) / (chi ** (c0 - q0) * chi1**q0)
    # Eq. 7, exterior term: B(c0-p0, p0, 1-chi) / ((1-chi)^(c0-p0) chi^p0)
    exterior = incomplete_beta(c0 - p0, p0, chi1) / (chi1 ** (c0 - p0) * chi**p0)

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

    """
    interior = 2.0 * p["G"] * mass_enclosed(p, r) / r**3
    return 4.0 * jnp.pi * p["G"] * density(p, r) - interior  # type: ignore[no-any-return]


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
    return dpotential_dr(p, r) * (xyz / r)  # type: ignore[no-any-return]


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
    radial = dpotential_dr(p, r) / r
    d2phi_dr2 = d2potential_dr2(p, r)

    rhat = xyz / r
    outer = rhat[..., :, None] * rhat[..., None, :]
    eye = jnp.eye(3, dtype=outer.dtype)
    _result = radial[..., None] * eye + (d2phi_dr2 - radial)[..., None] * outer
    return _result  # type: ignore[no-any-return]
