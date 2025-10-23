"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "gNFWPotential",
]

import functools as ft
from dataclasses import KW_ONLY
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from .base import rho0_of_m
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import r_spherical
from galax.utils._jax import vectorize_method

DimL = u.dimension("length")
DimT = u.dimension("time")


@final
class gNFWPotential(AbstractSinglePotential):
    r"""Generalized NFW Potential.

    This potential is a generalized version of the NFW potential, which is a
    popular model for dark matter halos. The density profile is given by:

    $$ \rho(r) = \rho_0 (\frac{r}{r_s})^{-\gamma} (1 + \frac{r}{r_s})^{\gamma-3}
    $$

    where $\rho_0$ is the central density, $r_s$ is the scale radius, and
    $\gamma$ is the inner slope profile.

    $\gamma$ has some notable properties:

    - $\gamma = 0$ is a cored density profile.
    - $\gamma > 0$ is a cuspy density profile.
    - $\gamma = 1$ is the NFW profile.
    - $\gamma >= 2$ has an infinite central potential.

    Therefore we require $\gamma \in [0, 2)$.

    We actually parametrize not with $\rho_0$ but by a characteristic mass $m$,
    defined as

    $$ m = 4 \pi \rho_0 r_s^3. $$

    The enclosed mass is given by:

    $$ M(<r) = m B_{x/(1+x)}(3-\gamma, 0)$$

    where $x = r / r_s$ is the dimensionless radius.

    The potential is given by:

    $$ \Phi(r) = -G \left( M(<r) / r + m / r_s B_{1/(1+x)}(1, 2-\gamma) \right)
    $$

    where $G$ is the gravitational constant.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when $\gamma = 1$:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.potential(x, t)
    Array(-1.23552744, dtype=float64)

    >>> jnp.isclose(gnfw.potential(x, t), nfw.potential(x, t), atol=1e-8)
    Array(True, dtype=bool)

    >>> x = jnp.array([0, 0, 0])
    >>> gnfw.potential(x, t), nfw.potential(x, t)
    (Array(nan, dtype=float64), Array(nan, dtype=float64))

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of profiles. For example, if we set $\gamma =
    0.5$, we get a density profile that is steeper than the NFW profile at small
    radii, so the potential is smaller:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.potential(x, t)
    Array(-1.09363913, dtype=float64)

    >>> gnfw2.potential(x, t) > gnfw.potential(x, t)
    Array(True, dtype=bool)

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass",
        doc="""Scale mass. This is NOT the total mass.""",
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]
    gamma: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Slope of the density profile at small radii.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def _density(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "gamma": self.gamma(t, ustrip=self.units["dimensionless"]),
        }
        return density(params, r)

    @ft.partial(jax.jit)
    def _potential(  # TODO: inputs w/ units
        self, xyz: gt.Sz3, t: gt.Sz0, /
    ) -> gt.Sz0:
        r = r_spherical(xyz, self.units["length"])
        t = u.Quantity.from_(t, self.units["time"])

        params = {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "gamma": self.gamma(t, ustrip=self.units["dimensionless"]),
        }
        return potential(params, r)

    @vectorize_method(signature="(3),()->(3)")
    @ft.partial(jax.jit)
    def _gradient(  # TODO: inputs w/ units
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.FloatSz3:
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t_ = u.Quantity.from_(t, self.units["time"])
        params = {
            "G": self.constants["G"].value,
            "m": self.m(t_, ustrip=self.units["mass"]),
            "r_s": self.r_s(t_, ustrip=self.units["length"]),
            "gamma": self.gamma(t_, ustrip=self.units["dimensionless"]),
        }
        return gradient(params, xyz)


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Density profile for the gNFW model.

    $$ \rho(r) = \rho_0 (\frac{r}{r_s})^{-\gamma} (1 + \frac{r}{r_s})^{\gamma-3}
    $$

    where $\rho_0$ is the central density, $r_s$ is the scale radius, and
    $\gamma$ is the inner slope profile. We actually parametrize not with
    $\rho_0$ but by a characteristic mass $m$, defined as

    $$ m = 4 \pi \rho_0 r_s^3. $$

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when $\gamma = 1$:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.density(x, t)
    Array(1.2280474e+08, dtype=float64)

    >>> jnp.isclose(gnfw.density(x, t), nfw.density(x, t), atol=1e-8)
    Array(True, dtype=bool)

    >>> x = jnp.array([0, 0, 0])
    >>> jnp.isclose(gnfw.density(x, t), nfw.density(x, t), atol=1e-8)
    Array(True, dtype=bool)

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of density profiles. For example, if we set
    $\gamma = 0.5$, we get a density profile that is steeper than the NFW
    profile at small radii and so the density falls off faster:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.density(x, t)
    Array(1.15781419e+08, dtype=float64)

    >>> gnfw2.density(x, t) < gnfw.density(x, t)
    Array(True, dtype=bool)

    """
    x = r / p["r_s"]
    rho0: gt.Sz0 = rho0_of_m(p)
    return rho0 * jnp.power(x, -p["gamma"]) * jnp.power(1 + x, p["gamma"] - 3)


# -----------------------------------------------

hyp2f1 = tfp.math.hypergeometric.hyp2f1_small_argument


@ft.partial(jax.jit)
def Bz_from_hyp2f1(a: gt.FloatSz0, b: gt.FloatSz0, z: gt.BBtFloatSz0) -> gt.BBtFloatSz0:
    r"""Incomplete beta function from hypergeometric function.

    $$ B_z(a, 0) = \frac{z^a}{a} \cdot {}_2F_1(a, 1 - b; a + 1; z) $$

    See NIST DLMF 8.17.7 @ https://dlmf.nist.gov/8.17

    Parameters
    ----------
    a, b
        The parameters of the incomplete beta function.
    z
        The value at which to evaluate the incomplete beta function.
        Must be in the range [0, 1].

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.scipy.special as jsp

    >>> a, b = 1.0, 2.0
    >>> z = jnp.array(0.5)

    >>> Bz_from_hyp2f1(a, b, z)
    Array(0.375, dtype=float64)

    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(0.375, dtype=float64)

    `Bz_from_hyp2f1` works for b = 0:

    >>> b = 0.0
    >>> Bz_from_hyp2f1(a, b, z)
    Array(0.69314718, dtype=float64)

    But `jsp.beta` does not work for b = 0:

    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(nan, dtype=float64)

    We can confirm that `Bz_from_hyp2f1` is correct by comparison when $b \sim 0$:

    >>> b = 1e-4
    >>> jsp.beta(a,b) * jsp.betainc(a, b, z)
    Array(0.69312316, dtype=float64)

    """
    return (z**a / a) * hyp2f1(a, 1 - b, a + 1, z)


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Enclosed mass for the NFW model.

    $$ M(<r) = m B_{x/(1+x)}(3-\gamma, 0)$$

    where $x = r / r_s$ is the dimensionless radius, $m$ is the characteristic
    mass, and $\gamma$ is the inner slope profile. $B_{x/(1+x)}$ is the
    incomplete beta function.

    Note that `jax.scipy.special.betainc` is the regularized incomplete beta
    function, which is related to the incomplete beta function by $$ B_z(a, b) =
    B(a, b) \cdot \text{betainc}(a, b, z) $$. However, $B(a, b)$ diverges for $b
    = 0$, so we use the 2F2 hypergeometric function instead.

    """
    x = r / p["r_s"]
    z = x / (1 + x)
    return p["m"] * Bz_from_hyp2f1(3.0 - p["gamma"], 0.0, z)


# -----------------------------------------------


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
    r"""Potential for the gNFW model.

    We solve the equation for the gNFW model, which is given by

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when $\gamma = 1$:

    >>> gnfw = gp.gNFWPotential(m=1e12,r_s=1,gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12,r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.potential(x, t)
    Array(-1.23552744, dtype=float64)

    >>> jnp.isclose(gnfw.potential(x, t), nfw.potential(x, t), atol=1e-8)
    Array(True, dtype=bool)

    >>> x = jnp.array([0, 0, 0])
    >>> gnfw.potential(x, t), nfw.potential(x, t)
    (Array(nan, dtype=float64), Array(nan, dtype=float64))

    The gNFW potential is a generalization of the NFW potential, so it can be
    used to model a wider range of profiles. For example, if we set
    $\gamma = 0.5$, we get a density profile that is steeper than the NFW
    profile at small radii, so the potential is smaller:

    >>> gnfw2 = gp.gNFWPotential(m=1e12,r_s=1,gamma=0.5, units="galactic")

    >>> x = jnp.array([8, 0, 0])
    >>> gnfw2.potential(x, t)
    Array(-1.09363913, dtype=float64)

    >>> gnfw2.potential(x, t) > gnfw.potential(x, t)
    Array(True, dtype=bool)

    """
    rs = p["r_s"]

    # Inner term from integral 0 to r
    inner = mass_enclosed(p, r) / r

    # Outer term from integral r to infinity
    z2 = 1 / (1 + r / rs)
    outer = (p["m"] / rs) * Bz_from_hyp2f1(1.0, 2.0 - p["gamma"], z2)

    return -p["G"] * (inner + outer)


@ft.partial(jax.jit)
def gradient(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz3:
    r"""Gradient of the potential for the gNFW model.

    $$ \nabla \Phi(r) = G M(<r) / r^2 \hat{r} $$

    where $M(<r)$ is the enclosed mass.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp

    A gNFW potential matches the NFW potential when $\gamma = 1$:

    >>> gnfw = gp.gNFWPotential(m=1e12, r_s=1, gamma=1, units="galactic")
    >>> nfw = gp.NFWPotential(m=1e12, r_s=1, units="galactic")

    >>> x, t = jnp.array([8, 0, 0]), 0
    >>> gnfw.gradient(x, t)
    Array([0.09196173, 0.        , 0.        ], dtype=float64)

    >>> jnp.allclose(gnfw.gradient(x, t), nfw.gradient(x, t), atol=1e-8)
    Array(True, dtype=bool)

    """
    r_mag = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    mass_enc = mass_enclosed(p, r_mag)
    grad_mag = p["G"] * mass_enc / (r_mag**2)
    return grad_mag * (xyz / r_mag)
