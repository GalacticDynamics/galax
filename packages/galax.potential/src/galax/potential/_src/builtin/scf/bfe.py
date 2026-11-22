"""Self-Consistent Field basis functions."""

__all__ = ["SCFPotential", "phi_nl", "rho_nl"]

import functools as ft
from dataclasses import KW_ONLY

from jaxtyping import Array, Float
from typing import Final, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from spexial import eval_gegenbauers
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.builtin.multipole import scaled_radius_and_direction
from galax.potential._src.builtin.sph_expansion import (
    AbstractRadialBasisExpansionPotential,
    sum_basis,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

SQRT_FOURPI: Final = 3.544907701811031
"""``sqrt(4 * pi)``, matching the literal in gala's ``bfe_helper.cpp``."""


def _nl_axes(
    n_max: int, l_max: int, s: Float[Array, "*batch"], /
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Broadcastable ``n``, ``l`` and Gegenbauer table for the given ``s``."""
    nbatch = jnp.ndim(s)
    ls = jnp.arange(l_max + 1, dtype=s.dtype)
    # n gets one axis for l plus one per batch axis; l gets one per batch axis.
    n = jnp.expand_dims(
        jnp.arange(n_max + 1, dtype=s.dtype), tuple(range(1, 2 + nbatch))
    )
    l = jnp.expand_dims(ls, tuple(range(1, 1 + nbatch)))
    # `eval_gegenbauers` broadcasts `alpha` against `x` and stacks the orders on
    # a new leading axis, so giving `alpha` the batch axes here is what turns a
    # broadcast into the outer product this wants: (n_max+1, l_max+1, *batch).
    cn = eval_gegenbauers(n_max, l * 2 + 1.5, (s - 1) / (s + 1))
    return n, l, cn


def _s_pow_l(
    l_max: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{l_max}+1 *batch"]:
    r""":math:`s^l` for :math:`l = 0 \ldots l_{max}`, stacked on a leading axis.

    Built with *integer* exponents rather than as ``s ** l`` against a float
    ``l`` array. The two agree in value, but the float form is not twice
    differentiable at :math:`s = 0`: the derivative of :math:`s^0` is
    :math:`0 \cdot s^{-1}`, which JAX evaluates as ``0 * inf`` and returns
    ``nan``, so the hessian of every SCF potential was NaN at the origin even
    though the value and gradient were finite. ``l_max`` is static, so the
    comprehension unrolls at trace time and `jax` uses `lax.integer_pow`,
    whose derivative at zero is exact.
    """
    return jnp.stack([s**i for i in range(l_max + 1)])  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnums=(0, 1))
def phi_nl(
    n_max: int, l_max: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{n_max}+1 {l_max}+1 *batch"]:
    r"""Radial potential expansion terms.

    $$ \phi_{nl}(s) = -\sqrt{4\pi} \frac{s^l}{(1+s)^{2l+1}} C_n^{2l+3/2}(\xi) $$

    with $\xi = (s-1)/(s+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import phi_nl

    The $n = l = 0$ term is the Hernquist profile up to normalization:

    >>> bool(jnp.allclose(phi_nl(0, 0, jnp.asarray(1.0)),
    ...                   -3.544907701811031 / 2))
    True

    """
    _, l, cn = _nl_axes(n_max, l_max, s)
    prefactor = -SQRT_FOURPI * _s_pow_l(l_max, s) / (1 + s) ** (2 * l + 1)
    return prefactor * cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def rho_nl(
    n_max: int, l_max: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{n_max}+1 {l_max}+1 *batch"]:
    r"""Radial density expansion terms.

    $$ \rho_{nl}(s) = \sqrt{4\pi} \frac{K_{nl}}{2\pi}
                      \frac{s^l}{s(1+s)^{2l+3}} C_n^{2l+3/2}(\xi) $$

    with $K_{nl} = \frac{1}{2}n(n+4l+3) + (l+1)(2l+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import rho_nl

    >>> rho_nl(0, 0, jnp.asarray(1.0)).shape
    (1, 1)

    """
    n, l, cn = _nl_axes(n_max, l_max, s)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    prefactor = (
        SQRT_FOURPI
        * (knl / (2 * jnp.pi))
        * _s_pow_l(l_max, s)
        / (s * (1 + s) ** (2 * l + 3))
    )
    return prefactor * cn  # type: ignore[no-any-return]


@final
class SCFPotential(AbstractRadialBasisExpansionPotential):
    r"""Self-Consistent Field (SCF) basis function expansion potential.

    The method of Hernquist & Ostriker (1992) and Lowing et al. (2011), with
    all coefficients real.

    $$ \Phi(r,\theta,\phi) = \frac{G M}{r_s} \sum_{nlm} \phi_{nl}(s)
       \left[ S_{nlm} \Re Y_l^m + T_{nlm} \Im Y_l^m \right] $$

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    The monopole term alone is the Hernquist potential:

    >>> Snlm = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    >>> pot = gp.SCFPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"),
    ...                       Snlm=Snlm, Tnlm=jnp.zeros_like(Snlm),
    ...                       units="galactic")
    >>> pot.n_max, pot.l_max
    (0, 0)

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Scale mass."
    )
    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )
    Snlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\cos(m\phi)$ terms, shape "
        r"``(n_max+1, l_max+1, l_max+1)``.",
    )
    Tnlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\sin(m\phi)$ terms, shape "
        r"``(n_max+1, l_max+1, l_max+1)``.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    n_max: int = eqx.field(init=False, static=True, repr=False)
    l_max: int = eqx.field(init=False, static=True, repr=False)

    # ==========================================================================

    def _radial_basis(
        self, xyz: gt.BBtSz3, p: dict[str, Array], /
    ) -> tuple[Float[Array, "{n_max}+1 {l_max}+1 *batch"], gt.BtSz3, Array]:
        s, uvec = scaled_radius_and_direction(xyz, p["r_s"])
        return phi_nl(self.n_max, self.l_max, s), uvec, p["G"] * p["m_tot"] / p["r_s"]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        if jnp.ndim(t) > 0:  # a time per position: coefficients differ per time
            return jnp.vectorize(self._density, signature="(3),()->()")(xyz, t)  # type: ignore[no-any-return]

        p = self.params(t)

        s, uvec = scaled_radius_and_direction(xyz, p["r_s"])
        rhonl = rho_nl(self.n_max, self.l_max, s)
        summation = sum_basis(self.l_max, rhonl, p["Snlm"], p["Tnlm"], uvec)

        return p["m_tot"] / p["r_s"] ** 3 * summation  # type: ignore[no-any-return]
