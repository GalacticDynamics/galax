r"""Evaluation of a multipole profile expansion.

Function-first, matching `galax.potential._src.builtin.zhao`: the numerics
take a flat ``gt.Params`` dict, and the potential classes build that dict from
their own parameters, so the numerics can be reused without inheriting from a
concrete potential class.

``p`` carries ``r_knots``, ``phi_lm``, ``dphi_lm``, ``phi_asympt_powers``,
``phi_asympt_scales``, ``rho_residual_lm``, ``drho_residual_lm``,
``rho_alpha`` and ``rho_amplitude``, each already stripped to the
potential's unit system. `build_expansion` returns exactly these.
"""

__all__: tuple[str, ...] = ()

import abc
import functools as ft

from jaxtyping import Array, Float

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.harmonic import (
    eval_log_spline,
    eval_log_spline_asympt,
    real_ylm,
)
from galax.potential._src.utils import safe_vector_norm


def _log_r_and_ylm(
    xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch n_modes"]]:
    r"""Split positions into :math:`\log r` and a mode-minor harmonic table."""
    r = safe_vector_norm(xyz)
    uvec = xyz / r[..., None]
    # `real_ylm` is mode-major; the radial splines are mode-minor.
    return jnp.log(r), jnp.moveaxis(real_ylm(l_max, keys, uvec), 0, -1)


@ft.partial(jax.jit, static_argnums=(2, 3))
def expansion_potential(
    p: gt.Params, xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> Float[Array, "*batch"]:
    r""":math:`\Phi = \sum_{lm} \Phi_{lm}(r) Y_{lm}(\hat{q})`.

    Outside ``[r_knots[0], r_knots[-1]]`` each mode continues as the power law
    fitted by `asymptotic_coeffs`, joined to the spline in value and slope.
    Inside the grid this is bit-identical to the spline alone.
    """
    log_r, Y = _log_r_and_ylm(xyz, l_max, keys)
    coefs = jnp.concatenate([p["phi_asympt_powers"], p["phi_asympt_scales"]], axis=1)
    phi_lm = eval_log_spline_asympt(
        jnp.log(p["r_knots"]), p["phi_lm"], p["dphi_lm"], coefs, log_r
    )
    return jnp.sum(phi_lm * Y, axis=-1)  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnums=(2, 3))
def expansion_density(
    p: gt.Params, xyz: gt.BtSz3, l_max: int, keys: tuple[tuple[int, int], ...], /
) -> Float[Array, "*batch"]:
    r""":math:`\rho = \sum_{lm} \rho_{lm}(r) Y_{lm}(\hat{q})`.

    :math:`\rho_{lm}` is the splined residual plus the analytic inner power
    law, :math:`\mathrm{amplitude} (r/r_0)^\alpha`.
    """
    log_r, Y = _log_r_and_ylm(xyz, l_max, keys)
    log_r0 = jnp.log(p["r_knots"][0])
    residual = eval_log_spline(
        jnp.log(p["r_knots"]),
        p["rho_residual_lm"],
        p["drho_residual_lm"],
        log_r,
    )
    background = p["rho_amplitude"] * jnp.exp(
        p["rho_alpha"] * (log_r[..., None] - log_r0)
    )
    return jnp.sum((residual + background) * Y, axis=-1)  # type: ignore[no-any-return]


class MultipoleProfileMixin(AbstractSinglePotential):
    """Supply ``_potential``/``_density`` from ``_params``, ``l_max``, ``lm_keys``.

    Carries the shared evaluation implementation but none of the field
    declarations, so a potential that stores its expansion differently can
    reuse the numerics by supplying ``_params``, ``l_max`` and ``lm_keys``.
    Mix it in ahead of the concrete class so that its ``_potential`` and
    ``_density`` win over `AbstractPotential`'s abstract stubs, the same way
    `LaplacianFromDensityMixin` is used.
    """

    #: The maximum multipole order.
    l_max: eqx.AbstractVar[int]

    @property
    @abc.abstractmethod
    def lm_keys(self) -> tuple[tuple[int, int], ...]:
        """The (l, m) mode keys, in the order the coefficient columns use."""
        ...

    @abc.abstractmethod
    def _params(self, t: gt.BBtQorVSz0, /) -> gt.Params:
        """Build the parameter dictionary from the potential's fields."""
        ...

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        return expansion_potential(  # type: ignore[no-any-return]
            self._params(t),
            xyz,
            self.l_max,
            self.lm_keys,
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        return expansion_density(  # type: ignore[no-any-return]
            self._params(t),
            xyz,
            self.l_max,
            self.lm_keys,
        )
