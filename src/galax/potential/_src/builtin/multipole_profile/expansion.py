r"""Evaluation of a multipole profile expansion.

Function-first, matching `galax.potential._src.builtin.zhao`: the numerics
take a flat ``gt.Params`` dict, and the potential classes build that dict from
their own parameters. Sub-projects 3 and 4 reuse these functions and the mixin
rather than inheriting from a concrete class.

``p`` carries ``r_knots``, ``phi_lm``, ``dphi_lm``, ``rho_residual_lm``,
``drho_residual_lm``, ``rho_alpha`` and ``rho_amplitude``, each already
stripped to the potential's unit system.
"""

__all__: tuple[str, ...] = ()

import functools as ft

from jaxtyping import Array, Float

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .funcs import eval_log_spline
from .project import real_ylm
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
    r""":math:`\Phi = \sum_{lm} \Phi_{lm}(r) Y_{lm}(\hat{q})`."""
    log_r, Y = _log_r_and_ylm(xyz, l_max, keys)
    phi_lm = eval_log_spline(jnp.log(p["r_knots"]), p["phi_lm"], p["dphi_lm"], log_r)
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


class MultipoleProfileMixin(eqx.Module):
    """Supply ``_potential``/``_density`` from ``_params``, ``l_max``, ``lm_keys``.

    A mixin rather than a base class so the transform wrapper of sub-project 4
    can combine it with `AbstractTransformedPotential` without putting a
    second `AbstractPotential` in its MRO. Mirrors the existing
    `LaplacianFromDensityMixin` pattern.
    """

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)  # type: ignore[attr-defined]
        return expansion_potential(  # type: ignore[no-any-return]
            self._params(t),  # type: ignore[attr-defined]
            xyz,
            self.l_max,  # type: ignore[attr-defined]
            self.lm_keys,  # type: ignore[attr-defined]
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)  # type: ignore[attr-defined]
        return expansion_density(  # type: ignore[no-any-return]
            self._params(t),  # type: ignore[attr-defined]
            xyz,
            self.l_max,  # type: ignore[attr-defined]
            self.lm_keys,  # type: ignore[attr-defined]
        )
