"""Multipole potential."""

__all__ = [
    "AbstractMultipolePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
]

import functools as ft
import math
from dataclasses import KW_ONLY

from jaxtyping import Array, Float
from typing import final

import equinox as eqx
import jax
import numpy as np
from equinox import field

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.utils import safe_vector_norm


class AbstractMultipolePotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Abstract Multipole Potential.

    Each term :math:`r^l Y_{lm}(\theta,\phi)` ("inner") and :math:`r^{-(l+1)}
    Y_{lm}(\theta,\phi)` ("outer") is a *solid harmonic*: an exact,
    source-free solution of Laplace's equation, :math:`\nabla^2 \Phi = 0`,
    for every :math:`l, m` (Binney & Tremaine 2008, *Galactic Dynamics*, 2nd
    ed., Sec. 2.4). Since :math:`\nabla^2` is linear, any sum of such terms
    — inner, outer, or both, as used by the concrete subclasses below — is
    itself source-free away from the origin. So the density is exactly zero
    everywhere these potentials are evaluated (:math:`r>0`); all of the
    represented mass sits, formally, at :math:`r=0` (for outer/mixed terms)
    or at infinity (for inner terms only).
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass."
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    _: KW_ONLY
    l_max: int = field(static=True)

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, _: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        return jnp.zeros(xyz.shape[:-1], dtype=xyz.dtype)  # type: ignore[no-any-return]


@final
class MultipoleInnerPotential(AbstractMultipolePotential):
    r"""Multipole inner expansion potential.

    .. math::

        \Phi^l_\mathrm{max}(r,\theta,\phi) =
            \sum_{l=0}^{l=l_\mathrm{max}}\sum_{m=0}^{m=l}
            r^l \, (S_{lm} \, \cos{m\,\phi} + T_{lm} \, \sin{m\,\phi})
            \, P_l^m(\cos\theta)

    """

    Slm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Spherical harmonic coefficients for the $\cos(m \phi)$ terms.",
    )

    Tlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Spherical harmonic coefficients for the $\sin(m \phi)$ terms.",
    )

    def __check_init__(self) -> None:
        shape = (self.l_max + 1, self.l_max + 1)
        t = u.Q(0.0, "Gyr")
        s_shape, t_shape = self.Slm(t).shape, self.Tlm(t).shape
        # TODO: check shape across time.
        msg = (
            "Slm and Tlm must have the shape (l_max + 1, l_max + 1). "
            f"Slm shape: {s_shape}, Tlm shape: {t_shape}"
        )
        _ = eqx.error_if(t, jnp.logical_or(s_shape != shape, t_shape != shape), msg)

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        Slm = self.Slm(t, ustrip=self.units["dimensionless"])
        Tlm = self.Tlm(t, ustrip=self.units["dimensionless"])

        # scaled radius and unit direction
        is_scalar = xyz.ndim == 1
        s, uvec = scaled_radius_and_direction(jnp.atleast_2d(xyz), r_s)

        # Compute the summation over l and m
        terms = [
            jnp.pow(s, l) * (Slm[l, m] * cYlm + Tlm[l, m] * sYlm)
            for l, m, cYlm, sYlm in iter_Ylm(self.l_max, uvec)
        ]
        summation = jnp.sum(jnp.stack(terms), axis=0)
        if is_scalar:
            summation = summation[0]

        _result = self.constants["G"].value * m_tot / r_s * summation
        return _result  # type: ignore[no-any-return]


@final
class MultipoleOuterPotential(AbstractMultipolePotential):
    r"""Multipole outer expansion potential.

    .. math::

        \Phi^l_\mathrm{max}(r,\theta,\phi) =
            \sum_{l=0}^{l=l_\mathrm{max}}\sum_{m=0}^{m=l}
            r^{-(l+1)} \, (S_{lm} \, \cos{m\,\phi} + T_{lm} \, \sin{m\,\phi})
            \, P_l^m(\cos\theta)

    """

    Slm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Spherical harmonic coefficients for the $\cos(m \phi)$ terms.",
    )

    Tlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Spherical harmonic coefficients for the $\sin(m \phi)$ terms.",
    )

    def __check_init__(self) -> None:
        shape = (self.l_max + 1, self.l_max + 1)
        t = u.Q(0.0, "Gyr")
        s_shape, t_shape = self.Slm(t).shape, self.Tlm(t).shape
        # TODO: check shape across time.
        msg = (
            "Slm and Tlm must have the shape (l_max + 1, l_max + 1). "
            f"Slm shape: {s_shape}, Tlm shape: {t_shape}"
        )
        _ = eqx.error_if(t, jnp.logical_or(s_shape != shape, t_shape != shape), msg)

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        Slm = self.Slm(t, ustrip=self.units["dimensionless"])
        Tlm = self.Tlm(t, ustrip=self.units["dimensionless"])

        # scaled radius and unit direction
        is_scalar = xyz.ndim == 1
        s, uvec = scaled_radius_and_direction(jnp.atleast_2d(xyz), r_s)

        # Compute the summation over l and m
        terms = [
            jnp.pow(s, -(l + 1)) * (Slm[l, m] * cYlm + Tlm[l, m] * sYlm)
            for l, m, cYlm, sYlm in iter_Ylm(self.l_max, uvec)
        ]
        summation = jnp.sum(jnp.stack(terms), axis=0)
        if is_scalar:
            summation = summation[0]

        _result = self.constants["G"].value * m_tot / r_s * summation
        return _result  # type: ignore[no-any-return]


@final
class MultipolePotential(AbstractMultipolePotential):
    r"""Multipole inner and outer expansion potential.

    .. math::

        \Phi^l_\mathrm{max}(r,\theta,\phi) =
            \sum_{l=0}^{l=l_\mathrm{max}}\sum_{m=0}^{m=l}
            [  (r^l IS_{lm} + r^{-(l+1)} OS_{lm}) \, \cos{m\,\phi}
             + (r^l IT_{lm} + r^{-(l+1)} OT_{lm}) \, \sin{m\,\phi}]
            \, P_l^m(\cos\theta)

    """

    ISlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Inner spherical harmonic coefficients for the $\cos(m \phi)$ terms.",
    )

    ITlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Inner Spherical harmonic coefficients for the $\sin(m \phi)$ terms.",
    )

    OSlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Outer spherical harmonic coefficients for the $\cos(m \phi)$ terms.",
    )

    OTlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Outer Spherical harmonic coefficients for the $\sin(m \phi)$ terms.",
    )

    def __check_init__(self) -> None:
        shape = (self.l_max + 1, self.l_max + 1)
        t = u.Q(0.0, "Gyr")
        iss, its = self.ISlm(t).shape, self.ITlm(t).shape
        oss, ots = self.OSlm(t).shape, self.OTlm(t).shape
        # Check shapes match expected
        msg = "I/OSlm and I/OTlm must have the shape (l_max + 1, l_max + 1)."
        pred = (iss != shape) or (its != shape) or (oss != shape) or (ots != shape)
        _ = eqx.error_if(t, pred, msg)

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        # Compute parameters
        u1 = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        ISlm, ITlm = self.ISlm(t, ustrip=u1), self.ITlm(t, ustrip=u1)
        OSlm, OTlm = self.OSlm(t, ustrip=u1), self.OTlm(t, ustrip=u1)

        # scaled radius and unit direction
        is_scalar = xyz.ndim == 1
        s, uvec = scaled_radius_and_direction(jnp.atleast_2d(xyz), r_s)

        # Compute the summation over l and m
        terms = [
            jnp.pow(s, l) * (ISlm[l, m] * cYlm + ITlm[l, m] * sYlm)
            + jnp.pow(s, -l - 1) * (OSlm[l, m] * cYlm + OTlm[l, m] * sYlm)
            for l, m, cYlm, sYlm in iter_Ylm(self.l_max, uvec)
        ]
        summation = jnp.sum(jnp.stack(terms), axis=0)
        if is_scalar:
            summation = summation[0]

        _result = self.constants["G"].value * m_tot / r_s * summation
        return _result  # type: ignore[no-any-return]


# ===== Helper functions =====


def scaled_radius_and_direction(
    q: gt.BtSz3, r_s: gt.Sz0, /
) -> tuple[gt.BtFloatSz0, gt.BtSz3]:
    r"""Split Cartesian positions into :math:`r/r_s` and a unit direction.

    .. math::

        r = \sqrt{x^2 + y^2 + z^2}, \qquad \hat{q} = q / r

    The angular dependence is carried by the Cartesian unit vector rather than
    by :math:`(\theta, \phi)`: ``atan2(y, x)`` has gradient
    :math:`-y/(x^2+y^2)`, which is :math:`0/0` on the whole z-axis, so any
    :math:`m \ge 1` term built from it has NaN Cartesian derivatives there.

    ``r`` uses `safe_vector_norm`, which floors it at the square root of the
    smallest normal float (~1e-154). Without that, :math:`\hat{q} = q/r` is
    ``0/0`` at the origin and the value itself -- not just its derivatives --
    comes back NaN. The floor is far below any physical position, so nothing
    off the origin is perturbed.
    """
    r = safe_vector_norm(q)
    return r / r_s, q / r[..., None]


def reduced_legendre(
    l: int, m: int, u: Float[Array, "*batch"], /
) -> Float[Array, "*batch"]:
    r"""Evaluate :math:`N_{lm} p_l^m(u)`, with :math:`p_l^m = P_l^m/(1-u^2)^{m/2}`.

    :math:`N_{lm} = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}` is the
    spherical-harmonic normalization, so this returns the *normalized* reduced
    associated Legendre function. Includes the Condon-Shortley phase, matching
    `scipy.special.lpmv` and GSL. ``l`` and ``m`` are static, so the recurrence
    unrolls at trace time.

    The normalization is folded into the recurrence rather than applied
    afterwards, because :math:`p_l^m` alone is astronomically large -- its seed
    is :math:`(2m-1)!!`, which overflows float64 near :math:`m = 90` and would
    then make the whole harmonic ``nan`` via ``inf * 0`` on the axis, exactly
    the failure this module now exists to avoid. Multiplying by the tiny
    :math:`N_{lm}` afterwards also loses roughly two digits by cancellation at
    moderate :math:`m`. The normalized quantity is O(1) at every order: the
    seed is built in log space, and the recurrence carries it directly.
    """
    # N_mm p_m^m, in log space so p_m^m itself is never materialized.
    log_seed = (
        0.5 * math.log((2 * m + 1) / (4 * math.pi))
        + 0.5 * math.lgamma(2 * m + 1)
        - m * math.log(2)
        - math.lgamma(m + 1)
    )
    q_prev = jnp.zeros_like(u)
    q_cur = jnp.full_like(u, (-1.0) ** m * math.exp(log_seed))
    for ll in range(m + 1, l + 1):
        a = math.sqrt((4 * ll * ll - 1) / (ll * ll - m * m))
        b = (
            math.sqrt(((ll - 1) ** 2 - m * m) / (4 * (ll - 1) ** 2 - 1))
            if ll - 1 >= 1
            else 0.0
        )
        q_prev, q_cur = q_cur, a * (u * q_cur - b * q_prev)
    return q_cur  # type: ignore[no-any-return]


def compute_Ylm(
    l: int, m: int, uvec: gt.BtSz3, /
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
    r"""Compute the real and imaginary parts of :math:`Y_l^m`.

    Evaluate the harmonic directly from the Cartesian unit direction
    :math:`\hat{q} = (x, y, z)/r`, using

    .. math::

        \sin^m\theta \, e^{i m \phi} = \left(\frac{x + i y}{r}\right)^m
        \quad\Longrightarrow\quad
        Y_l^m = N_{lm} \, p_l^m(z/r) \, \left(\frac{x + i y}{r}\right)^m

    where `reduced_legendre` supplies :math:`N_{lm} p_l^m` as a single
    normalized quantity. The right-hand side is polynomial in :math:`x` and
    :math:`y`, so unlike the :math:`(\theta, \phi)` form it is smooth on the
    z-axis.
    """
    ux, uy, uz = uvec[..., 0], uvec[..., 1], uvec[..., 2]

    # ((x + i y) / r)^m by repeated multiplication (m is static).
    cos_mphi, sin_mphi = jnp.ones_like(ux), jnp.zeros_like(ux)
    for _ in range(m):
        cos_mphi, sin_mphi = (
            cos_mphi * ux - sin_mphi * uy,
            cos_mphi * uy + sin_mphi * ux,
        )

    plm = reduced_legendre(l, m, uz)  # already carries N_lm
    return plm * cos_mphi, plm * sin_mphi


def iter_Ylm(
    l_max: int, uvec: gt.BtSz3, /
) -> list[tuple[int, int, Float[Array, "*batch"], Float[Array, "*batch"]]]:
    """Compute ``(l, m, Re Y_lm, Im Y_lm)`` for every ``0 <= m <= l <= l_max``."""
    ls, ms = np.tril_indices(l_max + 1)
    return [
        (l, m, *compute_Ylm(l, m, uvec))
        for l, m in zip(ls.tolist(), ms.tolist(), strict=True)
    ]
