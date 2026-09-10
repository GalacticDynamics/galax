"""Multipole potential."""

__all__ = [
    "AbstractMultipolePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
]

import functools as ft
from dataclasses import KW_ONLY

from jaxtyping import Array, Float
from typing import final

import equinox as eqx
import jax
from equinox import field

import quaxed.numpy as jnp
import unxt as u
from spexial import sph_harm_y_cart_all_terms
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

    The scaled radius uses `safe_vector_norm`, which floors ``r`` at
    ``sqrt(finfo(dtype).tiny)`` -- around 1e-154 in float64. Without that,
    :math:`\hat{q} = q/r` is ``0/0`` at the origin and the value itself, not
    just its derivatives, comes back NaN.

    The *direction* needs a larger floor than the radius does. The second
    derivative of :math:`q/r` carries a :math:`1/r^3` term, and at
    :math:`r \sim 10^{-154}` that cubes straight past the bottom of float64
    and evaluates to ``inf``, so the hessian is NaN even though the value and
    the gradient are finite. Flooring the direction's denominator at
    ``tiny**(1/3)`` keeps :math:`r^3` representable.

    That floor does perturb the direction for sufficiently small non-zero
    :math:`|q|`: measured in float64, ``q / r`` is bit-identical with and
    without it down to :math:`|q| \sim 10^{-90}` and starts to differ by
    :math:`10^{-95}`. Both floors therefore sit some eighty orders of
    magnitude below any physical position, but "unchanged everywhere except
    the origin" would be too strong a claim. The returned ``s`` is unaffected
    either way, since it is built from the ``r`` that is not floored for the
    direction.
    """
    r = safe_vector_norm(q)
    # `tiny**(2/3)` inside the square root floors `r` itself at `tiny**(1/3)`.
    cube_safe = jnp.finfo(jnp.promote_types(q.dtype, float)).tiny ** (2 / 3)
    r_dir = jnp.sqrt(jnp.sum(jnp.square(q), axis=-1) + cube_safe)
    return r / r_s, q / r_dir[..., None]


def iter_Ylm(
    l_max: int, uvec: gt.BtSz3, /
) -> list[tuple[int, int, Float[Array, "*batch"], Float[Array, "*batch"]]]:
    r"""Give ``(l, m, Re Y_lm, Im Y_lm)`` for every ``0 <= m <= l <= l_max``.

    A thin adapter over `spexial.sph_harm_y_cart_all_terms`, which runs the
    Legendre and azimuth recurrences *once* across the whole table -- one pass
    per ``m``, advancing :math:`((x+iy)/r)^m` by a single complex multiply and
    walking the Legendre recurrence up in ``l`` from its seed -- and evaluates
    each harmonic from the Cartesian unit direction as :math:`N_{lm}
    p_l^m(z/r) ((x+iy)/r)^m`. That form is polynomial in :math:`x` and
    :math:`y`, and so smooth on the z-axis, where :math:`\theta` and
    :math:`\phi` are singular and the chain rule sends the gradient of every
    :math:`m \ge 1` term to exactly zero.

    Why `spexial` rather than `jax.scipy.special.sph_harm_y`: upstream pairs
    ``l[i]`` with ``theta[i]`` instead of broadcasting, so it returns silently
    wrong values for a batch of positions, and its derivatives are ``nan`` at
    both poles. Both are documented in `spexial`, with regression tests
    asserting the defects directly.

    Two things are adapted, and only two. `spexial` returns one complex array
    per term, while every consumer here wants the real and imaginary parts
    separately, since :math:`S_{lm}` and :math:`T_{lm}` are real and multiply
    them independently. And `spexial`'s inner axis runs over
    :math:`-l_{max} \ldots l_{max}`, following SciPy, while the real
    expansions used here need only :math:`m \ge 0`.

    Note the ``_terms`` spelling: `spexial.sph_harm_y_cart_all` computes the
    same values but returns them *stacked* into one array, and indexing a
    stacked table is what stops XLA folding each term into the caller's
    summation as it is produced -- the whole table is materialized instead.
    Measured on `MultipoleInnerPotential` at ``l_max = 12`` over a million
    positions, the stacked form ran in 17.7 s against 10 ms here, for identical
    values. Every consumer of this function immediately sums the terms.

    Terms come out in m-major order rather than the l-major order of
    ``np.tril_indices``; every caller sums them, so the order is immaterial.
    """
    terms = sph_harm_y_cart_all_terms(l_max, l_max, uvec)
    return [
        (l, m, terms[l][m].real, terms[l][m].imag)
        for m in range(l_max + 1)
        for l in range(m, l_max + 1)
    ]
