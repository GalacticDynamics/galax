"""Private module that defines `MultipolePotential`, its base class and its functions.

The public API is `galax.potential.multipole`. The inner and the outer
expansions are in `multipoleinner` and `multipoleouter`; they share the base
class, the zero `density` and `scaled_radius_and_direction` defined here.
"""

__all__ = [
    # classes
    "AbstractMultipolePotential",
    "MultipolePotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
    "scaled_radius_and_direction",
]  # `expansion_potential` is shared with the inner and outer modules, not public

import functools as ft
from dataclasses import KW_ONLY

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
from equinox import field

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.harmonic.ylm import iter_Ylm
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
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

    The maximum degree ``l_max`` is a static field. The coefficient tables have
    shape ``(l_max + 1, l_max + 1)``, and the module functions read
    :math:`l_\mathrm{max}` back from that shape, so the two cannot disagree.
    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Total mass."
    )

    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    _: KW_ONLY
    l_max: int = field(static=True)

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, _: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        # The density is zero and reads only the shape and dtype of `xyz`.
        return density({}, xyz)  # type: ignore[no-any-return]


@final
class MultipolePotential(AbstractMultipolePotential):
    r"""Multipole inner and outer expansion potential.

    The potential of a source-free expansion in solid harmonics, with a
    regular part that grows with radius and a singular part that falls with
    radius. With :math:`s = r / r_s` and the orthonormal spherical harmonics
    :math:`Y_l^m(\theta, \phi)`, it is

    .. math::

        \Phi(r,\theta,\phi) = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            \Big[ \big(s^l\, IS_{lm} + s^{-(l+1)}\, OS_{lm}\big)
                  \,\mathrm{Re}\,Y_l^m
                + \big(s^l\, IT_{lm} + s^{-(l+1)}\, OT_{lm}\big)
                  \,\mathrm{Im}\,Y_l^m \Big],

    where :math:`\mathrm{Re}\,Y_l^m \propto P_l^m(\cos\theta)\cos m\phi` and
    :math:`\mathrm{Im}\,Y_l^m \propto P_l^m(\cos\theta)\sin m\phi`. Here
    :math:`m_{\mathrm{tot}}` is a mass, :math:`r_s` a length and :math:`G` the
    gravitational constant, so :math:`G m_{\mathrm{tot}} / r_s` has the units
    of a potential and the four coefficient tables ``ISlm``, ``ITlm``,
    ``OSlm`` and ``OTlm`` are dimensionless. Each is an array of shape
    ``(l_max + 1, l_max + 1)``; only the entries with :math:`m \le l` are used.
    The functions in `galax.potential.multipole` take :math:`l_\mathrm{max}` from
    that shape. The harmonics are the orthonormal ones with the
    Condon-Shortley phase, the same as ``mpmath.spherharm`` and
    ``scipy.special.sph_harm_y`` [BT08Multipole]_.

    Every term is a solid harmonic, so :math:`\nabla^2 \Phi = 0` for
    :math:`r > 0` and the density is exactly zero there. The mass of the
    model sits formally at the origin (the outer terms) and at infinity (the
    inner terms).

    Notes
    -----
    The potential is not defined at the origin when any outer coefficient is
    non-zero, because :math:`s^{-(l+1)}` diverges there. With only inner terms
    it is a polynomial in :math:`x, y, z`.

    The direction is carried by the Cartesian unit vector rather than by
    :math:`(\theta, \phi)`, so the gradient and Hessian are finite on the
    :math:`z` axis. The ``l_max`` argument of the class is static, and the
    cost grows as :math:`l_\mathrm{max}^2`.

    References
    ----------
    .. [BT08Multipole] Binney, J., & Tremaine, S. (2008). Galactic Dynamics,
       2nd ed. Princeton University Press, Sec. 2.4.
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

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:  # noqa: ARG001
    r"""Density of a multipole expansion: exactly zero.

    Every term is a solid harmonic, :math:`\nabla^2 \Phi = 0` for :math:`r > 0`,
    so there is no mass away from the origin. The same function is the
    ``density`` of `galax.potential.multipoleinner`,
    `galax.potential.multipoleouter` and `galax.potential.multipole`.
    """
    return jnp.zeros(xyz.shape[:-1], dtype=xyz.dtype)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Potential of the combined inner and outer multipole expansion.

    .. math::

        \Phi = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            \Big[ \big(s^l\, IS_{lm} + s^{-(l+1)}\, OS_{lm}\big)
                  \,\mathrm{Re}\,Y_l^m
                + \big(s^l\, IT_{lm} + s^{-(l+1)}\, OT_{lm}\big)
                  \,\mathrm{Im}\,Y_l^m \Big]

    with :math:`s = r / r_s`. The maximum degree is read from the coefficient
    shape, :math:`l_\mathrm{max} =` ``p["ISlm"].shape[0] - 1``. The function
    works on one point of shape ``(3,)`` and also on a leading batch
    ``(..., 3)``, so the class calls it directly on a batch of points.
    """
    return expansion_potential(
        p,
        xyz,
        (p["ISlm"], p["ITlm"], lambda l: l),
        (p["OSlm"], p["OTlm"], lambda l: -l - 1),
    )


def expansion_potential(
    p: gt.Params,
    xyz: gt.BBtSz3,
    /,
    *parts: tuple[jax.Array, jax.Array, Callable[[int], int]],
) -> gt.BBtFloatSz0:
    r"""Sum of ``s**e(l) * (S_lm Re Y_lm + T_lm Im Y_lm)`` over the ``parts``.

    Each part is a table ``S``, a table ``T`` and the exponent ``e(l)`` of
    :math:`s = r / r_s`. The maximum degree is the shape of the first table
    minus one. The prefactor :math:`G m_{\mathrm{tot}} / r_s` is included.
    """
    l_max = parts[0][0].shape[0] - 1
    # Leading axis so that a single point and a batch take the same path.
    s, uvec = scaled_radius_and_direction(xyz[None], p["r_s"])
    terms = [
        sum(jnp.pow(s, e(l)) * (S[l, m] * cYlm + T[l, m] * sYlm) for S, T, e in parts)
        for l, m, cYlm, sYlm in iter_Ylm(l_max, uvec)
    ]
    summation = jnp.sum(jnp.stack(terms), axis=0)[0]
    return p["G"] * p["m_tot"] / p["r_s"] * summation  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)


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
