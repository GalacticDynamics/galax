"""Private module that defines `MultipoleInnerPotential` and its functions.

The public API is `galax.potential.multipoleinner`. The base class, the zero
``density`` and `scaled_radius_and_direction` are shared with the other
multipole modules and live in `galax.potential._src.builtin.multipole`.
"""

__all__ = [
    # class
    "MultipoleInnerPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft

from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .multipole import (
    AbstractMultipolePotential,
    density,
    laplacian,
    solid_harmonic_sum,
)
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class MultipoleInnerPotential(AbstractMultipolePotential):
    r"""Multipole inner expansion potential.

    The potential of a source-free expansion in solid harmonics that grow with
    radius, as for the field inside a cavity. With :math:`s = r / r_s` and the
    orthonormal spherical harmonics :math:`Y_l^m(\theta, \phi)`, it is

    .. math::

        \Phi(r,\theta,\phi) = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            s^l \left( S_{lm}\,\mathrm{Re}\,Y_l^m
                     + T_{lm}\,\mathrm{Im}\,Y_l^m \right),

    where :math:`\mathrm{Re}\,Y_l^m \propto P_l^m(\cos\theta)\cos m\phi` and
    :math:`\mathrm{Im}\,Y_l^m \propto P_l^m(\cos\theta)\sin m\phi`. Here
    :math:`m_{\mathrm{tot}}` is a mass, :math:`r_s` a length and :math:`G` the
    gravitational constant, so :math:`G m_{\mathrm{tot}} / r_s` has the units
    of a potential and the tables ``Slm`` and ``Tlm`` are dimensionless. Each
    is an array of shape ``(l_max + 1, l_max + 1)``, and only the entries with
    :math:`m \le l` are used. The functions in
    `galax.potential.multipoleinner` take :math:`l_\mathrm{max}` from that
    shape. The harmonics are the orthonormal ones with the Condon-Shortley
    phase, the same as ``mpmath.spherharm`` and ``scipy.special.sph_harm_y``
    [BT08MultipoleInner]_.

    Every term is a solid harmonic, so :math:`\nabla^2 \Phi = 0` for all
    :math:`r` and the density is exactly zero. The :math:`l = 0` term is the
    constant :math:`G m_{\mathrm{tot}} S_{00} / (r_s \sqrt{4\pi})`, and the
    :math:`l = 1` terms are linear in :math:`x, y, z`, a uniform field.

    Notes
    -----
    The potential is a polynomial in :math:`x, y, z` of degree
    :math:`l_\mathrm{max}`, so its value is smooth everywhere, including the
    origin and the :math:`z` axis. The gradient and the Hessian are by automatic
    differentiation, and they are exact and finite on the :math:`z` axis. At the
    exact origin they are not the limits of the polynomial: the direction
    :math:`\vec{q}/r` is floored there, so with an :math:`l = 1` term the
    gradient is about zero instead of the uniform field, and the Hessian loses
    accuracy as :math:`r \to 0`. Evaluate them away from the origin. It grows
    without bound at large radius, so it models the field in a region and not an
    isolated system.

    For the field outside the sources use
    :class:`~galax.potential.MultipoleOuterPotential`, and for both together
    :class:`~galax.potential.MultipolePotential`.

    References
    ----------
    .. [BT08MultipoleInner] Binney, J., & Tremaine, S. (2008). Galactic
       Dynamics, 2nd ed. Princeton University Press, Sec. 2.4.
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

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Potential of the inner multipole expansion.

    .. math::

        \Phi = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            s^l \left( S_{lm}\,\mathrm{Re}\,Y_l^m
                     + T_{lm}\,\mathrm{Im}\,Y_l^m \right)

    with :math:`s = r / r_s`. The maximum degree is read from the coefficient
    shape, :math:`l_\mathrm{max} =` ``p["Slm"].shape[0] - 1``. The function
    works on one point of shape ``(3,)`` and also on a leading batch
    ``(..., 3)``, so the class calls it directly on a batch of points.
    """
    return solid_harmonic_sum(p, xyz, (p["Slm"], p["Tlm"], lambda l: l))


# `density` is the exact zero shared by the multipole modules (see
# `galax.potential._src.builtin.multipole.density`), imported above.
# Derived quantities, built from the kernels above.
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
