"""Private module that defines `MultipoleOuterPotential` and its functions.

The public API is `galax.potential.multipoleouter`. The base class, the zero
``density`` and `scaled_radius_and_direction` are shared with the other
multipole modules and live in `galax.potential._src.builtin.multipole`.
"""

__all__ = [
    # class
    "MultipoleOuterPotential",
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
class MultipoleOuterPotential(AbstractMultipolePotential):
    r"""Multipole outer expansion potential.

    The potential of a source-free expansion in solid harmonics that fall with
    radius, as for the field outside a bounded source. With
    :math:`s = r / r_s` and the orthonormal spherical harmonics
    :math:`Y_l^m(\theta, \phi)`, it is

    .. math::

        \Phi(r,\theta,\phi) = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            s^{-(l+1)} \left( S_{lm}\,\mathrm{Re}\,Y_l^m
                     + T_{lm}\,\mathrm{Im}\,Y_l^m \right),

    where :math:`\mathrm{Re}\,Y_l^m \propto P_l^m(\cos\theta)\cos m\phi` and
    :math:`\mathrm{Im}\,Y_l^m \propto P_l^m(\cos\theta)\sin m\phi`. Here
    :math:`m_{\mathrm{tot}}` is a mass, :math:`r_s` a length and :math:`G` the
    gravitational constant, so :math:`G m_{\mathrm{tot}} / r_s` has the units
    of a potential and the tables ``Slm`` and ``Tlm`` are dimensionless. Each
    is an array of shape ``(l_max + 1, l_max + 1)``, and only the entries with
    :math:`m \le l` are used. The functions in
    `galax.potential.multipoleouter` take :math:`l_\mathrm{max}` from that
    shape. The harmonics are the orthonormal ones with the Condon-Shortley
    phase, the same as ``mpmath.spherharm`` and ``scipy.special.sph_harm_y``
    [BT08MultipoleOuter]_.

    Every term is a solid harmonic, so :math:`\nabla^2 \Phi = 0` for
    :math:`r > 0` and the density is exactly zero there. The :math:`l = 0` term
    is :math:`G m_{\mathrm{tot}} S_{00} / (\sqrt{4\pi}\, r)`, the potential of
    a point mass :math:`-m_{\mathrm{tot}} S_{00} / \sqrt{4\pi}` at the origin,
    and the :math:`l = 1` terms are dipoles.

    Notes
    -----
    Every term falls to zero at large radius, so the potential is that of an
    isolated source. It diverges at the origin whenever a coefficient is
    non-zero, because :math:`s^{-(l+1)} \to \infty` there; evaluate it away
    from :math:`r = 0`. The gradient and Hessian are finite on the :math:`z`
    axis. For the field inside the sources use
    :class:`~galax.potential.MultipoleInnerPotential`, and for both together
    :class:`~galax.potential.MultipolePotential`.

    References
    ----------
    .. [BT08MultipoleOuter] Binney, J., & Tremaine, S. (2008). Galactic
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
    r"""Potential of the outer multipole expansion.

    .. math::

        \Phi = \frac{G m_{\mathrm{tot}}}{r_s}
            \sum_{l=0}^{l_\mathrm{max}} \sum_{m=0}^{l}
            s^{-(l+1)} \left( S_{lm}\,\mathrm{Re}\,Y_l^m
                     + T_{lm}\,\mathrm{Im}\,Y_l^m \right)

    with :math:`s = r / r_s`. The maximum degree is read from the coefficient
    shape, :math:`l_\mathrm{max} =` ``p["Slm"].shape[0] - 1``. The function
    works on one point of shape ``(3,)`` and also on a leading batch
    ``(..., 3)``, so the class calls it directly on a batch of points.
    """
    return solid_harmonic_sum(p, xyz, (p["Slm"], p["Tlm"], lambda l: -l - 1))


# Derived quantities, built from the kernels above.
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
