"""Private module that defines `MultipoleOuterPotential` and its functions.

The public API is `galax.potential.multipoleouter`.
"""

__all__ = [
    # classes
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
from galax.potential._src.builtin.multipole import (
    AbstractMultipolePotential,
    density,
    scaled_radius_and_direction,
)
from galax.potential._src.harmonic.ylm import iter_Ylm
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


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

        # Compute
        params = self.params(t)
        return potential(params, xyz)  # type: ignore[no-any-return]


# ====================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtFloatSz0:
    r"""Potential of the multipole outer expansion.

    .. math::

        \Phi = \frac{G m_\mathrm{tot}}{r_s} \sum_{l, m} s^{-(l+1)}
            \left(S_{lm}\, \mathrm{Re}\,Y_{lm}
                  + T_{lm}\, \mathrm{Im}\,Y_{lm}\right)

    with :math:`s = r / r_s`. The sum runs over the real and imaginary parts of the
    spherical harmonics :math:`Y_{lm}`, evaluated from the Cartesian direction, for
    :math:`0 \le m \le l \le l_\mathrm{max}`. The maximum degree
    :math:`l_\mathrm{max}` is read from the shape of the coefficients, which is
    ``(l_max + 1, l_max + 1)``, so a kernel needs no ``l_max`` argument.
    """
    l_max = p["Slm"].shape[0] - 1

    # scaled radius and unit direction
    is_scalar = xyz.ndim == 1
    s, uvec = scaled_radius_and_direction(jnp.atleast_2d(xyz), p["r_s"])

    # Compute the summation over l and m
    terms = [
        jnp.pow(s, -(l + 1)) * (p["Slm"][l, m] * cYlm + p["Tlm"][l, m] * sYlm)
        for l, m, cYlm, sYlm in iter_Ylm(l_max, uvec)
    ]
    summation = jnp.sum(jnp.stack(terms), axis=0)
    if is_scalar:
        summation = summation[0]

    return p["G"] * p["m_tot"] / p["r_s"] * summation  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
