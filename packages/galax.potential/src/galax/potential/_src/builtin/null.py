"""Private module that defines `NullPotential` and its functions.

The public API is `galax.potential.null`.
"""

__all__ = [
    # class
    "NullPotential",
    # functions
    "d2potential_dr2",
    "density",
    "dpotential_dr",
    "gradient",
    "hessian",
    "laplacian",
    "mass_enclosed",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import Any, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.symmetry import Symmetry
from galax.potential._src.utils import r_spherical


@final
class NullPotential(AbstractSinglePotential):
    r"""Null potential, i.e. no potential.

    The potential, and everything derived from it, is zero everywhere:
    :math:`\Phi = 0` and :math:`\rho = 0`. The class is spherical, and the
    functions in `galax.potential.null` return zeros of the shape and dtype of
    their input.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> pot
    NullPotential(
      constants=ImmutableMap({'G': ...})
    )

    >>> q = u.Q([1, 0, 0], "kpc")
    >>> t = u.Q(0, "Gyr")
    >>> pot.potential(q, t)
    Q(0., 'kpc2 / Myr2')

    """

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        default=u.unitsystems.galactic, converter=u.unitsystem, static=True
    )
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    symmetry = Symmetry.SPHERICAL
    """`Symmetry.SPHERICAL`: the density depends only on $r$."""

    @ft.partial(jax.jit, inline=True)
    def _potential(self, q: gt.BBtQorVSz3, t: Any, /) -> gt.BBtSz0:
        r = r_spherical(q, self.units["length"])
        return potential(self.params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _gradient(self, q: gt.BBtQorVSz3, /, _: gt.BBtQorVSz0) -> gt.BBtSz3:
        """See ``gradient``."""
        _result = jnp.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
        return _result  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _laplacian(self, xyz: gt.BBtQorVSz3, /, _: gt.BBtQorVSz0) -> gt.BBtFloatSz0:
        """See ``laplacian``."""
        return jnp.zeros(xyz.shape[:-1], dtype=xyz.dtype)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        """See ``density``."""
        r = r_spherical(xyz, self.units["length"])
        return density(self.params(t), r)  # type: ignore[no-any-return]

    @ft.partial(jax.jit, inline=True)
    def _hessian(self, q: gt.BBtQorVSz3, _: gt.BBtQorVSz0, /) -> gt.BBtSz33:
        """See ``hessian``."""
        _result = jnp.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype)
        return _result  # type: ignore[no-any-return]


# ====================================================================
# Every function is zero. They take the same arguments as those of the other
# spherical modules, so that `galax.potential.null` can stand in for them.


@ft.partial(jax.jit)
def potential(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Potential of the null potential: exactly zero, with the shape of ``r``."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Density of the null potential: exactly zero, with the shape of ``r``."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def laplacian(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Laplacian of the null potential: exactly zero, with the shape of ``r``."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def mass_enclosed(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Enclosed mass of the null potential: exactly zero."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def dpotential_dr(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Radial derivative of the null potential: exactly zero."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def d2potential_dr2(p: gt.Params, r: gt.Sz0, /) -> gt.FloatSz0:  # noqa: ARG001
    r"""Second radial derivative of the null potential: exactly zero."""
    return jnp.zeros_like(r)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def gradient(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz3:  # noqa: ARG001
    r"""Gradient of the null potential: zeros, with the shape of ``xyz``."""
    return jnp.zeros_like(xyz)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def hessian(p: gt.Params, xyz: gt.BBtSz3, /) -> gt.BBtSz33:  # noqa: ARG001
    r"""Hessian of the null potential: zeros, shape ``(*xyz.shape, 3)``."""
    return jnp.zeros((*xyz.shape, 3), dtype=xyz.dtype)  # type: ignore[no-any-return]
