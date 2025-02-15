"""Null potential."""

__all__ = ["NullPotential"]

from dataclasses import KW_ONLY
from functools import partial
from typing import Any, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.unitsystems import AbstractUnitSystem, galactic
from xmmutablemap import ImmutableMap

import galax.typing as gt
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential


@final
class NullPotential(AbstractSinglePotential):
    """Null potential, i.e. no potential.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NullPotential()
    >>> pot
    NullPotential( units=..., constants=ImmutableMap({'G': ...}) )

    >>> q = u.Quantity([1, 0, 0], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(0, dtype=int64), unit='kpc2 / Myr2')

    """

    _: KW_ONLY
    units: AbstractUnitSystem = eqx.field(
        default=galactic, converter=u.unitsystem, static=True
    )
    constants: ImmutableMap[str, u.Quantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @partial(jax.jit)
    def _potential(self, q: gt.BtQuSz3 | gt.BtSz3, _: Any, /) -> gt.BtSz0:
        return jnp.zeros(q.shape[:-1], dtype=q.dtype)

    @partial(jax.jit, inline=True)
    def _gradient(
        self, q: gt.BtQuSz3 | gt.BtSz3, /, _: gt.RealQuSz0 | gt.RealSz0
    ) -> gt.BtSz3:
        """See ``gradient``."""
        return jnp.zeros(q.shape[:-1] + (3,), dtype=q.dtype)

    @partial(jax.jit, inline=True)
    def _laplacian(self, q: gt.QuSz3, /, _: gt.RealQuSz0) -> gt.FloatQuSz0:
        """See ``laplacian``."""
        return u.Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1], dtype=q.dtype), galactic["frequency drift"]
        )

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BtQuSz3, _: gt.BtRealQuSz0 | gt.RealQuSz0, /
    ) -> gt.BtFloatQuSz0:
        """See ``density``."""
        return u.Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1], dtype=q.dtype), galactic["mass density"]
        )

    @partial(jax.jit, inline=True)
    def _hessian(self, q: gt.QuSz3, _: gt.RealQuSz0, /) -> gt.QuSz33:
        """See ``hessian``."""
        return u.Quantity(  # TODO: better unit handling
            jnp.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype), galactic["frequency drift"]
        )
