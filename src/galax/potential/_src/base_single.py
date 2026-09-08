__all__ = ["AbstractSinglePotential", "LaplacianFromDensityMixin"]

import functools as ft
import uuid
from dataclasses import KW_ONLY

from typing import Any

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .base import AbstractPotential, default_constants
from .composite import CompositePotential


class AbstractSinglePotential(AbstractPotential):
    """Abstract base class for all potential objects."""

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def __post_init__(self) -> None:
        self._apply_unitsystem()

    ###########################################################################

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotential):
            return NotImplemented

        # CompositePotential has better methods for combining potentials
        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})


class LaplacianFromDensityMixin(AbstractSinglePotential):
    """Mixin for potentials with a closed-form ``_density``.

    Provides ``_laplacian`` via Poisson's equation, ``laplacian(Phi) =
    4 pi G density``, which is exact and much cheaper than the default
    ``jax.hessian(potential)`` + trace. Mix in alongside
    ``AbstractSinglePotential`` on any potential that already overrides
    ``_density`` with a closed-form expression.
    """

    @ft.partial(jax.jit)
    def _laplacian(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtFloatSz0:
        return 4 * jnp.pi * self.constants["G"].value * self._density(xyz, t)  # type: ignore[no-any-return]
