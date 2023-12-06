from __future__ import annotations

__all__ = ["AbstractPotentialBase", "AbstractPotential"]

import abc
import uuid
from dataclasses import KW_ONLY, fields
from typing import TYPE_CHECKING, Any

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as xp
import jax.typing as jt
from astropy.constants import G as _G

from galdynamix.integrate._builtin import DiffraxIntegrator
from galdynamix.potential._potential.param.field import ParameterField
from galdynamix.units import UnitSystem, dimensionless
from galdynamix.utils import partial_jit

if TYPE_CHECKING:
    from galdynamix.integrate._base import AbstractIntegrator
    from galdynamix.potential._potential.composite import CompositePotential


class AbstractPotentialBase(eqx.Module):  # type: ignore[misc]
    """Potential Class."""

    units: eqx.AbstractVar[UnitSystem]

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    def potential_energy(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        E : :class:`~jax.Array`
            The potential energy per unit mass or value of the potential.
        """
        raise NotImplementedError

    ###########################################################################
    # Parsing

    def _init_units(self) -> None:
        G = 1 if self.units == dimensionless else _G.decompose(self.units).value
        object.__setattr__(self, "_G", G)

        # Handle unit conversion for all fields, e.g. the parameters.
        for f in fields(self):
            # Process ParameterFields
            param = getattr(self.__class__, f.name, None)
            if isinstance(param, ParameterField):
                # Set, since the ``.units`` are now known
                param.__set__(self, getattr(self, f.name))

            # Other fields, check their metadata
            elif "dimensions" in f.metadata:
                value = getattr(self, f.name)
                if isinstance(value, u.Quantity):
                    value = value.to_value(
                        self.units[f.metadata.get("dimensions")],
                        equivalencies=f.metadata.get("equivalencies", None),
                    )
                    object.__setattr__(self, f.name, value)

    ###########################################################################
    # Core methods that use the above implemented functions

    @partial_jit()
    def __call__(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the potential energy at the given position(s).

        See Also
        --------
        potential_energy
        """
        return self.potential_energy(q, t)

    @partial_jit()
    def gradient(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The gradient of the potential.
        """
        return jax.grad(self.potential_energy)(q, t)

    @partial_jit()
    def density(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the density value at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The potential energy or value of the potential.
        """
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        lap = xp.trace(jax.jacfwd(self.gradient)(q, t))
        return lap / (4 * xp.pi * self._G)

    @partial_jit()
    def hessian(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : :class:`~jax.Array`
            The time at which to compute the value of the potential.

        Returns
        -------
        :class:`~jax.Array`
            The Hessian matrix of second derivatives of the potential.
        """
        return jax.hessian(self.potential_energy)(q, t)

    ###########################################################################
    # Convenience methods

    @partial_jit()
    def acceleration(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : :class:`~jax.Array`
            Position to compute the acceleration at.
        t : :class:`~jax.Array`
            Time at which to compute the acceleration.

        Returns
        -------
        :class:`~jax.Array`
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t)

    # =========================================================================

    @partial_jit()
    def _integrator_F(
        self, t: jt.Array, qp: jt.Array, args: tuple[Any, ...]
    ) -> jt.Array:
        return xp.hstack([qp[3:], self.acceleration(qp[:3], t)])

    @partial_jit(static_argnames=("Integrator", "integrator_kw"))
    def integrate_orbit(
        self,
        w0: jt.Array,
        t0: jt.Array,
        t1: jt.Array,
        ts: jt.Array | None,
        *,
        Integrator: type[AbstractIntegrator] = DiffraxIntegrator,
        integrator_kw: dict[str, Any] | None = None,
    ) -> jt.Array:
        from galdynamix.dynamics._orbit import Orbit

        integrator = Integrator(self._integrator_F, **(integrator_kw or {}))
        ws = integrator.run(w0, t0, t1, ts)
        return Orbit(q=ws[:, :3], p=ws[:, 3:-1], t=ws[:, -1], potential=self)

    ###########################################################################
    # Composite potentials

    def __add__(self, other: Any) -> CompositePotential:
        if not isinstance(other, AbstractPotentialBase):
            return NotImplemented

        from galdynamix.potential._potential.composite import CompositePotential

        if isinstance(other, CompositePotential):
            return other.__ror__(self)

        return CompositePotential({str(uuid.uuid4()): self, str(uuid.uuid4()): other})


# ===========================================================================


class AbstractPotential(AbstractPotentialBase):
    _: KW_ONLY
    units: UnitSystem = eqx.field(
        default=None,
        converter=lambda x: dimensionless if x is None else UnitSystem(x),
        static=True,
    )
    _G: float = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        self._init_units()
