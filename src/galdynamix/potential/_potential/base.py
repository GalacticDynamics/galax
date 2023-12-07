__all__ = ["AbstractPotentialBase"]

import abc
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import astropy.units as u
import equinox as eqx
import jax.numpy as xp
from astropy.constants import G as _G
from jax import grad, hessian, jacfwd
from jaxtyping import Array, Float

from galdynamix.integrate._base import AbstractIntegrator
from galdynamix.integrate._builtin import DiffraxIntegrator
from galdynamix.typing import (
    BatchableFloatLike,
    BatchFloatScalar,
    BatchVec3,
    FloatScalar,
    Vec3,
    Vec6,
)
from galdynamix.units import UnitSystem, dimensionless
from galdynamix.utils import partial_jit, vectorize_method

if TYPE_CHECKING:
    from galdynamix.dynamics._orbit import Orbit


class AbstractPotentialBase(eqx.Module):  # type: ignore[misc]
    """Potential Class."""

    units: eqx.AbstractVar[UnitSystem]

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    # @partial_jit()
    # @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: FloatScalar) -> FloatScalar:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractPotentialBase.potential_energy`` for an example.
        """
        raise NotImplementedError

    ###########################################################################
    # Parsing

    def _init_units(self) -> None:
        G = 1 if self.units == dimensionless else _G.decompose(self.units).value
        object.__setattr__(self, "_G", G)

        from galdynamix.potential._potential.param.field import ParameterField

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
    # Core methods that use the potential energy

    # ---------------------------------------
    # Potential energy

    def potential_energy(
        self, q: BatchVec3, /, t: BatchableFloatLike
    ) -> BatchFloatScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential.
        t : float | Array[float, *batch]
            The time at which to compute the value of the potential.

        Returns
        -------
        E : Array[float, *batch]
            The potential energy per unit mass or value of the potential.
        """
        return self._potential_energy(q, t)

    @partial_jit()
    def __call__(self, q: BatchVec3, /, t: BatchableFloatLike) -> BatchFloatScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential.
        t : float | Array[float, *batch]
            The time at which to compute the value of the potential.

        Returns
        -------
        E : Array[float, *batch]
            The potential energy per unit mass or value of the potential.

        See Also
        --------
        potential_energy
        """
        return self.potential_energy(q, t)

    # ---------------------------------------
    # Gradient

    @partial_jit()
    @vectorize_method(signature="(3),()->(3)")
    def _gradient(self, q: Vec3, /, t: FloatScalar) -> Vec3:
        """See ``gradient``."""
        return grad(self.potential_energy)(q, t)

    def gradient(self, q: BatchVec3, /, t: BatchableFloatLike) -> BatchVec3:
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : float | Array[float, *batch]
            The time at which to compute the value of the potential.

        Returns
        -------
        grad : Array[float, (*batch, 3)]
            The gradient of the potential.
        """
        return self._gradient(q, t)  # vectorize doesn't allow kwargs

    # ---------------------------------------
    # Density

    @partial_jit()
    @vectorize_method(signature="(3),()->()")
    def _density(self, q: Vec3, /, t: FloatScalar) -> FloatScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        lap = xp.trace(jacfwd(self.gradient)(q, t))
        return lap / (4 * xp.pi * self._G)

    def density(self, q: BatchVec3, /, t: BatchableFloatLike) -> BatchFloatScalar:
        """Compute the density value at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : float | Array[float, *batch]
            The time at which to compute the value of the potential.

        Returns
        -------
        rho : Array[float, *batch]
            The potential energy or value of the potential.
        """
        return self._density(q, t)

    # ---------------------------------------
    # Hessian

    @partial_jit()
    @vectorize_method(signature="(3),()->(3,3)")
    def _hessian(self, q: Vec3, /, t: FloatScalar) -> Float[Array, "3 3"]:
        """See ``hessian``."""
        return hessian(self.potential_energy)(q, t)

    def hessian(
        self, q: BatchVec3, /, t: BatchableFloatLike
    ) -> Float[Array, "*batch 3 3"]:
        """Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : float | Array[float, *batch]
            The time at which to compute the value of the potential.

        Returns
        -------
        Array[float, (*batch, 3, 3)]
            The Hessian matrix of second derivatives of the potential.
        """
        return self._hessian(q, t)

    ###########################################################################
    # Convenience methods

    @partial_jit()
    @vectorize_method(signature="(3),()->(3)")
    def _acceleration(self, q: Vec3, /, t: FloatScalar) -> Vec3:
        """See ``acceleration``."""
        return -self.gradient(q, t)

    def acceleration(self, q: BatchVec3, /, t: BatchableFloatLike) -> BatchVec3:
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            Position to compute the acceleration at.
        t : float | Array[float, *batch]
            Time at which to compute the acceleration.

        Returns
        -------
        Array[float, (*batch, 3)]
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t)

    # =========================================================================
    # Integrating orbits

    @partial_jit()
    def _integrator_F(self, t: FloatScalar, xv: Vec6, args: tuple[Any, ...]) -> Vec6:
        return xp.hstack([xv[3:6], self.acceleration(xv[:3], t)])

    @partial_jit(static_argnames=("Integrator", "integrator_kw"))
    def integrate_orbit(
        self,
        qp0: Vec6,
        t0: FloatScalar,
        t1: FloatScalar,
        ts: Float[Array, "time"] | None,
        *,
        Integrator: type[AbstractIntegrator] = DiffraxIntegrator,
        integrator_kw: dict[str, Any] | None = None,
    ) -> "Orbit":
        from galdynamix.dynamics._orbit import Orbit

        integrator = Integrator(self._integrator_F, **(integrator_kw or {}))
        ws = integrator.run(qp0, t0, t1, ts)
        return Orbit(q=ws[:, :3], p=ws[:, 3:-1], t=ws[:, -1], potential=self)
