__all__ = ["AbstractPotentialBase"]

import abc
from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import astropy.units as u
import equinox as eqx
import jax.numpy as xp
from astropy.constants import G as _G
from jax import grad, hessian, jacfwd
from jaxtyping import Array, Float

from galax.integrate._base import AbstractIntegrator
from galax.integrate._builtin import DiffraxIntegrator
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatScalar,
    BatchMatrix33,
    BatchVec3,
    FloatOrIntScalar,
    FloatScalar,
    Matrix33,
    Vec3,
    Vec6,
)
from galax.units import UnitSystem, dimensionless
from galax.utils import partial_jit, vectorize_method
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    from galax.dynamics._orbit import Orbit


class AbstractPotentialBase(eqx.Module, metaclass=ModuleMeta):  # type: ignore[misc]
    """Potential Class."""

    units: eqx.AbstractVar[UnitSystem]

    ###########################################################################
    # Abstract methods that must be implemented by subclasses

    @abc.abstractmethod
    # @partial_jit()
    # @vectorize_method(signature="(3),()->()")
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
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

        from galax.potential._potential.param.field import ParameterField

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
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential.
        t : Array[float | int, *batch] | float | int
            The time at which to compute the value of the potential.

        Returns
        -------
        E : Array[float, *batch]
            The potential energy per unit mass or value of the potential.
        """
        return self._potential_energy(q, t)

    @partial_jit()
    def __call__(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential.
        t : Array[float | int, *batch] | float | int
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
    def _gradient(self, q: Vec3, /, t: FloatOrIntScalar) -> Vec3:
        """See ``gradient``."""
        return grad(self._potential_energy)(q, t)

    def gradient(self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike) -> BatchVec3:
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : Array[float | int, *batch] | float | int
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
    def _density(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        lap = xp.trace(jacfwd(self.gradient)(q, t))
        return lap / (4 * xp.pi * self._G)

    def density(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchFloatScalar:
        """Compute the density value at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : Array[float | int, *batch] | float | int
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
    def _hessian(self, q: Vec3, /, t: FloatOrIntScalar) -> Matrix33:
        """See ``hessian``."""
        return hessian(self._potential_energy)(q, t)

    def hessian(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchMatrix33:
        """Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : Array[float | int, *batch] | float | int
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

    def acceleration(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchVec3:
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            Position to compute the acceleration at.
        t : Array[float | int, *batch] | float | int
            Time at which to compute the acceleration.

        Returns
        -------
        Array[float, (*batch, 3)]
            The acceleration. Will have the same shape as the input
            position array, ``q``.
        """
        return -self.gradient(q, t)

    @partial_jit()
    def tidal_tensor(
        self, q: BatchVec3, /, t: BatchableFloatOrIntScalarLike
    ) -> BatchMatrix33:
        """Compute the tidal tensor.

        See https://en.wikipedia.org/wiki/Tidal_tensor

        .. note::

            This is in cartesian coordinates with the Euclidean metric tensor.
            Also, this isn't correct for GR.

        Parameters
        ----------
        q : Array[float, (*batch, 3,)]
            Position to compute the tidal tensor at.
        t : Array[float | int, *batch] | float | int
            Time at which to compute the tidal tensor.

        Returns
        -------
        Array[float, (*batch, 3, 3)]
            The tidal tensor.
        """
        J = self.hessian(q, t)  # (*batch, 3, 3)
        batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
        traced = (
            expand_batch_dims(xp.eye(3), ndim=len(batch_shape))
            * expand_arr_dims(xp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
            / 3
        )
        return J - traced

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
        Integrator: type[AbstractIntegrator] | None = None,
        integrator_kw: Mapping[str, Any] | None = None,
    ) -> "Orbit":
        from galax.dynamics._orbit import Orbit

        integrator_cls = Integrator if Integrator is not None else DiffraxIntegrator

        integrator = integrator_cls(self._integrator_F, **(integrator_kw or {}))
        ws = integrator.run(qp0, t0, t1, ts)
        return Orbit(q=ws[:, :3], p=ws[:, 3:-1], t=ws[:, -1], potential=self)
