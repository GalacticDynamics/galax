__all__ = ["AbstractPotentialBase"]

import abc
from dataclasses import KW_ONLY, fields, replace
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
    BatchVec6,
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


default_integrator = DiffraxIntegrator()


class AbstractPotentialBase(eqx.Module, metaclass=ModuleMeta, strict=True):  # type: ignore[misc]
    """Potential Class."""

    _: KW_ONLY
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
    def _integrator_F(self, t: FloatScalar, qp: Vec6, args: tuple[Any, ...]) -> Vec6:
        """Return the derivative of the phase-space position."""
        return xp.hstack([qp[3:6], self.acceleration(qp[0:3], t)])  # v, a

    @partial_jit(static_argnames=("integrator"))
    def integrate_orbit(
        self,
        qp0: BatchVec6,
        t: Float[Array, "time"],
        *,
        integrator: AbstractIntegrator | None = None,
    ) -> "Orbit":
        """Integrate an orbit in the potential.

        Parameters
        ----------
        qp0 : Array[float, (6,)]
            Initial position and velocity.
        t: Array[float, (T,)]
            Array of times at which to compute the orbit. The first element
            should be the initial time and the last element should be the final
            time and the array should be monotonically moving from the first to
            final time.  See the Examples section for options when constructing
            this argument.

            .. note::

                To integrate backwards in time, ...

            .. warning::

                This is NOT the timesteps to use for integration, which are
                controlled by the `integrator`; the default integrator
                :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
                timesteps.

        integrator : AbstractIntegrator, keyword-only
            Integrator to use. If `None`, the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` is used.

        Returns
        -------
        orbit : Orbit
            The integrated orbit evaluated at the given times.

        Examples
        --------
        We start by integrating a single orbit in the potential of a point mass.
        A few standard imports are needed:

        >>> import astropy.units as u
        >>> import jax.experimental.array_api as xp  # preferred over `jax.numpy`
        >>> import galax.potential as gp
        >>> from galax.units import galactic

        We can then create the point-mass' potential, with galactic units:

        >>> potential = gp.KeplerPotential(m=1e12 * u.Msun, units=galactic)

        We can then integrate an initial phase-space position in this potential
        to get an orbit:

        >>> xv0 = xp.asarray([10., 0., 0., 0., 0.1, 0.])  # (x, v) galactic units
        >>> ts = xp.linspace(0., 1000, 4)  # (1 Gyr, 4 steps)
        >>> orbit = potential.integrate_orbit(xv0, ts)
        >>> orbit
        Orbit(
            q=f64[4,3], p=f64[4,3], t=f64[4], potential=KeplerPotential(...)
        )

        Note how there are 4 points in the orbit, corresponding to the 4 steps.
        Changing the number of steps is easy:

        >>> ts = xp.linspace(0., 1000, 10)  # (1 Gyr, 4 steps)
        >>> orbit = potential.integrate_orbit(xv0, ts)
        >>> orbit
        Orbit(
            q=f64[10,3], p=f64[10,3], t=f64[10], potential=KeplerPotential(...)
        )

        We can also integrate a batch of orbits at once:

        >>> xv0 = xp.asarray([[10., 0., 0., 0., 0.1, 0.], [10., 0., 0., 0., 0.2, 0.]])
        >>> orbit = potential.integrate_orbit(xv0, ts)
        >>> orbit
        Orbit(
            q=f64[2,10,3], p=f64[2,10,3], t=f64[10], potential=KeplerPotential(...)
        )
        """
        # TODO: ꜛ get NORMALIZE_WHITESPACE to work correctly so Orbit is 1 line
        from galax.dynamics._orbit import Orbit

        integrator_ = default_integrator if integrator is None else replace(integrator)

        ws = integrator_.run(self._integrator_F, qp0, t)
        return Orbit(q=ws[..., :3], p=ws[..., 3:-1], t=t, potential=self)
