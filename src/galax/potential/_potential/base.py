__all__ = ["AbstractPotentialBase"]

import abc
from dataclasses import KW_ONLY, fields
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

import array_api_jax_compat as xp
import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.constants import G as _G  # pylint: disable=no-name-in-module
from astropy.coordinates import BaseRepresentation
from astropy.units import Quantity as AstropyQuantity
from jax import grad, hessian, jacfwd

from jax_quantity import Quantity

from .utils import convert_input_to_array, convert_inputs_to_arrays
from galax.coordinates import PhaseSpacePosition, PhaseSpaceTimePosition
from galax.potential._potential.param.attr import ParametersAttribute
from galax.potential._potential.param.utils import all_parameters
from galax.typing import (
    BatchableFloatOrIntScalarLike,
    BatchFloatOrIntQScalar,
    BatchFloatQScalar,
    BatchFloatScalar,
    BatchMatrix33,
    BatchQVec3,
    BatchVec3,
    BatchVec6,
    FloatOrIntScalar,
    FloatScalar,
    Matrix33,
    QVecTime,
    Vec3,
    Vec6,
    VecTime,
)
from galax.units import UnitSystem, dimensionless
from galax.utils._jax import vectorize_method
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    from galax.dynamics._dynamics.integrate._api import Integrator
    from galax.dynamics._dynamics.orbit import Orbit


class AbstractPotentialBase(eqx.Module, metaclass=ModuleMeta, strict=True):  # type: ignore[misc]
    """Abstract Potential Class."""

    parameters: ClassVar = ParametersAttribute(MappingProxyType({}))

    _: KW_ONLY
    units: eqx.AbstractVar[UnitSystem]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass."""
        # Replace the ``parameters`` attribute with a mapping of the values
        type(cls).__setattr__(
            cls,
            "parameters",
            ParametersAttribute(MappingProxyType(all_parameters(cls))),
        )

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
                param.__set__(self, getattr(self, f.name))  # pylint: disable=C2801

            # Other fields, check their metadata
            elif "dimensions" in f.metadata:
                value = getattr(self, f.name)
                if isinstance(value, AstropyQuantity):
                    value = value.to_value(
                        self.units[f.metadata.get("dimensions")],
                        equivalencies=f.metadata.get("equivalencies", None),
                    )
                    object.__setattr__(self, f.name, value)

    ###########################################################################
    # Core methods that use the potential energy

    # ---------------------------------------
    # Potential energy

    # @partial(jax.jit)
    # @vectorize_method(signature="(3),()->()")
    @abc.abstractmethod
    def _potential_energy(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractPotentialBase.potential_energy`` for an example.
        """
        raise NotImplementedError

    def potential_energy(
        self,
        q: BatchVec3 | AstropyQuantity | BaseRepresentation,
        /,
        t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike | AstropyQuantity,
    ) -> BatchFloatQScalar:
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
        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        q = convert_input_to_array(q, units=self.units, no_differentials=True)
        return Quantity(self._potential_energy(q, t), self.units["specific energy"])

    @partial(jax.jit)
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
        return self._potential_energy(q, t)

    # ---------------------------------------
    # Gradient

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3)")
    def _gradient(self, q: Vec3, /, t: FloatOrIntScalar) -> Vec3:
        """See ``gradient``."""
        return grad(self._potential_energy)(q, t)

    def gradient(
        self,
        q: BatchVec3 | AstropyQuantity | BaseRepresentation,
        /,
        t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike,
    ) -> BatchQVec3:
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
        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        q = convert_input_to_array(q, units=self.units, no_differentials=True)
        return Quantity(self._gradient(q, t), self.units["acceleration"])

    # ---------------------------------------
    # Density

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _density(self, q: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        lap = jnp.trace(jacfwd(self._gradient)(q, t))
        return lap / (4 * xp.pi * self._G)

    def density(
        self, q: BatchVec3, /, t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike
    ) -> BatchFloatQScalar:
        """Compute the density value at the given position(s).

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            The position to compute the value of the potential. If the
            input position object has no units (i.e. is an `~numpy.ndarray`),
            it is assumed to be in the same unit system as the potential.
        t : (Quantity|Array)[float | int, *batch] | float | int
            The time at which to compute the value of the potential.

        Returns
        -------
        rho : Quantity[float, *batch, 'mass / length^3']
            The potential energy or value of the potential.
        """
        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        q = convert_input_to_array(q, units=self.units, no_differentials=True)
        return Quantity(self._density(q, t), self.units["mass density"])

    # ---------------------------------------
    # Hessian

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3,3)")
    def _hessian(self, q: Vec3, /, t: FloatOrIntScalar) -> Matrix33:
        """See ``hessian``."""
        return hessian(self._potential_energy)(q, t)

    def hessian(
        self,
        q: BatchVec3 | AstropyQuantity | BaseRepresentation,
        /,
        t: BatchableFloatOrIntScalarLike,
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
        q, t = convert_inputs_to_arrays(q, t, units=self.units, no_differentials=True)
        return self._hessian(q, t)

    ###########################################################################
    # Convenience methods

    def acceleration(
        self,
        q: BatchVec3 | AstropyQuantity | BaseRepresentation,
        /,
        t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike,
    ) -> BatchQVec3:
        """Compute the acceleration due to the potential.

        Parameters
        ----------
        q : Array[float, (*batch, 3)]
            Cartesian position to compute the acceleration at.
        t : (Quantity|Array)[float | int, *batch] | float | int
            Time at which to compute the acceleration.

        Returns
        -------
        Quantity[float, (*batch, 3), 'length / time^2']
            The acceleration in Cartesian coordinates. Will have the same shape
            as the input position array, ``q``.
        """
        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        q = convert_input_to_array(q, units=self.units, no_differentials=True)
        return Quantity(-self._gradient(q, t), self.units["acceleration"])

    @partial(jax.jit)
    def tidal_tensor(
        self, q: BatchVec3, /, t: BatchFloatOrIntQScalar | BatchableFloatOrIntScalarLike
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
        t : (Quantity|Array)[float | int, *batch] | float | int
            Time at which to compute the tidal tensor.

        Returns
        -------
        Array[float, (*batch, 3, 3)]
            The tidal tensor.
        """
        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        J = self.hessian(q, t)  # (*batch, 3, 3)
        batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
        traced = (
            expand_batch_dims(xp.eye(3), ndim=len(batch_shape))
            * expand_arr_dims(jnp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
            / 3
        )
        return J - traced

    # =========================================================================
    # Integrating orbits

    @partial(jax.jit)
    def _integrator_F(self, t: FloatScalar, w: Vec6, args: tuple[Any, ...]) -> Vec6:
        """Return the derivative of the phase-space position."""
        return jnp.hstack([w[3:6], self.acceleration(w[0:3], t).value])  # v, a

    # @partial(jax.jit, static_argnames=("integrator",))
    def integrate_orbit(
        self,
        w0: PhaseSpacePosition | PhaseSpaceTimePosition | BatchVec6,
        t: QVecTime | VecTime | AstropyQuantity,
        *,
        integrator: "Integrator | None" = None,
    ) -> "Orbit":
        """Integrate an orbit in the potential, from `w0` at time ``t[0]``.

        See :func:`~galax.dynamics.integrate_orbit` for more details and
        examples. If you want to use a time-aware orbit calculator see
        :meth:`~galax.potential.AbstractPotentialBase.evaluate_orbit`.

        Parameters
        ----------
        w0 : PhaseSpacePosition | Array[float, (*batch, 6)]
            The phase-space position (includes velocity) from which to
            integrate.

            - :class:`~galax.coordinates.PhaseSpacePosition`[float, (*batch,)]:
                The phase-space position. `w0` will be integrated from ``t[0]``
                to ``t[1]`` assuming that `w0` is defined at ``t[0]``, returning
                the orbit calculated at `t`.
            - :class:`~galax.coordinates.PhaseSpaceTimePosition`:
                The phase-space position, including a time. The time will be
                ignored and the orbit will be integrated from ``t[0]`` to
                ``t[1]``, returning the orbit calculated at `t`. Note: this will
                raise a warning.
            - Array[float, (*batch, 6)]:
                A :class:`~galax.coordinates.PhaseSpacePosition` will be
                constructed, interpreting the array as the  'q', 'p' (each
                Array[float, (*batch, 3)]) arguments, with 't' set to ``t[0]``.

        t: Quantity[float, (time,)]
            Array of times at which to compute the orbit. The first element
            should be the initial time and the last element should be the final
            time and the array should be monotonically moving from the first to
            final time.  See the Examples section for options when constructing
            this argument.

            .. warning::

                This is NOT the timesteps to use for integration, which are
                controlled by the `integrator`; the default integrator
                :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
                timesteps.

        integrator : AbstractIntegrator | None, keyword-only
            Integrator to use. If `None`, the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` is used.

        Returns
        -------
        orbit : Orbit
            The integrated orbit evaluated at the given times.

        See Also
        --------
        :meth:`~galax.potential.AbstractPotentialBase.evaluate_orbit`
            A higher-level function that computes the orbit using time
            information from `w0`.
        galax.dynamics.evaluate_orbit
            The function which
            :meth:`~galax.potential.AbstractPotentialBase.evaluate_orbit` calls.
        galax.dynamics.integrate_orbit
            The function for which this method is a wrapper. It has more details
            and examples.
        """
        from galax.dynamics._dynamics.orbit import integrate_orbit

        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        return cast("Orbit", integrate_orbit(self, w0, t, integrator=integrator))

    def evaluate_orbit(
        self,
        w0: PhaseSpacePosition | PhaseSpaceTimePosition | BatchVec6,
        t: QVecTime | VecTime | AstropyQuantity,  # TODO: must be a Quantity
        *,
        integrator: "Integrator | None" = None,
    ) -> "Orbit":
        """Compute an orbit in a potential.

        This method is similar to
        :meth:`~galax.potential.AbstractPotentialBase.integrate_orbit`, but can
        behave differently when ``w0`` is a
        :class:`~galax.coordinates.PhaseSpacePositionTime`.
        :class:`~galax.coordinates.PhaseSpacePositionTime` includes a time in
        addition to the position (and velocity) information, enabling the orbit
        to be evaluated over a time range that is different from the initial
        time of the position. See the Examples section of
        :func:`~galax.dynamics.evaluate_orbit` for more details.

        Parameters
        ----------
        pot : :class:`~galax.potential.AbstractPotentialBase`
            The potential in which to compute the orbit.
        w0 : PhaseSpaceTimePosition
            The phase-space position (includes velocity and time) from which to
            integrate. Integration includes the time of the initial position, so
            be sure to set the initial time to the desired value. See the `t`
            argument for more details.

            - :class:`~galax.dynamics.PhaseSpacePosition`[float, (*batch,)]:
                The full phase-space position, including position, velocity, and
                time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
                integrated from ``t[0]`` to ``t[1]``, returning the orbit
                calculated at `t`.
            - :class:`~galax.coordinates.PhaseSpacePosition`[float, (*batch,)]:
                The phase-space position. `w0` will be integrated from ``t[0]``
                to ``t[1]`` assuming that `w0` is defined at ``t[0]``, returning
                the orbit calculated at `t`.
            - Array[float, (*batch, 6)]:
                A :class:`~galax.coordinates.PhaseSpacePosition` will be
                constructed, interpreting the array as the  'q', 'p' (each
                Array[float, (*batch, 3)]) arguments, with 't' set to ``t[0]``.
        t: Quantity[float, (time,)]
            Array of times at which to compute the orbit. The first element
            should be the initial time and the last element should be the final
            time and the array should be monotonically moving from the first to
            final time.  See the Examples section for options when constructing
            this argument.

            .. note::

                This is NOT the timesteps to use for integration, which are
                controlled by the `integrator`; the default integrator
                :class:`~galax.integrator.DiffraxIntegrator` uses adaptive
                timesteps.

        integrator : :class:`~galax.integrate.Integrator`, keyword-only
            Integrator to use.  If `None`, the default integrator
            :class:`~galax.integrator.DiffraxIntegrator` is used.

        Returns
        -------
        orbit : :class:`~galax.dynamics.Orbit`
            The integrated orbit evaluated at the given times.

        See Also
        --------
        galax.dynamics.evaluate_orbit
            The function for which this method is a wrapper. It has more details
            and examples.
        """
        from galax.dynamics._dynamics.orbit import evaluate_orbit

        t = Quantity.constructor(t, self.units["time"]).value  # TODO: value
        return cast("Orbit", evaluate_orbit(self, w0, t, integrator=integrator))
