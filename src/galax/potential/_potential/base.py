__all__ = ["AbstractPotentialBase"]

import abc
from dataclasses import KW_ONLY, fields, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from astropy.constants import G as _CONST_G  # pylint: disable=no-name-in-module
from astropy.units import Quantity as APYQuantity
from jaxtyping import Array, Float, Shaped

import coordinax as cx
import quaxed.array_api as xp
import quaxed.numpy as qnp
import unxt
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from galax.coordinates import PhaseSpacePosition
from galax.potential._potential.param.attr import ParametersAttribute
from galax.potential._potential.param.utils import all_parameters, all_vars
from galax.utils._collections import ImmutableDict
from galax.utils._jax import vectorize_method
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    from galax.dynamics._dynamics.integrate._api import Integrator
    from galax.dynamics._dynamics.orbit import Orbit


QMatrix33: TypeAlias = Float[Quantity, "3 3"]
BatchMatrix33: TypeAlias = Shaped[Float[Array, "3 3"], "*batch"]
BatchQMatrix33: TypeAlias = Shaped[QMatrix33, "*batch"]
HessianVec: TypeAlias = Shaped[Quantity["1/s^2"], "*#shape 3 3"]  # TODO: shape -> batch

# Position and time input options
PositionalLike: TypeAlias = (
    cx.AbstractPosition3D | gt.LengthBroadBatchVec3 | Shaped[Array, "*#batch 3"]
)
TimeOptions: TypeAlias = (
    gt.BatchRealQScalar
    | gt.FloatQScalar
    | gt.IntQScalar
    | gt.BatchableRealScalarLike
    | gt.FloatScalar
    | gt.IntScalar
    | APYQuantity
)

CONST_G = Quantity(_CONST_G.value, _CONST_G.unit)


default_constants = ImmutableDict({"G": CONST_G})


##############################################################################


class AbstractPotentialBase(eqx.Module, metaclass=ModuleMeta, strict=True):  # type: ignore[misc]
    """Abstract Potential Class."""

    parameters: ClassVar = ParametersAttribute(MappingProxyType({}))

    _: KW_ONLY
    units: eqx.AbstractVar[AbstractUnitSystem]
    """The unit system of the potential."""

    constants: eqx.AbstractVar[ImmutableDict[Quantity]]
    """The constants used by the potential."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the subclass."""
        # Replace the ``parameters`` attribute with a mapping of the values
        paramattr = replace(
            all_vars(cls)["parameters"],
            parameters=MappingProxyType(all_parameters(cls)),
        )
        paramattr.__set_name__(cls, "parameters")
        type(cls).__setattr__(cls, "parameters", paramattr)

    ###########################################################################
    # Parsing

    def _init_units(self) -> None:
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
                if isinstance(value, APYQuantity):
                    value = value.to_units_value(
                        self.units[f.metadata.get("dimensions")],
                        equivalencies=f.metadata.get("equivalencies", None),
                    )
                    object.__setattr__(self, f.name, value)

        # Do unit conversion for the constants
        if self.units != unxt.unitsystems.dimensionless:
            constants = ImmutableDict(
                {k: v.decompose(self.units) for k, v in self.constants.items()}
            )
            object.__setattr__(self, "constants", constants)

    ###########################################################################
    # Core methods that use the potential energy

    # ---------------------------------------
    # Potential energy

    # TODO: inputs w/ units
    # @partial(jax.jit)
    # @vectorize_method(signature="(3),()->()")
    @abc.abstractmethod
    def _potential(
        self, q: gt.QVec3, t: gt.RealQScalar, /
    ) -> Shaped[Quantity["specific energy"], ""]:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractPotentialBase.potential`` for an example.

        Parameters
        ----------
        q : Quantity[float, (3,), 'length']
            The Cartesian position at which to compute the value of the
            potential. The units are the same as the potential's unit system.
        t : Quantity[float, (), 'time']
            The time at which to compute the value of the potential.
            The units are the same as the potential's unit system.

        Returns
        -------
        E : Quantity[Array, (), 'specific energy']
            The specific potential energy in the unit system of the potential.
        """
        raise NotImplementedError

    def potential(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy at the given position(s).

        See :func:`~galax.potential.potential` for details.
        """
        from .funcs import potential

        return potential(self, *args, **kwargs)

    @partial(jax.jit)
    def __call__(self, *args: Any) -> Float[Quantity["specific energy"], "*batch"]:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        *args : Any
            Arguments to pass to the potential method.
            See :func:`~galax.potential.potential`.

        Returns
        -------
        E : Quantity[float, (*batch,), 'specific energy']
            The potential energy per unit mass or value of the potential.

        See Also
        --------
        :func:`galax.potential.potential`
        :meth:`galax.potential.AbstractPotentialBase.potential`
        """
        return self.potential(*args)

    # ---------------------------------------
    # Gradient

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3)")
    def _gradient(self, q: gt.BatchQVec3, /, t: gt.RealQScalar) -> gt.BatchQVec3:
        """See ``gradient``."""
        grad_op = unxt.experimental.grad(
            self._potential, units=(self.units["length"], self.units["time"])
        )
        return grad_op(q, t)

    def gradient(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient of the potential at the given position(s).

        See :func:`~galax.potential.gradient` for details.
        """
        from .funcs import gradient

        return gradient(self, *args, **kwargs)

    # ---------------------------------------
    # Laplacian

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _laplacian(self, q: gt.QVec3, /, t: gt.RealQScalar) -> gt.FloatQScalar:
        """See ``laplacian``."""
        jac_op = unxt.experimental.jacfwd(
            self._gradient, units=(self.units["length"], self.units["time"])
        )
        return qnp.trace(jac_op(q, t))

    def laplacian(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian of the potential at the given position(s).

        See :func:`~galax.potential.laplacian` for details.
        """
        from .funcs import laplacian

        return laplacian(self, *args, **kwargs)

    # ---------------------------------------
    # Density

    @partial(jax.jit)
    def _density(
        self, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        return self._laplacian(q, t) / (4 * xp.pi * self.constants["G"])

    def density(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        See :func:`~galax.potential.density` for details.
        """
        from .funcs import density

        return density(self, *args, **kwargs)

    # ---------------------------------------
    # Hessian

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3,3)")
    def _hessian(self, q: gt.QVec3, /, t: gt.RealQScalar) -> QMatrix33:
        """See ``hessian``."""
        hess_op = unxt.experimental.hessian(
            self._potential, units=(self.units["length"], self.units["time"])
        )
        return hess_op(q, t)

    def hessian(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> BatchQMatrix33:
        """Compute the hessian of the potential at the given position(s).

        See :func:`~galax.potential.hessian` for details.
        """
        from .funcs import hessian

        return hessian(self, *args, **kwargs)

    ###########################################################################
    # Convenience methods

    # ---------------------------------------
    # Acceleration

    def acceleration(
        self: "AbstractPotentialBase", *args: Any, **kwargs: Any
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration due to the potential at the given position(s).

        See :func:`~galax.potential.acceleration` for details.
        """
        from .funcs import acceleration

        return acceleration(self, *args, **kwargs)

    # ---------------------------------------
    # Tidal tensor

    def tidal_tensor(self, *args: Any, **kwargs: Any) -> BatchMatrix33:
        """Compute the tidal tensor.

        See :func:`~galax.potential.tidal_tensor` for details.
        """
        from .funcs import tidal_tensor

        return tidal_tensor(self, *args, **kwargs)

    # =========================================================================
    # Integrating orbits

    @partial(jax.jit, inline=True)  # TODO: inline?
    @vectorize_method(  # TODO: vectorization the func itself
        signature="(),(6)->(6)", excluded=(2,)
    )
    def _integrator_F(
        self,
        t: gt.FloatScalar,
        w: gt.Vec6,
        args: tuple[Any, ...],  # noqa: ARG002
    ) -> gt.Vec6:
        """Return the derivative of the phase-space position."""
        a = self.acceleration(w[0:3], t).to_units_value(self.units["acceleration"])
        return jnp.hstack([w[3:6], a])  # v, a

    def evaluate_orbit(
        self,
        w0: PhaseSpacePosition | gt.BatchVec6,
        t: gt.QVecTime | gt.VecTime | APYQuantity,  # TODO: must be a Quantity
        *,
        integrator: "Integrator | None" = None,
        interpolated: Literal[True, False] = False,
    ) -> "Orbit":
        """Compute an orbit in a potential.

        :class:`~galax.coordinates.PhaseSpacePosition` includes a time in
        addition to the position (and velocity) information, enabling the orbit
        to be evaluated over a time range that is different from the initial
        time of the position. See the Examples section of
        :func:`~galax.dynamics.evaluate_orbit` for more details.

        Parameters
        ----------
        pot : :class:`~galax.potential.AbstractPotentialBase`
            The potential in which to compute the orbit.
        w0 : PhaseSpacePosition
            The phase-space position (includes velocity and time) from which to
            integrate. Integration includes the time of the initial position, so
            be sure to set the initial time to the desired value. See the `t`
            argument for more details.

            - :class:`~galax.dynamics.PhaseSpacePosition`[float, (*batch,)]:
                The full phase-space position, including position, velocity, and
                time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
                integrated from ``t[0]`` to ``t[1]``, returning the orbit
                calculated at `t`. If ``w0.t`` is `None`, the initial time is
                assumed to be ``t[0]``.
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

        interpolated: bool, optional keyword-only
                If `True`, return an interpolated orbit.  If `False`, return the orbit
                at the requested times.  Default is `False`.


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
        from galax.dynamics import evaluate_orbit

        return cast(
            "Orbit",
            evaluate_orbit(
                self, w0, t, integrator=integrator, interpolated=interpolated
            ),
        )
