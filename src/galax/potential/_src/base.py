__all__ = ["AbstractBasePotential"]

import abc
from dataclasses import KW_ONLY, fields, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import equinox as eqx
import jax
from astropy.constants import G as _CONST_G  # pylint: disable=no-name-in-module
from astropy.units import Quantity as APYQuantity
from jaxtyping import Float
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as ux
from xmmutablemap import ImmutableMap

import galax.typing as gt
from .io import AbstractInteroperableLibrary, GalaxLibrary, convert_potential
from .plot import PlotPotentialDescriptor
from galax.potential._src.params.attr import ParametersAttribute
from galax.potential._src.params.utils import all_parameters, all_vars
from galax.utils._jax import vectorize_method
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    from galax.dynamics import Orbit
    from galax.dynamics.integrate import Integrator

default_constants = ImmutableMap({"G": ux.Quantity(_CONST_G.value, _CONST_G.unit)})


##############################################################################


class AbstractBasePotential(eqx.Module, metaclass=ModuleMeta, strict=True):  # type: ignore[misc]
    """Abstract Potential Class."""

    parameters: ClassVar = ParametersAttribute(MappingProxyType({}))
    plot: ClassVar = PlotPotentialDescriptor()

    _: KW_ONLY
    units: eqx.AbstractVar[ux.AbstractUnitSystem]
    """The unit system of the potential."""

    constants: eqx.AbstractVar[ImmutableMap[str, ux.Quantity]]
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

    def _apply_unitsystem(self) -> None:
        from galax.potential._src.params.field import ParameterField

        usys = self.units

        # Handle unit conversion for all fields, e.g. the parameters.
        for f in fields(self):
            # Process ParameterFields
            param = getattr(self.__class__, f.name, None)
            if isinstance(param, ParameterField):
                # Re-call setter, since the ``.units`` are now known
                param.__set__(self, getattr(self, f.name))  # pylint: disable=C2801

            # Other fields, check their metadata
            elif "dimensions" in f.metadata:
                value = getattr(self, f.name)
                # Only need to set again if a conversion is needed
                if isinstance(value, ux.AbstractQuantity | APYQuantity):
                    value = ux.uconvert(usys[f.metadata.get("dimensions")], value)
                    object.__setattr__(self, f.name, value)

        # Do unit conversion for the constants
        if self.units != ux.unitsystems.dimensionless:
            constants = ImmutableMap(
                {k: v.decompose(usys) for k, v in self.constants.items()}
            )
            object.__setattr__(self, "constants", constants)

    ###########################################################################
    # Core methods that use the potential energy

    # ---------------------------------------
    # Potential energy

    @abc.abstractmethod
    def _potential(
        self, q: gt.BatchQVec3, t: gt.BatchableRealQScalar, /
    ) -> gt.SpecificEnergyBatchScalar:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractBasePotential.potential`` for an example.

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
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> ux.Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy at the given position(s).

        See :func:`~galax.potential.potential` for details.
        """
        from .funcs import potential

        return potential(self, *args, **kwargs)

    @partial(jax.jit, inline=True)
    def __call__(self, *args: Any) -> Float[ux.Quantity["specific energy"], "*batch"]:
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
        :meth:`galax.potential.AbstractBasePotential.potential`
        """
        return self.potential(*args)

    # ---------------------------------------
    # Gradient

    @partial(jax.jit, inline=True)
    @vectorize_method(signature="(3),()->(3)")
    def _gradient(self, q: gt.BatchQVec3, t: gt.RealQScalar, /) -> gt.BatchQVec3:
        """See ``gradient``."""
        grad_op = ux.experimental.grad(
            self._potential, units=(self.units["length"], self.units["time"])
        )
        return grad_op(q, t)

    def gradient(
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> cx.vecs.CartesianAcc3D:  # TODO: shape hint
        """Compute the gradient of the potential at the given position(s).

        See :func:`~galax.potential.gradient` for details.
        """
        from .funcs import gradient

        return gradient(self, *args, **kwargs)

    # ---------------------------------------
    # Laplacian

    @partial(jax.jit, inline=True)
    @vectorize_method(signature="(3),()->()")
    def _laplacian(self, q: gt.QVec3, /, t: gt.RealQScalar) -> gt.FloatQScalar:
        """See ``laplacian``."""
        jac_op = ux.experimental.jacfwd(
            self._gradient, units=(self.units["length"], self.units["time"])
        )
        return jnp.trace(jac_op(q, t))

    def laplacian(
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> ux.Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian of the potential at the given position(s).

        See :func:`~galax.potential.laplacian` for details.
        """
        from .funcs import laplacian

        return laplacian(self, *args, **kwargs)

    # ---------------------------------------
    # Density

    @partial(jax.jit, inline=True)
    def _density(
        self, q: gt.BatchQVec3, t: gt.BatchRealQScalar | gt.RealQScalar, /
    ) -> gt.BatchFloatQScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        return self._laplacian(q, t) / (4 * jnp.pi * self.constants["G"])

    def density(
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> ux.Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        See :func:`~galax.potential.density` for details.
        """
        from .funcs import density

        return density(self, *args, **kwargs)

    # ---------------------------------------
    # Hessian

    @partial(jax.jit, inline=True)
    @vectorize_method(signature="(3),()->(3,3)")
    def _hessian(self, q: gt.QVec3, t: gt.RealQScalar, /) -> gt.QMatrix33:
        """See ``hessian``."""
        hess_op = ux.experimental.hessian(
            self._potential, units=(self.units["length"], self.units["time"])
        )
        return hess_op(q, t)

    def hessian(
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> gt.BatchQMatrix33:
        """Compute the hessian of the potential at the given position(s).

        See :func:`~galax.potential.hessian` for details.
        """
        from .funcs import hessian as hessian_func

        return hessian_func(self, *args, **kwargs)

    ###########################################################################
    # Convenience methods

    def acceleration(
        self: "AbstractBasePotential", *args: Any, **kwargs: Any
    ) -> cx.vecs.CartesianAcc3D:  # TODO: shape hint
        """Compute the acceleration due to the potential at the given position(s).

        See :func:`~galax.potential.acceleration` for details.
        """
        from .funcs import acceleration

        return acceleration(self, *args, **kwargs)

    def tidal_tensor(self, *args: Any, **kwargs: Any) -> gt.BatchQMatrix33:
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
    def _dynamics_deriv(
        self,
        t: gt.FloatScalar,
        w: gt.Vec6,
        args: tuple[Any, ...],  # noqa: ARG002
    ) -> gt.Vec6:
        r"""Differential equation derivative for dynamics.

        This is Hamilton's equations for motion for a particle in a potential.

        .. math::

                \dot{q} = \frac{dH}{dp} \\ \dot{p} = -\frac{dH}{dq}

        Parameters
        ----------
        t : float
            Time at which to evaluate the derivative.
        w : Array[float, (6,)]
            Phase-space position [q (3,), p (3,)].
        args : tuple
            Additional arguments to pass to the derivative. Not used, but
            required by some dynamics solvers.

        Returns
        -------
        dw : Array[float, (6,)]
            Derivative [p (3,), a (3,)] at the phase-space position.
        """
        # TODO: not require unit munging
        a = ux.ustrip(
            self.units["acceleration"],
            -self._gradient(
                ux.Quantity(w[0:3], self.units["length"]),
                ux.Quantity(t, self.units["time"]),
            ),
        )
        return jnp.hstack([w[3:6], a])  # v, a

    def evaluate_orbit(
        self,
        w0: Any,
        t: Any,
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
        w0 : Any
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
                :class:`~galax.integrator.Integrator` uses adaptive
                timesteps.

        integrator : :class:`~galax.integrate.Integrator`, keyword-only
            Integrator to use.  If `None`, the default integrator
            :class:`~galax.integrator.Integrator` is used.

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

    # =========================================================================
    # Interoperability

    def as_interop(self, library: type[AbstractInteroperableLibrary], /) -> object:
        """Convert the potential to an object of a different library.

        Parameters
        ----------
        library : :class:`~galax.potential.io.AbstractInteroperableLibrary`
            The library type to convert the potential to.

        Examples
        --------
        .. invisible-code-block: python

            from galax._interop.optional_deps import OptDeps

        .. skip: start if(not OptDeps.GALA.installed, reason="requires gala")

        >>> import galax.potential as gp
        >>> pot = gp.MilkyWayPotential()

        Convert the potential to a :mod:`gala` potential
        >>> gala_pot = pot.as_interop(gp.io.GalaLibrary)
        >>> gala_pot
        <CompositePotential disk,bulge,nucleus,halo>

        Now converting back to a :mod:`galax` potential

        >>> pot2 = gp.io.convert_potential(gp.io.GalaxLibrary, gala_pot)
        >>> pot2 == pot
        Array(True, dtype=bool)

        .. skip: end
        """
        return convert_potential(library, self)


##############################################################################


@dispatch  # type: ignore[misc]
def convert_potential(
    to_: type[GalaxLibrary],  # noqa: ARG001
    from_: AbstractBasePotential,
    /,
) -> AbstractBasePotential:
    """Convert the potential to an object of a different library."""
    return from_
