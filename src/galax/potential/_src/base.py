__all__ = ["AbstractPotential"]

import abc
from collections.abc import Mapping
from dataclasses import KW_ONLY, fields, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import equinox as eqx
import jax
from astropy.constants import G as _CONST_G  # pylint: disable=no-name-in-module
from astropy.units import Quantity as APYQuantity
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity
from xmmutablemap import ImmutableMap

import galax.typing as gt
from . import api
from .io import AbstractInteroperableLibrary, GalaxLibrary, convert_potential
from .plot import PlotPotentialDescriptor
from galax.potential._src.params.attr import ParametersAttribute
from galax.potential._src.params.utils import all_parameters, all_vars
from galax.utils._jax import vectorize_method
from galax.utils._unxt import AllowValue
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    import galax.dynamics  # noqa: ICN001

default_constants = ImmutableMap({"G": u.Quantity.from_(_CONST_G)})
DimL = u.dimension("length")
DimT = u.dimension("time")


##############################################################################


class AbstractPotential(eqx.Module, metaclass=ModuleMeta, strict=True):  # type: ignore[misc]
    """Abstract Potential Class."""

    parameters: ClassVar = ParametersAttribute(MappingProxyType({}))
    plot: ClassVar = PlotPotentialDescriptor()

    _: KW_ONLY
    units: eqx.AbstractVar[u.AbstractUnitSystem]
    """The unit system of the potential."""

    constants: eqx.AbstractVar[ImmutableMap[str, u.AbstractQuantity]]
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
                if isinstance(value, u.AbstractQuantity | APYQuantity):
                    value = u.uconvert(usys[f.metadata.get("dimensions")], value)
                    object.__setattr__(self, f.name, value)

        # Do unit conversion for the constants
        if self.units != u.unitsystems.dimensionless:
            constants = ImmutableMap(
                {k: v.decompose(usys) for k, v in self.constants.items()}
            )
            object.__setattr__(self, "constants", constants)

    ###########################################################################
    # Constructors

    @classmethod
    @dispatch.abstract
    def from_(
        cls: "type[AbstractPotential]", *args: Any, **kwargs: Any
    ) -> "AbstractPotential":
        """Create a potential from a set of arguments."""
        raise NotImplementedError  # pragma: no cover

    ###########################################################################

    # ---------------------------------------
    # Potential energy

    @abc.abstractmethod
    def _potential(
        self, q: gt.BtQuSz3, t: gt.BBtRealQuSz0, /
    ) -> gt.SpecificEnergyBtSz0:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractPotential.potential`` for an example.

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

    def potential(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the potential energy at the given position(s).

        See :func:`~galax.potential.potential` for details.
        """
        return api.potential(self, *args, **kwargs)

    @partial(jax.jit)
    def __call__(self, *args: Any) -> Any:
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
        :meth:`galax.potential.AbstractPotential.potential`
        """
        return self.potential(*args)

    # ---------------------------------------
    # Gradient

    @vectorize_method(signature="(3),()->(3)")
    @partial(jax.jit)
    def _gradient(self, xyz: gt.BtFloatQuSz3, t: gt.RealQuSz0, /) -> gt.BtSz3:
        """See ``gradient``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        grad_op = jax.grad(self._potential)
        return grad_op(xyz, t)

    def gradient(
        self: "AbstractPotential", *args: Any, **kwargs: Any
    ) -> cx.vecs.CartesianAcc3D:  # TODO: shape hint
        """Compute the gradient of the potential at the given position(s).

        See :func:`~galax.potential.gradient` for details.
        """
        return api.gradient(self, *args, **kwargs)

    # ---------------------------------------
    # Laplacian

    @vectorize_method(signature="(3),()->()")
    @partial(jax.jit)
    def _laplacian(self, xyz: gt.BtFloatQuSz3, /, t: gt.RealQuSz0) -> gt.FloatQuSz0:
        """See ``laplacian``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        jac_op = jax.jacfwd(self._gradient, argnums=0)
        return BareQuantity(jnp.trace(jac_op(xyz, t)), self.units["frequency drift"])

    def laplacian(
        self: "AbstractPotential", *args: Any, **kwargs: Any
    ) -> u.Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian of the potential at the given position(s).

        See :func:`~galax.potential.laplacian` for details.
        """
        return api.laplacian(self, *args, **kwargs)

    # ---------------------------------------
    # Density

    @partial(jax.jit)
    def _density(
        self, q: gt.BtFloatQuSz3, t: gt.BtRealQuSz0 | gt.RealQuSz0, /
    ) -> gt.BtFloatQuSz0:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        return self._laplacian(q, t) / (4 * jnp.pi * self.constants["G"])

    def density(
        self: "AbstractPotential", *args: Any, **kwargs: Any
    ) -> u.Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        See :func:`~galax.potential.density` for details.
        """
        return api.density(self, *args, **kwargs)

    # ---------------------------------------
    # Hessian

    @vectorize_method(signature="(3),()->(3,3)")
    @partial(jax.jit)
    def _hessian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.RealQuSz0 | gt.RealSz0, /
    ) -> gt.FloatQuSz33:
        """See ``hessian``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        hess_op = jax.hessian(self._potential)
        return u.Quantity(hess_op(xyz, t), self.units["frequency"] ** 2)

    def hessian(self: "AbstractPotential", *args: Any, **kwargs: Any) -> gt.BtQuSz33:
        """Compute the hessian of the potential at the given position(s).

        See :func:`~galax.potential.hessian` for details.
        """
        return api.hessian(self, *args, **kwargs)

    # ---------------------------------------

    def acceleration(
        self: "AbstractPotential", *args: Any, **kwargs: Any
    ) -> cx.vecs.CartesianAcc3D:  # TODO: shape hint
        """Compute the acceleration due to the potential at the given position(s).

        See :func:`~galax.potential.acceleration` for details.
        """
        return api.acceleration(self, *args, **kwargs)

    # ---------------------------------------

    def tidal_tensor(self, *args: Any, **kwargs: Any) -> gt.BtQuSz33:
        """Compute the tidal tensor.

        See :func:`~galax.potential.tidal_tensor` for details.
        """
        return api.tidal_tensor(self, *args, **kwargs)

    # ---------------------------------------

    def local_circular_velocity(self, *args: Any, **kwargs: Any) -> u.Quantity["speed"]:
        """Compute the local circular velocity.

        See `~galax.potential.local_circular_velocity` for details.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.coordinates as gc
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")
        >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([8.0, 0.0, 0.0], "kpc"),
        ...                             p=u.Quantity([0.0, 0.0, 0.0], "km/s"),
        ...                             t=u.Quantity(0.0, "Gyr"))
        >>> pot.local_circular_velocity(w)
        Quantity['speed'](Array(0.74987517, dtype=float64), unit='kpc / Myr')

        """
        return api.local_circular_velocity(self, *args, **kwargs)

    # ---------------------------------------

    def dpotential_dr(self, *args: Any, **kwargs: Any) -> u.Quantity["acceleration"]:
        """Compute the radial derivative of the potential.

        See :func:`~galax.potential.dpotential_dr` for details.

        """
        return api.dpotential_dr(self, *args, **kwargs)

    def d2potential_dr2(
        self, *args: Any, **kwargs: Any
    ) -> u.Quantity["frequency drift"]:
        """Compute the second radial derivative of the potential.

        See :func:`~galax.potential.d2potential_dr2` for details.

        """
        return api.d2potential_dr2(self, *args, **kwargs)

    # =========================================================================
    # Integrating orbits

    def compute_orbit(
        self,
        w0: Any,
        t: Any,
        *,
        solver: "galax.dynamics.DynamicsSolver | None" = None,
        dense: Literal[True, False] = False,
    ) -> "galax.dynamics.Orbit":
        """Compute an orbit in a potential.

        :class:`~galax.coordinates.PhaseSpaceCoordinate` includes a time in
        addition to the position (and velocity) information, enabling the orbit
        to be evaluated over a time range that is different from the initial
        time of the position. See the Examples section of
        :func:`~galax.dynamics.compute_orbit` for more details.

        Parameters
        ----------
        w0 : Any
            The phase-space coordinate from which to integrate. Integration
            includes the time of the initial position, so be sure to set the
            initial time to the desired value. See the `t` argument for more
            details.

            - :class:`~galax.dynamics.Coordinate`[float, (*batch,)]:
                The full phase-space position, including position, velocity, and
                time. `w0` will be integrated from ``w0.t`` to ``t[0]``, then
                integrated from ``t[0]`` to ``t[1]``, returning the orbit
                calculated at `t`.
            - :class:`~galax.dynamics.PhaseSpacePosition`[float, (*batch,)]:
                The full phase-space position and velocity, without time. `w0`
                will be integrated from ``t[0]`` to ``t[1]``, returning the
                orbit calculated at all `t`.
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
                :class:`~galax.integrator.Integrator` uses adaptive timesteps.

        solver : :class:`~galax.dynamics.DynamicsSolver`, keyword-only
            The solver to use.  If `None`, the default solver
            :class:`~galax.dynamics.DynamicsSolver` is used.

        dense: bool, optional keyword-only
            If `True`, return a dense (interpolated) orbit.  If `False`, return
            the orbit at the requested times.  Default is `False`.

        See Also
        --------
        galax.dynamics.compute_orbit
            The function for which this method is a wrapper. It has more details
            and examples.

        """
        from galax.dynamics import compute_orbit

        return cast(
            "galax.dynamics.Orbit",
            compute_orbit(self, w0, t, solver=solver, dense=dense),
        )

    # TODO: deprecate
    def evaluate_orbit(
        self,
        w0: Any,
        t: Any,
        *,
        integrator: "galax.dynamics.integrate.Integrator | None" = None,
        dense: Literal[True, False] = False,
    ) -> "galax.dynamics.Orbit":
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
                :class:`~galax.integrator.Integrator` uses adaptive timesteps.

        integrator : :class:`~galax.integrate.Integrator`, keyword-only
            Integrator to use.  If `None`, the default integrator
            :class:`~galax.integrator.Integrator` is used.

        dense: bool, optional keyword-only
            If `True`, return a dense (interpolated) orbit.  If `False`, return
            the orbit at the requested times.  Default is `False`.


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
            "galax.dynamics.Orbit",
            evaluate_orbit(self, w0, t, integrator=integrator, dense=dense),
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


@dispatch
def convert_potential(
    _: type[GalaxLibrary], from_: AbstractPotential, /
) -> AbstractPotential:
    """Convert the potential to an object of a different library."""
    return from_


@AbstractPotential.from_.dispatch
def from_(cls: type[AbstractPotential], obj: AbstractPotential, /) -> AbstractPotential:
    """Potential from an instance of the same type.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e11, "Msun"), units="galactic")
    >>> pot2 = gp.KeplerPotential.from_(pot)
    >>> pot2 is pot
    True

    >>> try: gp.HernquistPotential.from_(pot)
    ... except TypeError as e: print(e)
    cannot create <class 'galax.potential...HernquistPotential'>
    from <class 'galax.potential...KeplerPotential'>

    """
    if type(obj) is not cls:
        msg = f"cannot create {cls} from {type(obj)}"
        raise TypeError(msg)

    return obj


@AbstractPotential.from_.dispatch
def from_(
    cls: type[AbstractPotential],
    obj: Mapping[str, Any],
    /,
) -> AbstractPotential:
    """Convert the potential to an object of a different library.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential.from_(
    ...     {"m": u.Quantity(1e12, "Msun"), "r_s": u.Quantity(3, "kpc"),
    ...      "units": "galactic"})
    >>> pot
    NFWPotential( units=..., constants=ImmutableMap({'G': ...}),
                  m=ConstantParameter( ... ), r_s=ConstantParameter( ... ) )

    """
    return cls(**obj)
