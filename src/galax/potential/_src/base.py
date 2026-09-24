__all__ = ["AbstractPotential"]

import abc
import functools as ft
from dataclasses import KW_ONLY, fields, replace

from collections.abc import Mapping
from jaxtyping import Array
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import equinox as eqx
import jax
from astropy.constants import G as _CONST_G  # pylint: disable=no-name-in-module
from astropy.units import Quantity as APYQuantity
from plum import dispatch

import coordinax.ops as cxo
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.coordinates as gc
import galax.potential.custom_types as gt
from . import api
from .io import AbstractInteroperableLibrary, GalaxLibrary, convert_potential
from .plot import PlotPotentialDescriptor
from .symmetry import Symmetry
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.attr import ParametersAttribute
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.constant import ConstantParameter
from galax.potential._src.params.utils import all_parameters, all_vars
from galax.potential.dataclasses import ModuleMeta

if TYPE_CHECKING:
    import galax.dynamics  # noqa: ICN001

default_constants = ImmutableMap({"G": u.Q.from_(_CONST_G)})
DimL = u.dimension("length")
DimT = u.dimension("time")


##############################################################################


_MOVING_FRAMES = (cxo.GalileanBoost, gc.ops.ConstantRotationZOperator)


def _is_time_node(x: Any, /) -> bool:
    return isinstance(x, (AbstractPotential, AbstractParameter, *_MOVING_FRAMES))


def _time_dependence(pot: "AbstractPotential", /) -> tuple[bool, list[Any]]:
    """How ``pot`` depends on time: ``(by type, rates)``.

    "By type" is a non-constant parameter anywhere in ``pot``, including its
    component potentials: that is known without looking at any values, so it
    is known at trace time. "Rates" are quantities -- a boost velocity, a
    frame's or a bar's pattern speed -- that make ``pot`` time-dependent only
    if they are nonzero, which is a value.
    """
    by_type, rates = False, list(pot._time_rates())  # noqa: SLF001
    for x in jax.tree.leaves(pot, is_leaf=lambda x: x is not pot and _is_time_node(x)):
        if isinstance(x, AbstractPotential):
            sub_by_type, sub_rates = _time_dependence(x)
            by_type |= sub_by_type
            rates += sub_rates
        elif isinstance(x, AbstractParameter):
            by_type |= not isinstance(x, ConstantParameter)
        elif isinstance(x, cxo.GalileanBoost):
            rates.append(x.velocity)
        elif isinstance(x, gc.ops.ConstantRotationZOperator):
            rates.append(x.Omega_z)
    return by_type, rates


def _any_nonzero(rates: list[Any], /) -> Any:
    """Whether any rate is nonzero, as a boolean array (possibly traced)."""
    leaves = jax.tree.leaves(rates)
    return jnp.any(jnp.stack([jnp.any(jnp.asarray(x) != 0) for x in leaves]))


class AbstractPotential(eqx.Module, metaclass=ModuleMeta):
    """Abstract Potential Class."""

    parameters: ClassVar = ParametersAttribute(MappingProxyType({}))
    plot: ClassVar = PlotPotentialDescriptor()

    _: KW_ONLY
    units: eqx.AbstractVar[u.AbstractUnitSystem]
    """The unit system of the potential."""

    constants: eqx.AbstractVar[ImmutableMap[str, u.AbstractQuantity]]
    """The constants used by the potential."""

    @property
    def symmetry(self) -> Symmetry:
        """The symmetry this potential asserts about itself.

        Defaults to `Symmetry.NONE` -- no symmetry asserted. Subclasses
        override this to declare more, which lets callers pass inputs that are
        only well defined under that symmetry (e.g. a
        `coordinax.vecs.RadialPos`).

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> pot = gp.MiyamotoNagaiPotential(m_tot=u.Q(1e12, "Msun"),
        ...     a=u.Q(5, "kpc"), b=u.Q(1, "kpc"), units="galactic")
        >>> pot.symmetry
        <Symmetry.NONE: 'none'>

        >>> pot = gp.HernquistPotential(m_tot=u.Q(1e12, "Msun"),
        ...                             r_s=u.Q(5, "kpc"), units="galactic")
        >>> pot.symmetry
        <Symmetry.SPHERICAL: 'spherical'>

        """
        return Symmetry.NONE

    @property
    def is_time_dependent(self) -> bool:
        """Whether this potential depends on time.

        A time-independent potential can be evaluated without a time, e.g.
        ``pot.potential(xyz)``; a time-dependent one requires ``t``.

        A potential -- including any component or base potential -- depends on
        time if it has a parameter that is not a `ConstantParameter`, or a
        nonzero rate: the velocity of a `coordinax.ops.GalileanBoost`, the
        ``Omega_z`` of a `galax.coordinates.ops.ConstantRotationZOperator`, or
        whatever a subclass's ``_time_rates`` returns. A zero rate, e.g. the
        default boost of a rotation-only `coordinax.ops.GalileanOperator`, is
        time-independent. Under `jax.jit` a rate's value is unknown, so a
        potential with any rate counts as time-dependent here; evaluating it
        without a time still works, and fails at run time if the rate is
        nonzero.

        It is conservative: a custom parameter that ignores time still counts
        as time-dependent.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")
        >>> pot.is_time_dependent
        False
        >>> pot.potential(u.Q([8.0, 0, 0], "kpc"))
        Q(-0.56231277, 'kpc2 / Myr2')

        >>> m_tot = gp.params.LinearParameter(
        ...     slope=u.Q(1e9, "Msun / Myr"), point_time=u.Q(0, "Myr"),
        ...     point_value=u.Q(1e12, "Msun"))
        >>> tpot = gp.KeplerPotential(m_tot=m_tot, units="galactic")
        >>> tpot.is_time_dependent
        True
        >>> try:
        ...     tpot.potential(u.Q([8.0, 0, 0], "kpc"))
        ... except TypeError as e:
        ...     print(e)
        KeplerPotential depends on time, so a time is required. Pass `t`.

        A moving frame depends on time; a rotated one does not:

        >>> boost = cx.ops.GalileanBoost.from_([10, 0, 0], "km/s")
        >>> gp.TransformedPotential(pot, boost).is_time_dependent
        True
        >>> rot = cx.ops.GalileanOperator(
        ...     rotation=cx.ops.GalileanRotation.from_euler("z", u.Q(30, "deg")))
        >>> gp.TransformedPotential(pot, rot).is_time_dependent
        False

        """
        by_type, rates = _time_dependence(self)
        if by_type or not rates:
            return by_type
        try:
            return bool(_any_nonzero(rates))
        except jax.errors.ConcretizationTypeError:  # traced: value unknown
            return True

    def _time_rates(self) -> tuple[Any, ...]:
        """Rates that make this potential time-dependent when nonzero.

        A subclass whose ``_potential`` uses ``t`` directly -- not only through
        its parameters -- overrides this, e.g. to return a pattern speed.
        """
        return ()

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
    def _potential(self, q: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtQorVSz0:
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

    @ft.partial(jax.jit)
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
    @ft.partial(jax.jit)
    def _gradient(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0, /
    ) -> gt.FloatSz3:
        """See ``gradient``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        grad_op = jax.grad(self._potential)
        return grad_op(xyz, t)  # type: ignore[no-any-return]

    def gradient(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the gradient of the potential at the given position(s).

        See :func:`~galax.potential.gradient` for details.
        """
        return api.gradient(self, *args, **kwargs)

    # ---------------------------------------
    # Laplacian

    @vectorize_method(signature="(3),()->()")
    @ft.partial(jax.jit)
    def _laplacian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, /, t: gt.QuSz0 | gt.Sz0
    ) -> gt.FloatSz0:
        """See ``laplacian``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        hess_op = jax.hessian(self._potential, argnums=0)
        return jnp.trace(hess_op(xyz, t))  # type: ignore[no-any-return]

    def laplacian(self, *args: Any, **kwargs: Any) -> u.Quantity["1/s^2"] | Array:
        """Compute the laplacian of the potential at the given position(s).

        See :func:`~galax.potential.laplacian` for details.
        """
        return api.laplacian(self, *args, **kwargs)

    # ---------------------------------------
    # Density

    @ft.partial(jax.jit)
    def _density(self, q: gt.BBtQuSz3, t: gt.BBtQuSz0 | gt.QuSz0, /) -> gt.BBtFloatSz0:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        laplacian = self._laplacian(q, t)
        _result = laplacian / (4 * jnp.pi * self.constants["G"].value)
        return _result  # type: ignore[no-any-return]

    def density(self, *args: Any, **kwargs: Any) -> gt.BBtFloatSz0 | gt.BBtFloatQuSz0:
        """Compute the density at the given position(s).

        See :func:`~galax.potential.density` for details.
        """
        return api.density(self, *args, **kwargs)

    # ---------------------------------------
    # Hessian

    @vectorize_method(signature="(3),()->(3,3)")
    @ft.partial(jax.jit)
    def _hessian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.Sz33:
        """See ``hessian``."""
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        t = u.ustrip(AllowValue, self.units[DimT], t)
        hess_op = jax.hessian(self._potential)
        return hess_op(xyz, t)  # type: ignore[no-any-return]

    def hessian(self, *args: Any, **kwargs: Any) -> gt.BBtQuSz33 | gt.BBtSz33:
        """Compute the hessian of the potential at the given position(s).

        See :func:`~galax.potential.hessian` for details.
        """
        return api.hessian(self, *args, **kwargs)

    # ---------------------------------------

    def acceleration(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the acceleration due to the potential at the given position(s).

        See :func:`~galax.potential.acceleration` for details.
        """
        return api.acceleration(self, *args, **kwargs)

    # ---------------------------------------

    def tidal_tensor(self, *args: Any, **kwargs: Any) -> Any:
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

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")
        >>> w = gc.PhaseSpaceCoordinate(q=u.Q([8.0, 0.0, 0.0], "kpc"),
        ...                             p=u.Q([0.0, 0.0, 0.0], "km/s"),
        ...                             t=u.Q(0.0, "Gyr"))
        >>> pot.local_circular_velocity(w)
        Q(0.74987517, 'kpc / Myr')

        """
        return api.local_circular_velocity(self, *args, **kwargs)

    # ---------------------------------------

    def dpotential_dr(self, *args: Any, **kwargs: Any) -> gt.BtQuSz0 | gt.BtSz0:
        """Compute the radial derivative of the potential.

        See :func:`~galax.potential.dpotential_dr` for details.

        """
        return api.dpotential_dr(self, *args, **kwargs)

    def d2potential_dr2(self, *args: Any, **kwargs: Any) -> gt.BtQuSz0 | gt.BtSz0:
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
        solver: "galax.dynamics.OrbitSolver | None" = None,
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

        solver : :class:`~galax.dynamics.OrbitSolver`, keyword-only
            The solver to use.  If `None`, the default solver
            :class:`~galax.dynamics.OrbitSolver` is used.

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

            from galax.interop.optional_deps import OptDeps

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

    >>> pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
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

    >>> pot = gp.NFWPotential.from_({"m": 1e12, "r_s": 3, "units": "galactic"})
    >>> pot
    NFWPotential( units=..., constants=ImmutableMap({'G': ...}),
                  m=ConstantParameter(...), r_s=ConstantParameter(...) )

    """
    return cls(**obj)
