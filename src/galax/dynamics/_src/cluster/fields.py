"""Fields for mass evolution."""

__all__ = [
    "MassVectorField",
    "AbstractMassRateField",
    "CustomMassRateField",
    "ZeroMassRate",
    "ConstantMassRate",
    "Baumgardt1998MassLossRate",
]

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any, Protocol, TypeAlias, TypedDict, cast, final, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real, Shaped

import unxt as u
from unxt.quantity import AllowValue, BareQuantity as FastQ

from .api import relaxation_time, tidal_radius
from .radius import AbstractTidalRadiusMethod, King1962
from .relax_time import AbstractRelaxationTimeMethod, Baumgardt1998
from galax.dynamics._src.fields import AbstractField
from galax.dynamics._src.orbit.orbit import Orbit

Time: TypeAlias = Any
ClusterMass: TypeAlias = Any


class FieldArgs(TypedDict, total=False):
    orbit: Orbit | None
    # Add other optional keys here if needed


@runtime_checkable
class MassVectorField(Protocol):
    """Protocol for mass vector field.

    This is a function that returns the derivative of the mass vector with
    respect to time.

    Examples
    --------
    >>> from galax.dynamics.cluster import MassVectorField

    >>> def mass_deriv(t, Mc, args, **kwargs): pass

    >>> isinstance(mass_deriv, MassVectorField)
    True

    """

    def __call__(
        self, t: Time, Mc: ClusterMass, args: FieldArgs, /, **kwargs: Any
    ) -> Array: ...


class AbstractMassRateField(AbstractField):
    """ABC for mass fields.

    Methods
    -------
    __call__ : `galax.dynamics.cluster.MassVectorField`
        Compute the mass field.
    terms : the `diffrax.AbstractTerm` `jaxtyping.PyTree` for integration.

    """

    @abstractmethod
    def __call__(
        self, t: Time, Mc: ClusterMass, args: FieldArgs, /, **kwargs: Any
    ) -> Array:
        raise NotImplementedError  # pragma: no cover

    @AbstractField.terms.dispatch  # type: ignore[misc]
    def terms(
        self: "AbstractMassRateField", _: dfx.AbstractSolver, /
    ) -> PyTree[dfx.AbstractTerm]:
        """Return diffeq terms for integration.

        Examples
        --------
        >>> import diffrax as dfx
        >>> import galax.dynamics as gd

        >>> field = gd.cluster.ZeroMassRate(units="galactic")
        >>> field.terms(dfx.Dopri8())
        ODETerm(
            vector_field=_JitWrapper( fn='ZeroMassRate.__call__', ... ) )

        """
        return dfx.ODETerm(eqx.filter_jit(self.__call__))


#####################################################


class CustomMassRateField(AbstractMassRateField):
    """User-defined mass field.

    This takes a user-defined function of type
    `galax.dynamics.cluster.MassVectorField`.

    """

    #: User-defined mass derivative function of type
    #: `galax.dynamics.cluster.MassVectorField`
    mass_deriv: MassVectorField

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        converter=u.unitsystem, default="galactic", static=True
    )

    def __call__(
        self, t: Time, Mc: ClusterMass, args: FieldArgs, /, **kwargs: Any
    ) -> Array:
        return self.mass_deriv(t, Mc, args, **kwargs)


#####################################################


@final
class ZeroMassRate(AbstractMassRateField):
    r"""Constant mass (zero mass loss) field.

    $$
        \frac{dM(t)}{dt} = 0
    $$

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> dmdt_fn = gdc.ZeroMassRate(units="galactic")

    Evaluating the vector field:

    >>> t = u.Quantity(0, "Gyr")
    >>> M = u.Quantity(1e4, "Msun")
    >>> dmdt = dmdt_fn(t, M, {})
    >>> dmdt
    Array(0., dtype=float64)

    Showing it in the mass solver:

    >>> mass_solver = gdc.MassSolver()
    >>> t0, t1 = u.Quantity([0, 1], "Gyr")
    >>> saveat = jnp.linspace(t0, t1, 5)
    >>> mass_history = mass_solver.solve(dmdt_fn, M, t0, t1, saveat=saveat)
    >>> mass_history.ys
    Array([10000., 10000., 10000., 10000., 10000.], dtype=float64)

    """

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        converter=u.unitsystem, default="galactic", static=True
    )

    def __call__(
        self,
        t: Time,  # noqa: ARG002
        M: ClusterMass,
        args: FieldArgs,  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> Shaped[Array, "{M}"]:
        return jnp.zeros_like(M)


######################################################


@final
class ConstantMassRate(AbstractMassRateField):
    r"""Constant mass rate, ie linear mass change.

    $$
        \frac{dM(t)}{dt} = x
    $$

    Where $x$ is a constant. It should be negative for mass loss, 0 for constant
    mass, and positive for mass gain.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> dmdt_fn = gdc.ConstantMassRate(-1, units="galactic")

    Evaluating the vector field:

    >>> t = u.Quantity(0, "Gyr")
    >>> M = u.Quantity(1e4, "Msun")
    >>> dmdt = dmdt_fn(t, M, {})
    >>> dmdt
    Array(-1., dtype=float64)

    Showing it in the mass solver:

    >>> mass_solver = gdc.MassSolver()
    >>> t0, t1 = u.Quantity([0, 1], "Gyr")
    >>> saveat = jnp.linspace(t0, t1, 5)
    >>> mass_history = mass_solver.solve(dmdt_fn, M, t0, t1, saveat=saveat)
    >>> mass_history.ys
    Array([10000.,  9750.,  9500.,  9250.,  9000.], dtype=float64)

    """

    dm_dt: Real[Array | u.AbstractQuantity, ""] = eqx.field(converter=jnp.asarray)
    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        converter=u.unitsystem, default="galactic", static=True
    )

    def __call__(
        self,
        t: Time,  # noqa: ARG002
        M: ClusterMass,
        args: FieldArgs,  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> Shaped[Array, "{M}"]:
        unit = self.units["mass"] / self.units["time"]
        return u.ustrip(AllowValue, unit, self.dm_dt) * jnp.ones_like(M)


######################################################


class Baumgardt1998MassLossRate(AbstractMassRateField):
    r"""Mass loss field from Baumgardt (1998).

    This is the mass loss field from Baumgardt (1998) for modeling the mass loss
    of a star cluster due to tidal stripping.

    $$
        \frac{dM(t)}{dt} = -\xi_0 \sqrt{1 + (\alpha \frac{r_h}{r_t(t)})^3} \frac{M(t)}{t_{relax}(t)}
    $$

    where $\\xi_0$ and $\\alpha$ are constant, $r_h$ is the half-mass radius,
    $r_t$ is the tidal radius, $t_{relax}$ is the relaxation time, and $M(t)$ is
    the mass of the cluster at time $t$.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> dMdt_fn = gdc.Baumgardt1998MassLossRate(units="galactic")
    >>> dMdt_fn.tidal_radius_flag
    <class 'galax.dynamics...King1962'>
    >>> dMdt_fn.relaxation_time_flag
    <class 'galax.dynamics...Baumgardt1998'>

    In order to evaluate the mass loss we need a `galax.dynamics.Orbit` object.

    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.MilkyWayPotential2022()
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([8, 0, 0], "kpc"),
    ...     p=u.Quantity([0, 180, 0], "km/s"), t=u.Quantity(0, "Gyr"))
    >>> orbit = gd.evaluate_orbit(pot, w0, u.Quantity([0, 1], "Gyr"), dense=True)

    >>> kwargs = {"orbit": orbit, "m_avg": u.Quantity(3, "Msun"),
    ...           "xi0": 0.001, "alpha": 14.9, "r_hm": u.Quantity(1, "pc")}

    >>> dMdt_fn(0, u.Quantity(1e4, "Msun"), kwargs)  # [Msun/Myr]
    Array(-1.10392877, dtype=float64)

    """  # noqa: E501

    #: Tidal radius method to use. See `galax.dynamics.cluster.radius` for more
    #: options. The default is `King1962`, which is $$ r_t^3 = \frac{G
    #: M_c}{\Omega^2 - \frac{d^2\Phi}{dr^2}} $$.
    tidal_radius_flag: AbstractTidalRadiusMethod = eqx.field(
        default=King1962, static=True
    )
    #: Relaxation time method to use. See `galax.dynamics.cluster.relax_time`
    #: for more options. The default is `Baumgardt1998`, which is : $$ t_r =
    #: \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \ln(0.4 N)} $$.
    relaxation_time_flag: AbstractRelaxationTimeMethod = eqx.field(
        default=Baumgardt1998, static=True
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(
        converter=u.unitsystem, default="galactic", static=True
    )

    def __call__(self, t: Time, M: ClusterMass, args: Any, /, **kw: Any) -> Array:  # noqa: ARG002
        # Setup
        orbit = args["orbit"]
        orbit = cast(Orbit, eqx.error_if(orbit, orbit is None, "need orbit"))
        pot = orbit.potential
        usys = pot.units
        Mq = FastQ.from_(M, usys["mass"])

        # Compute
        r_t = tidal_radius(self.tidal_radius_flag, pot, orbit(t), mass=Mq)
        r_t = r_t.uconvert("pc")
        t_relax = relaxation_time(
            self.relaxation_time_flag,
            Mq,
            args["r_hm"],
            args["m_avg"],
            G=pot.constants["G"],
        ).uconvert("Myr")
        r_ratio = u.ustrip("", args["r_hm"] / r_t)

        dmdt = (
            -args["xi0"] * jnp.sqrt(1.0 + (args["alpha"] * r_ratio) ** 3) * Mq / t_relax
        )
        return u.ustrip(usys["mass"] / usys["time"], dmdt)
