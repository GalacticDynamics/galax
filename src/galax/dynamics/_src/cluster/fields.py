"""Fields for mass evolution."""

__all__ = [
    "MassVectorField",
    "AbstractMassField",
    "CustomMassField",
    "ConstantMass",
    "Baumgardt1998MassLoss",
]

from abc import abstractmethod
from dataclasses import KW_ONLY
from typing import Any, Protocol, TypeAlias, TypedDict, cast, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import unxt as u
from unxt.quantity import BareQuantity as FastQ

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


class AbstractMassField(AbstractField):
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
        self: "AbstractMassField", _: dfx.AbstractSolver, /
    ) -> PyTree[dfx.AbstractTerm]:
        """Return diffeq terms for integration.

        Examples
        --------
        >>> import diffrax as dfx
        >>> import galax.dynamics as gd

        >>> field = gd.cluster.ConstantMass()
        >>> field.terms(dfx.Dopri8())
        ODETerm(
            vector_field=_JitWrapper( fn='ConstantMass.__call__', ... ) )

        """
        return dfx.ODETerm(eqx.filter_jit(self.__call__))


#####################################################


class CustomMassField(AbstractMassField):
    """User-defined mass field.

    This takes a user-defined function of type
    `galax.dynamics.cluster.MassVectorField`.

    """

    #: User-defined mass derivative function of type
    #: `galax.dynamics.cluster.MassVectorField`
    mass_deriv: MassVectorField

    _: KW_ONLY

    def __call__(
        self, t: Time, Mc: ClusterMass, args: FieldArgs, /, **kwargs: Any
    ) -> Array:
        return self.mass_deriv(t, Mc, args, **kwargs)


#####################################################


class ConstantMass(AbstractMassField):
    """Constant mass field.

    This is a constant mass field.

    """

    def __call__(
        self,
        t: Time,  # noqa: ARG002
        Mc: ClusterMass,
        args: FieldArgs,  # noqa: ARG002
        /,
        **kwargs: Any,  # noqa: ARG002
    ) -> Array:
        return jnp.zeros_like(Mc)


######################################################


class Baumgardt1998MassLoss(AbstractMassField):
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

    >>> mass_loss = gdc.Baumgardt1998MassLoss()
    >>> mass_loss.tidal_radius_flag
    <class 'galax.dynamics...King1962'>
    >>> mass_loss.relaxation_time_flag
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

    >>> mass_loss(0, u.Quantity(1e4, "Msun"), kwargs)  # [Msun/Myr]
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
