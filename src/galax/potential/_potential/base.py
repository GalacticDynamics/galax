__all__ = ["AbstractPotentialBase"]

import abc
from dataclasses import KW_ONLY, fields
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import G as _CONST_G  # pylint: disable=no-name-in-module
from astropy.coordinates import BaseRepresentation as APYRepresentation
from astropy.units import Quantity as APYQuantity
from jaxtyping import Array, Float, Shaped
from plum import dispatch

import quaxed.array_api as xp
import quaxed.numpy as qnp
import unxt
from coordinax import Abstract3DVector, FourVector
from unxt import AbstractUnitSystem, Quantity

import galax.typing as gt
from .utils import _convert_from_3dvec, parse_to_quantity
from galax.coordinates import AbstractPhaseSpacePosition, PhaseSpacePosition
from galax.potential._potential.param.attr import ParametersAttribute
from galax.potential._potential.param.utils import all_parameters
from galax.utils._collections import ImmutableDict
from galax.utils._jax import vectorize_method
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims
from galax.utils.dataclasses import ModuleMeta

if TYPE_CHECKING:
    from galax.dynamics._dynamics.integrate._api import Integrator
    from galax.dynamics._dynamics.orbit import Orbit


BatchRealQScalar: TypeAlias = Shaped[gt.RealQScalar, "*batch"]
QMatrix33: TypeAlias = Float[Quantity, "3 3"]
BatchMatrix33: TypeAlias = Shaped[Float[Array, "3 3"], "*batch"]
BatchQMatrix33: TypeAlias = Shaped[QMatrix33, "*batch"]
HessianVec: TypeAlias = Shaped[Quantity["1/s^2"], "*#shape 3 3"]  # TODO: shape -> batch

# Position and time input options
PositionalLike: TypeAlias = (
    Abstract3DVector | gt.LengthBroadBatchVec3 | Shaped[Array, "*#batch 3"]
)
TimeOptions: TypeAlias = (
    BatchRealQScalar
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
        type(cls).__setattr__(
            cls,
            "parameters",
            ParametersAttribute(MappingProxyType(all_parameters(cls))),
        )

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
    def _potential_energy(
        self, q: gt.QVec3, t: gt.RealQScalar, /
    ) -> Shaped[Quantity["specific energy"], ""]:
        """Compute the potential energy at the given position(s).

        This method MUST be implemented by subclasses.

        It is recommended to both JIT and vectorize this function.
        See ``AbstractPotentialBase.potential_energy`` for an example.

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

    @dispatch
    def potential_energy(
        self: "AbstractPotentialBase",
        pspt: AbstractPhaseSpacePosition | FourVector,
        /,
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the value of the
            potential.

        Returns
        -------
        E : Quantity[float, *batch, 'specific energy']
            The potential energy per unit mass or value of the potential.

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the potential energy:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.potential_energy(w)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        We can also compute the potential energy at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.potential_energy(w)
        Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.potential_energy(w)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')
        """  # noqa: E501
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return self._potential_energy(q, pspt.t)

    @dispatch
    def potential_energy(
        self: "AbstractPotentialBase", q: PositionalLike, /, t: TimeOptions
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the value of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the value of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._potential_energy(q, t)

    @dispatch
    def potential_energy(
        self: "AbstractPotentialBase", q: PositionalLike, /, *, t: TimeOptions
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.potential_energy(q, t=t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        See the other examples in the positional-only case.
        """  # noqa: E501
        return self.potential_energy(q, t)

    @dispatch
    def potential_energy(
        self: "AbstractPotentialBase",
        q: APYRepresentation | APYQuantity | np.ndarray,
        /,
        t: TimeOptions,
    ) -> Quantity["specific energy"]:  # TODO: shape hint
        """Compute the potential energy at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.potential_energy` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.coordinates as c
        >>> import astropy.units as u
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array([-0.55372734, -0.46647294], dtype=float64), unit='kpc2 / Myr2')

        Instead of passing a
        :class:`astropy.coordinates.CartesianRepresentation`,
        we can instead pass a :class:`astropy.units.Quantity`, which is
        interpreted as a Cartesian position:

        >>> q = u.Quantity([1, 2, 3], "kpc")
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        Again, this can be batched.  Also, If the input position object has no
        units (i.e. is an `~numpy.ndarray`), it is assumed to be in the same
        unit system as the potential.

        >>> q = np.array([[1, 2, 3], [4, 5, 6]])
        >>> pot.potential_energy(q, t)
        Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._potential_energy(q, t)

    @dispatch
    def potential_energy(
        self: "AbstractPotentialBase",
        q: APYRepresentation | APYQuantity | np.ndarray,
        /,
        *,
        t: TimeOptions,
    ) -> Float[Quantity["specific energy"], "*batch"]:
        """Compute the potential energy when `t` is keyword-only.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.coordinates as c
        >>> import astropy.units as u
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.potential_energy(q, t=t)
        Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

        See the other examples in the positional-only case.
        """  # noqa: E501
        return self.potential_energy(q, t)

    @partial(jax.jit)
    def __call__(
        self, q: gt.LengthBatchVec3, /, t: gt.BatchableRealQScalar
    ) -> Float[Quantity["specific energy"], "*batch"]:
        """Compute the potential energy at the given position(s).

        Parameters
        ----------
        q : Quantity[float, (*batch, 3), 'length']
            The position to compute the value of the potential.
        t : Array[float | int, *batch] | float | int
            The time at which to compute the value of the potential.

        Returns
        -------
        E : Quantity[float, (*batch,), 'specific energy']
            The potential energy per unit mass or value of the potential.

        See Also
        --------
        :meth:`galax.potential.AbstractPotentialBase.potential_energy`
        """
        return self.potential_energy(q, t)

    # ---------------------------------------
    # Gradient

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3)")
    def _gradient(self, q: gt.BatchQVec3, /, t: gt.RealQScalar) -> gt.BatchQVec3:
        """See ``gradient``."""
        grad_op = unxt.experimental.grad(
            self._potential_energy, units=(self.units["length"], self.units["time"])
        )
        return grad_op(q, t)

    @dispatch
    def gradient(
        self: "AbstractPotentialBase",
        pspt: AbstractPhaseSpacePosition | FourVector,
        /,
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the gradient.

        Returns
        -------
        grad : Quantity[float, *batch, 'acceleration']
            The gradient of the potential.

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the potential energy:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.gradient(w)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.gradient(w)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.gradient(w)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return self._gradient(q, pspt.t)

    @dispatch
    def gradient(
        self, q: PositionalLike, /, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`vector.Abstract3DVector` | (Quantity|Array)[float, (*batch, 3)]
            The position to compute the gradient of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : Array[float | int, *batch] | float | int
            The time at which to compute the gradient of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64), unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1., 2, 3], [4, 5, 6]])
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._gradient(q, t)

    @dispatch
    def gradient(
        self, q: PositionalLike, /, *, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the gradient of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the gradient of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the gradient at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the gradient at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        return self.gradient(q, t)

    @dispatch
    def gradient(
        self, q: APYRepresentation | APYQuantity, /, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.gradient` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Parameters
        ----------
        q : PositionalLike
            The position to compute the value of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the value of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.gradient(q, t)
        Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                        [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])  # TODO: value
        return self._gradient(q, t)

    @dispatch
    def gradient(
        self, q: APYRepresentation | APYQuantity, /, *, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the gradient when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.gradient(q, t=t)
        Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        See the other examples in the positional-only case.
        """  # noqa: E501
        return self.gradient(q, t)

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

    @dispatch
    def laplacian(
        self: "AbstractPotentialBase",
        pspt: AbstractPhaseSpacePosition | FourVector,
        /,
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian of the potential at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the laplacian.

        Returns
        -------
        grad : Quantity[float, *batch, 'acceleration']
            The laplacian of the potential.

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the potential energy:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.laplacian(w)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        We can also compute the potential energy at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.laplacian(w)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.laplacian(w)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')
        """  # noqa: E501
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return self._laplacian(q, pspt.t)

    @dispatch
    def laplacian(
        self, q: PositionalLike, /, t: TimeOptions
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian of the potential at the given position(s).

        Parameters
        ----------
        q : :class:`vector.Abstract3DVector` | (Quantity|Array)[float, (*batch, 3)]
            The position to compute the laplacian of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : Array[float | int, *batch] | float | int
            The time at which to compute the laplacian of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._laplacian(q, t)

    @dispatch
    def laplacian(
        self, q: PositionalLike, /, *, t: TimeOptions
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the laplacian of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the laplacian of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the laplacian at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        We can also compute the laplacian at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')
        """  # noqa: E501
        return self.laplacian(q, t)

    @dispatch
    def laplacian(
        self, q: APYRepresentation | APYQuantity, /, t: TimeOptions
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.laplacian` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Parameters
        ----------
        q : PositionalLike
            The position to compute the value of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the value of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.laplacian(q, t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.laplacian(q, t)
        Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._laplacian(q, t)

    @dispatch
    def laplacian(
        self, q: APYRepresentation | APYQuantity, /, *, t: TimeOptions
    ) -> Quantity["1/s^2"]:  # TODO: shape hint
        """Compute the laplacian when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.laplacian(q, t=t)
        Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

        See the other examples in the positional-only case.
        """
        return self.laplacian(q, t)

    # ---------------------------------------
    # Density

    @partial(jax.jit)
    def _density(
        self, q: gt.BatchQVec3, /, t: BatchRealQScalar | gt.RealQScalar
    ) -> gt.BatchFloatQScalar:
        """See ``density``."""
        # Note: trace(jacobian(gradient)) is faster than trace(hessian(energy))
        return self._laplacian(q, t) / (4 * xp.pi * self.constants["G"])

    @dispatch
    def density(
        self: "AbstractPotentialBase",
        pspt: AbstractPhaseSpacePosition | FourVector,
        /,
    ) -> Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the density.

        Returns
        -------
        rho : Quantity[float, *batch, 'mass density']
            The density of the potential at the given position(s).

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the density:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.density(w)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        We can also compute the density at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.density(w)
        Quantity['mass density'](Array([4.90989768e-07, 0.00000000e+00], dtype=float64), unit='solMass / kpc3')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.density(w)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')
        """  # noqa: E501
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return self._density(q, pspt.t)

    @dispatch
    def density(
        self, q: PositionalLike, /, t: TimeOptions
    ) -> Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the density of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the density of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the density at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.density(q, t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        We can also compute the density at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.density(q, t)
        Quantity['mass density'](Array([4.90989768e-07, 0.00000000e+00], dtype=float64), unit='solMass / kpc3')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.density(q, t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.density(q, t)
        Quantity['mass density'](Array([4.90989768e-07, 0.00000000e+00], dtype=float64), unit='solMass / kpc3')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._density(q, t)

    @dispatch
    def density(
        self, q: PositionalLike, /, *, t: TimeOptions
    ) -> Quantity["mass density"]:
        """Compute the density when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.density(q, t=t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        See the other examples in the positional-only case.
        """  # noqa: E501
        return self.density(q, t)

    @dispatch
    def density(
        self: "AbstractPotentialBase",
        q: APYRepresentation | APYQuantity | np.ndarray,
        /,
        t: TimeOptions,
    ) -> Quantity["mass density"]:  # TODO: shape hint
        """Compute the density at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.density` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.coordinates as c
        >>> import astropy.units as u
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

        We can compute the density at a position (and time, if any
        parameters are time-dependent):

        >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.density(q, t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        We can also compute the density at multiple positions:

        >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
        >>> pot.density(q, t)
        Quantity['mass density'](Array([3.06868605e-08, 1.53434303e-08], dtype=float64),
                                 unit='solMass / kpc3')

        Instead of passing a
        :class:`astropy.coordinates.CartesianRepresentation`,
        we can instead pass a :class:`astropy.units.Quantity`, which is
        interpreted as a Cartesian position:

        >>> q = u.Quantity([1, 2, 3], "kpc")
        >>> pot.density(q, t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64), unit='solMass / kpc3')

        Again, this can be batched.  Also, If the input position object has no
        units (i.e. is an `~numpy.ndarray`), it is assumed to be in the same
        unit system as the potential.

        >>> q = np.array([[1, 2, 3], [4, 5, 6]])
        >>> pot.density(q, t)
        Quantity['mass density'](Array([4.90989768e-07, 0.00000000e+00], dtype=float64),
                                 unit='solMass / kpc3')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._density(q, t)

    @dispatch
    def density(
        self: "AbstractPotentialBase",
        q: APYRepresentation | APYQuantity | np.ndarray,
        /,
        *,
        t: TimeOptions,
    ) -> Quantity["mass density"]:  # TODO: shape hint
        """Compute the density when `t` is keyword-only.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.coordinates as c
        >>> import astropy.units as u
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

        We can compute the density at a position (and time, if any
        parameters are time-dependent):

        >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.density(q, t=t)
        Quantity['mass density'](Array(4.90989768e-07, dtype=float64),
                                 unit='solMass / kpc3')

        See the other examples in the positional-only case.
        """
        return self.density(q, t)

    # ---------------------------------------
    # Hessian

    @partial(jax.jit)
    @vectorize_method(signature="(3),()->(3,3)")
    def _hessian(self, q: gt.QVec3, /, t: gt.RealQScalar) -> QMatrix33:
        """See ``hessian``."""
        hess_op = unxt.experimental.hessian(
            self._potential_energy, units=(self.units["length"], self.units["time"])
        )
        return hess_op(q, t)

    @dispatch
    def hessian(
        self: "AbstractPotentialBase", pspt: AbstractPhaseSpacePosition | FourVector, /
    ) -> BatchQMatrix33:
        """Compute the hessian of the potential at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the hessian of the
            potential.

        Returns
        -------
        H : BatchMatrix33
            The hessian matrix of the potential.

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the hessian:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.hessian(w)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        We can also compute the potential energy at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.hessian(w)
        Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                              [-0.03680435,  0.01226812, -0.11041304],
                              [-0.05520652, -0.11041304, -0.07974275]],
                             [[ 0.00250749, -0.00518791, -0.00622549],
                              [-0.00518791,  0.00017293, -0.00778186],
                              [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                      unit='1 / Myr2')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.hessian(w)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')
        """
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return self._hessian(q, pspt.t)

    @dispatch
    def hessian(
        self: "AbstractPotentialBase", q: PositionalLike, /, t: TimeOptions
    ) -> HessianVec:
        """Compute the hessian of the potential at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the hessian of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the hessian of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the hessian at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.hessian(q, t)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        We can also compute the hessian at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.hessian(q, t)
        Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                              [-0.03680435,  0.01226812, -0.11041304],
                              [-0.05520652, -0.11041304, -0.07974275]],
                             [[ 0.00250749, -0.00518791, -0.00622549],
                              [-0.00518791,  0.00017293, -0.00778186],
                              [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                      unit='1 / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.hessian(q, t)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.hessian(q, t)
        Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                              [-0.03680435,  0.01226812, -0.11041304],
                              [-0.05520652, -0.11041304, -0.07974275]],
                             [[ 0.00250749, -0.00518791, -0.00622549],
                              [-0.00518791,  0.00017293, -0.00778186],
                              [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                      unit='1 / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._hessian(q, t)

    @dispatch
    def hessian(
        self: "AbstractPotentialBase", q: PositionalLike, /, *, t: TimeOptions
    ) -> HessianVec:
        """Compute the hessian when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.hessian(q, t=t)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        See the other examples in the positional-only case.
        """
        return self.hessian(q, t)

    @dispatch
    def hessian(
        self, q: APYRepresentation | APYQuantity | np.ndarray, /, t: TimeOptions
    ) -> HessianVec:
        """Compute the hessian at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.hessian` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Examples
        --------
        >>> import numpy as np
        >>> import astropy.coordinates as c
        >>> import astropy.units as u
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

        We can compute the hessian at a position (and time, if any
        parameters are time-dependent):

        >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
        >>> t = u.Quantity(0, "Gyr")
        >>> pot.hessian(q, t)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        We can also compute the hessian at multiple positions:

        >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
        >>> pot.hessian(q, t)
        Quantity[...](Array([[[ 0.00800845, -0.00152542, -0.00266948],
                              [-0.00152542,  0.00228813, -0.01067794],
                              [-0.00266948, -0.01067794, -0.01029658]],
                             [[ 0.00436863, -0.00161801, -0.00258882],
                              [-0.00161801,  0.00097081, -0.00647205],
                              [-0.00258882, -0.00647205, -0.00533944]]], dtype=float64),
                      unit='1 / Myr2')

        Instead of passing a
        :class:`astropy.coordinates.CartesianRepresentation`,
        we can instead pass a :class:`astropy.units.Quantity`, which is
        interpreted as a Cartesian position:

        >>> q = u.Quantity([1, 2, 3], "kpc")
        >>> pot.hessian(q, t)
        Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                             [-0.03680435,  0.01226812, -0.11041304],
                             [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                      unit='1 / Myr2')

        Again, this can be batched.  Also, If the input position object has no
        units (i.e. is an `~numpy.ndarray`), it is assumed to be in the same
        unit system as the potential.

        >>> q = np.array([[1, 2, 3], [4, 5, 6]])
        >>> pot.hessian(q, t)
        Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                              [-0.03680435,  0.01226812, -0.11041304],
                              [-0.05520652, -0.11041304, -0.07974275]],
                             [[ 0.00250749, -0.00518791, -0.00622549],
                              [-0.00518791,  0.00017293, -0.00778186],
                              [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                      unit='1 / Myr2')
        """
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return self._hessian(q, t)

    @dispatch
    def hessian(
        self, q: APYRepresentation | APYQuantity | np.ndarray, /, *, t: TimeOptions
    ) -> HessianVec:
        return self.hessian(q, t)

    ###########################################################################
    # Convenience methods

    # ---------------------------------------
    # Acceleration

    @dispatch
    def acceleration(
        self: "AbstractPotentialBase",
        pspt: AbstractPhaseSpacePosition | FourVector,
        /,
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
            The phase-space + time position to compute the acceleration.

        Returns
        -------
        grad : Quantity[float, *batch, 'acceleration']
            The acceleration of the potential.

        Examples
        --------
        For this example we will use a simple potential, the Kepler potential.

        First some imports:

        >>> from unxt import Quantity
        >>> import galax.potential as gp
        >>> import galax.coordinates as gc

        Then we can construct a potential and compute the potential energy:

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> w = gc.PhaseSpacePosition(q=Quantity([1, 2, 3], "kpc"),
        ...                           p=Quantity([4, 5, 6], "km/s"),
        ...                           t=Quantity(0, "Gyr"))

        >>> pot.acceleration(w)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions and times:

        >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
        ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
        ...                           t=Quantity([0, 1], "Gyr"))
        >>> pot.acceleration(w)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a
        :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
        we can instead pass a :class:`~vector.FourVector`:

        >>> from coordinax import FourVector
        >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
        >>> pot.acceleration(w)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = _convert_from_3dvec(pspt.q, units=self.units)
        return -self._gradient(q, pspt.t)

    @dispatch
    def acceleration(
        self, q: PositionalLike, /, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : :class:`vector.Abstract3DVector` | (Quantity|Array)[float, (*batch, 3)]
            The position to compute the acceleration of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : Array[float | int, *batch] | float | int
            The time at which to compute the acceleration of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return -self._gradient(q, t)

    @dispatch
    def acceleration(
        self, q: PositionalLike, /, *, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration at the given position(s).

        Parameters
        ----------
        q : PositionalLike
            The position to compute the acceleration of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the acceleration of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the acceleration at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the acceleration at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        return self.acceleration(q, t)

    @dispatch
    def acceleration(
        self, q: APYRepresentation | APYQuantity, /, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration at the given position(s).

        :meth:`~galax.potential.AbstractPotentialBase.acceleration` also
        supports Astropy objects, like
        :class:`astropy.coordinates.BaseRepresentation` and
        :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
        counterparts :class:`~vector.Abstract3DVector` and
        :class:`~unxt.Quantity`.

        Parameters
        ----------
        q : PositionalLike
            The position to compute the value of the potential.  If unitless
            (i.e. is an `~jax.Array`), it is assumed to be in the unit system of
            the potential.
        t : TimeOptions
            The time at which to compute the value of the potential.  If
            unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
            system of the potential.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        We can compute the potential energy at a position (and time, if any
        parameters are time-dependent):

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        We can also compute the potential energy at multiple positions:

        >>> q = cx.Cartesian3DVector.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')

        Instead of passing a :class:`~vector.Abstract3DVector` (in this case a
        :class:`~vector.Cartesian3DVector`), we can instead pass a
        :class:`unxt.Quantity`, which is interpreted as a Cartesian
        position:

        >>> q = Quantity([1., 2, 3], "kpc")
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        Again, this can be batched.  If the input position object has no units
        (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
        as the potential.

        >>> import jax.numpy as jnp
        >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        >>> pot.acceleration(q, t)
        Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                        [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                 unit='kpc / Myr2')
        """  # noqa: E501
        q = parse_to_quantity(q, unit=self.units["length"])
        t = Quantity.constructor(t, self.units["time"])
        return -self._gradient(q, t)

    @dispatch
    def acceleration(
        self, q: APYRepresentation | APYQuantity, /, *, t: TimeOptions
    ) -> Quantity["acceleration"]:  # TODO: shape hint
        """Compute the acceleration when `t` is keyword-only.

        Examples
        --------
        All these examples are covered by the case where `t` is positional.
        :mod:`plum` dispatches on positional arguments only, so it necessary
        to redispatch here.

        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> import galax.potential as gp

        >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

        >>> q = cx.Cartesian3DVector.constructor(Quantity([1, 2, 3], "kpc"))
        >>> t = Quantity(0, "Gyr")
        >>> pot.acceleration(q, t=t)
        Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                 unit='kpc / Myr2')

        See the other examples in the positional-only case.
        """  # noqa: E501
        return self.acceleration(q, t)

    # ---------------------------------------
    # Tidal tensor

    @partial(jax.jit)
    def tidal_tensor(self, q: gt.BatchQVec3, /, t: BatchRealQScalar) -> BatchMatrix33:
        """Compute the tidal tensor.

        See https://en.wikipedia.org/wiki/Tidal_tensor

        .. note::

            This is in cartesian coordinates with the Euclidean metric tensor.
            Also, this isn't correct for GR.

        Parameters
        ----------
        q : Quantity[float, (*batch, 3,), 'length']
            Position to compute the tidal tensor at.
        t : Quantity[float | int, (*batch,), 'time']
            Time at which to compute the tidal tensor.

        Returns
        -------
        Quantity[float, (*batch, 3, 3), '1/time^2']
            The tidal tensor.
        """
        J = self.hessian(q, t)  # (*batch, 3, 3)
        batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
        traced = (
            expand_batch_dims(xp.eye(3), ndim=len(batch_shape))
            * expand_arr_dims(qnp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
            / 3
        )
        return J - traced

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
