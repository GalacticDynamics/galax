"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
]

from typing import TypeAlias

import numpy as np
from astropy.coordinates import BaseRepresentation as APYRepresentation
from astropy.units import Quantity as APYQuantity
from jaxtyping import Array, Float, Shaped
from plum import dispatch

import coordinax as cx
import quaxed.array_api as xp
import quaxed.numpy as qnp
from unxt import Quantity

import galax.coordinates as gc
import galax.typing as gt
from .base import AbstractPotentialBase
from .utils import _convert_from_3dvec, parse_to_quantity
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims

QMatrix33: TypeAlias = Float[Quantity, "3 3"]
BatchQMatrix33: TypeAlias = Shaped[QMatrix33, "*batch"]
HessianVec: TypeAlias = Shaped[Quantity["1/s^2"], "*#shape 3 3"]  # TODO: shape -> batch

# Position and time input options
PositionalLike: TypeAlias = (
    cx.AbstractPosition3D
    | gt.LengthBroadBatchVec3
    | Shaped[Quantity, "*#batch 3"]
    | Shaped[Array, "*#batch 3"]
)
TimeOptions: TypeAlias = (
    gt.BatchRealQScalar
    | gt.FloatQScalar
    | gt.IntQScalar
    | gt.BatchableRealScalarLike
    | gt.FloatScalar
    | gt.IntScalar
    | int  # .e.g. 0
    | APYQuantity
)

# =============================================================================
# Potential Energy


@dispatch  # type: ignore[misc]
def potential(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
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

    >>> pot.potential(w)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=Quantity([0, 1], "Gyr"))
    >>> pot.potential(w)
    Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a
    :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
    we can instead pass a :class:`~coordinax.FourVector`:

    >>> from coordinax import FourVector
    >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
    >>> pot.potential(w)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')
    """  # noqa: E501
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return potential._potential(q, pspt.t)  # noqa: SLF001


_potential = potential  # Needed to bypass namespace restrictions


@dispatch
def potential(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
    :class:`unxt.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = Quantity([1., 2, 3], "kpc")
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    Again, this can be batched.  If the input position object has no units
    (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
    as the potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')
    """  # noqa: E501
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._potential(q, t)  # noqa: SLF001


@dispatch
def potential(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.potential(q, t=t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    See the other examples in the positional-only case.
    """
    return _potential(potential, q, t)


@dispatch
def potential(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity | np.ndarray,
    /,
    t: TimeOptions,
) -> Quantity["specific energy"]:  # TODO: shape hint
    """Compute the potential energy at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.potential` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array([-0.55372734, -0.46647294], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a
    :class:`astropy.coordinates.CartesianRepresentation`,
    we can instead pass a :class:`astropy.units.Quantity`, which is
    interpreted as a Cartesian position:

    >>> q = u.Quantity([1, 2, 3], "kpc")
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    Again, this can be batched.  Also, If the input position object has no
    units (i.e. is an `~numpy.ndarray`), it is assumed to be in the same
    unit system as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.potential(q, t)
    Quantity['specific energy'](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')
    """  # noqa: E501
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._potential(q, t)  # noqa: SLF001


@dispatch
def potential(
    potential: AbstractPotentialBase,
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
    >>> pot.potential(q, t=t)
    Quantity['specific energy'](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    See the other examples in the positional-only case.
    """
    return _potential(potential, q, t)


# =============================================================================
# Gradient


@dispatch  # type: ignore[misc]
def gradient(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
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
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return potential._gradient(q, pspt.t)  # noqa: SLF001


_gradient = gradient  # Needed to bypass namespace restrictions


@dispatch
def gradient(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
) -> Quantity["acceleration"]:  # TODO: shape hint
    """Compute the gradient of the potential at the given position(s).

    Parameters
    ----------
    q : :class:`vector.AbstractPosition3D` | (Quantity|Array)[float, (*batch, 3)]
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64), unit='kpc / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                    [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._gradient(q, t)  # noqa: SLF001


@dispatch
def gradient(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    We can also compute the gradient at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                    [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    return _gradient(potential, q, t)


@dispatch
def gradient(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    t: TimeOptions,
) -> Quantity["acceleration"]:  # TODO: shape hint
    """Compute the gradient at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.gradient` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.gradient(q, t)
    Quantity['acceleration'](Array([[0.08587681, 0.17175361, 0.25763042],
                                    [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])  # TODO: value
    return potential._gradient(q, t)  # noqa: SLF001


@dispatch
def gradient(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    *,
    t: TimeOptions,
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.gradient(q, t=t)
    Quantity['acceleration'](Array([0.08587681, 0.17175361, 0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    See the other examples in the positional-only case.
    """
    return _gradient(potential, q, t)


# =============================================================================
# Laplacian


@dispatch  # type: ignore[misc]
def laplacian(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
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
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return potential._laplacian(q, pspt.t)  # noqa: SLF001


_laplacian = laplacian  # Needed to bypass namespace restrictions


@dispatch
def laplacian(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
) -> Quantity["1/s^2"]:  # TODO: shape hint
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    q : :class:`vector.AbstractPosition3D` | (Quantity|Array)[float, (*batch, 3)]
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._laplacian(q, t)  # noqa: SLF001


@dispatch
def laplacian(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the laplacian at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    return _laplacian(potential, q, t)


@dispatch
def laplacian(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    t: TimeOptions,
) -> Quantity["1/s^2"]:  # TODO: shape hint
    """Compute the laplacian at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.laplacian` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._laplacian(q, t)  # noqa: SLF001


@dispatch
def laplacian(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    *,
    t: TimeOptions,
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.laplacian(q, t=t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    See the other examples in the positional-only case.
    """
    return _laplacian(potential, q, t)


# =============================================================================
# Density


@dispatch  # type: ignore[misc]
def density(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
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
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=Quantity([0, 1], "Gyr"))
    >>> pot.density(w)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a
    :class:`~galax.coordinates.AbstractPhaseSpacePosition`,
    we can instead pass a :class:`~vector.FourVector`:

    >>> from coordinax import FourVector
    >>> w = FourVector(q=Quantity([1, 2, 3], "kpc"), t=Quantity(0, "Gyr"))
    >>> pot.density(w)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')
    """
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return potential._density(q, pspt.t)  # noqa: SLF001


_density = density  # Needed to bypass namespace restrictions


@dispatch
def density(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
    :class:`unxt.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = Quantity([1., 2, 3], "kpc")
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  If the input position object has no units
    (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
    as the potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')
    """
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._density(q, t)  # noqa: SLF001


@dispatch
def density(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.density(q, t=t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    See the other examples in the positional-only case.
    """
    return _density(potential, q, t)


@dispatch
def density(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity | np.ndarray,
    /,
    t: TimeOptions,
) -> Quantity["mass density"]:  # TODO: shape hint
    """Compute the density at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.density` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a
    :class:`astropy.coordinates.CartesianRepresentation`,
    we can instead pass a :class:`astropy.units.Quantity`, which is
    interpreted as a Cartesian position:

    >>> q = u.Quantity([1, 2, 3], "kpc")
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  Also, If the input position object has no
    units (i.e. is an `~numpy.ndarray`), it is assumed to be in the same
    unit system as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')
    """
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._density(q, t)  # noqa: SLF001


@dispatch
def density(
    potential: AbstractPotentialBase,
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
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    See the other examples in the positional-only case.
    """
    return _density(potential, q, t)


# =============================================================================
# Hessian


@dispatch  # type: ignore[misc]
def hessian(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
    /,
) -> BatchQMatrix33:
    """Compute the hessian of the potential at the given position(s).

    Parameters
    ----------
    pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
        The phase-space + time position to compute the hessian of the
        potential.

    Returns
    -------
    H : BatchQMatrix33
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
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return potential._hessian(q, pspt.t)  # noqa: SLF001


_hessian = hessian  # Needed to bypass namespace restrictions


@dispatch
def hessian(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.hessian(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the hessian at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.hessian(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    """
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._hessian(q, t)  # noqa: SLF001


@dispatch
def hessian(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.hessian(q, t=t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    See the other examples in the positional-only case.
    """
    return _hessian(potential, q, t)


@dispatch
def hessian(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity | np.ndarray,
    /,
    t: TimeOptions,
) -> HessianVec:
    """Compute the hessian at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.hessian` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return potential._hessian(q, t)  # noqa: SLF001


@dispatch
def hessian(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity | np.ndarray,
    /,
    *,
    t: TimeOptions,
) -> HessianVec:
    return _hessian(potential, q, t)


# =============================================================================
# Acceleration


@dispatch  # type: ignore[misc]
def acceleration(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
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
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return -potential._gradient(q, pspt.t)  # noqa: SLF001


_acceleration = acceleration  # needed to bypass namespace restrictions


@dispatch
def acceleration(
    potential: AbstractPotentialBase, q: PositionalLike, /, t: TimeOptions
) -> Quantity["acceleration"]:  # TODO: shape hint
    """Compute the acceleration due to the potential at the given position(s).

    Parameters
    ----------
    q : :class:`vector.AbstractPosition3D` | (Quantity|Array)[float, (*batch, 3)]
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                    [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return -potential._gradient(q, t)  # noqa: SLF001


@dispatch
def acceleration(
    potential: AbstractPotentialBase, q: PositionalLike, /, *, t: TimeOptions
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    We can also compute the acceleration at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                    [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    return _acceleration(potential, q, t)


@dispatch
def acceleration(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    t: TimeOptions,
) -> Quantity["acceleration"]:  # TODO: shape hint
    """Compute the acceleration at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.acceleration` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.acceleration(q, t)
    Quantity['acceleration'](Array([[-0.08587681, -0.17175361, -0.25763042],
                                    [-0.02663127, -0.03328908, -0.0399469 ]], dtype=float64),
                                unit='kpc / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
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
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return -potential._gradient(q, t)  # noqa: SLF001


@dispatch
def acceleration(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    *,
    t: TimeOptions,
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

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.acceleration(q, t=t)
    Quantity['acceleration'](Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64),
                                unit='kpc / Myr2')

    See the other examples in the positional-only case.
    """  # noqa: E501
    return _acceleration(potential, q, t)


# =============================================================================
# Tidal Tensor


@dispatch(precedence=1)  # type: ignore[misc]
def tidal_tensor(
    potential: AbstractPotentialBase, q: gt.BatchQVec3, /, t: gt.BatchRealQScalar
) -> BatchQMatrix33:
    """Compute the tidal tensor.

    See https://en.wikipedia.org/wiki/Tidal_tensor

    .. note::

        This is in cartesian coordinates with a Euclidean metric.
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
    J = hessian(potential, q, t)  # (*batch, 3, 3)
    batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
    traced = (
        expand_batch_dims(xp.eye(3), ndim=len(batch_shape))
        * expand_arr_dims(qnp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
        / 3
    )
    return J - traced


_tidal_tensor = tidal_tensor  # Needed to bypass namespace restrictions


@dispatch
def tidal_tensor(
    potential: AbstractPotentialBase,
    pspt: gc.AbstractPhaseSpacePosition | cx.FourVector,
    /,
) -> BatchQMatrix33:
    """Compute the tidal tensor of the potential at the given position(s).

    Parameters
    ----------
    pspt : :class:`~galax.coordinates.AbstractPhaseSpacePosition`
        The phase-space + time position to compute the tidal tensor of the
        potential.

    Returns
    -------
    T : BatchQMatrix33
        The tidal tensor matrix of the potential.

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

    >>> pot.tidal_tensor(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=Quantity([0, 1], "Gyr"))
    >>> pot.tidal_tensor(w)
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
    >>> pot.tidal_tensor(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')
    """
    q = _convert_from_3dvec(pspt.q, units=potential.units)
    return _tidal_tensor(potential, q, pspt.t)


@dispatch
def tidal_tensor(
    potential: AbstractPotentialBase,
    q: PositionalLike,
    /,
    t: TimeOptions,
) -> BatchQMatrix33:
    """Compute the tidal tensor of the potential at the given position(s).

    Parameters
    ----------
    q : PositionalLike
        The position to compute the tidal tensor of the potential.  If unitless
        (i.e. is an `~jax.Array`), it is assumed to be in the unit system of the
        potential.
    t : TimeOptions
        The time at which to compute the tidal tensor of the potential.  If
        unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
        system of the potential.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

    We can compute the tidal tensor at a position (and time, if any
    parameters are time-dependent):

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions:

    >>> q = cx.CartesianPosition3D.constructor(Quantity([[1, 2, 3], [4, 5, 6]], "kpc"))
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a :class:`~vector.AbstractPosition3D` (in this case a
    :class:`~vector.CartesianPosition3D`), we can instead pass a
    :class:`unxt.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = Quantity([1., 2, 3], "kpc")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units
    (i.e. is an `~jax.Array`), it is assumed to be in the same unit system
    as the potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')
    """
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return _tidal_tensor(potential, q, t)


@dispatch
def tidal_tensor(
    potential: AbstractPotentialBase,
    q: PositionalLike,
    /,
    *,
    t: TimeOptions,
) -> BatchQMatrix33:
    """Compute the tidal tensor when `t` is keyword-only.

    Examples
    --------
    All these examples are covered by the case where `t` is positional.
    :mod:`plum` dispatches on positional arguments only, so it necessary
    to redispatch here.

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=Quantity(1e12, "Msun"), units="galactic")

    >>> q = cx.CartesianPosition3D.constructor(Quantity([1, 2, 3], "kpc"))
    >>> t = Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t=t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    See the other examples in the positional-only case.
    """
    return _tidal_tensor(potential, q, t)


@dispatch
def tidal_tensor(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    t: TimeOptions,
) -> BatchQMatrix33:
    """Compute the tidal tensor at the given position(s).

    :meth:`~galax.potential.AbstractPotentialBase.tidal_tensor` also
    supports Astropy objects, like
    :class:`astropy.coordinates.BaseRepresentation` and
    :class:`astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts :class:`~vector.AbstractPosition3D` and
    :class:`~unxt.Quantity`.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.coordinates as c
    >>> import astropy.units as u
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the tidal tensor at a position (and time, if any
    parameters are time-dependent):

    >>> q = c.CartesianRepresentation([1, 2, 3], unit=u.kpc)
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions:

    >>> q = c.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit=u.kpc)
    >>> pot.tidal_tensor(q, t)
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
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.
    """
    q = parse_to_quantity(q, unit=potential.units["length"])
    t = Quantity.constructor(t, potential.units["time"])
    return _tidal_tensor(potential, q, t)


@dispatch
def tidal_tensor(
    potential: AbstractPotentialBase,
    q: APYRepresentation | APYQuantity,
    /,
    *,
    t: TimeOptions,
) -> BatchQMatrix33:
    return _tidal_tensor(potential, q, t)
