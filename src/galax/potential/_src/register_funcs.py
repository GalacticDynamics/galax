"""galax: Galactic Dynamix in Jax."""

__all__: list[str] = []

from functools import partial
from typing import Any, TypeAlias

import jax
from jaxtyping import Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.typing as gt
from . import api
from .base import AbstractPotential
from .utils import parse_to_quantity
from galax.utils._shape import batched_shape, expand_arr_dims, expand_batch_dims

# TODO: shape -> batch
HessianVec: TypeAlias = Shaped[u.Quantity["1/s^2"], "*#shape 3 3"]

# =============================================================================
# Potential Energy


@dispatch
def potential(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> u.Quantity["specific energy"]:
    """Compute the potential energy at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the value of the potential.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> pot.potential(w)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> pot.potential(w)
    Quantity[...](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~coordinax.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.potential(w)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')
    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    return pot._potential(q, pspt.t)  # noqa: SLF001


@dispatch
def potential(
    pot: AbstractPotential, q: Any, t: Any, /
) -> u.Quantity["specific energy"]:
    """Compute the potential energy at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the value of.
    q : Any
        The position to compute the value of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the value of the potential. See
        :meth:`unxt.Quantity.from_` for more details.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.potential(q, t)
    Quantity[...](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.potential(q, t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.potential(q, t)
    Quantity[...](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    - - -

    `~galax.potential.potential` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and `astropy.units.Quantity`, which
    are interpreted like their jax'ed counterparts `~coordinax.AbstractPos3D`
    and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import numpy as np
    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = apyu.Quantity(0, "Gyr")
    >>> pot.potential(q, t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.potential(q, t)
    Quantity[...](Array([-0.55372734, -0.46647294], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`, we can
    instead pass a `astropy.units.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.potential(q, t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.potential(q, t)
    Quantity[...](Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    .. skip: end

    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._potential(q, t)  # noqa: SLF001


@dispatch
def potential(
    pot: AbstractPotential, q: Any, /, *, t: Any
) -> u.Quantity["specific energy"]:
    """Compute the potential energy when `t` is keyword-only.

    Examples
    --------
    All these examples are covered by the case where `t` is positional.
    :mod:`plum` dispatches on positional arguments only, so it necessary to
    redispatch here. See the other examples in the positional-only case.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(q, t=t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - - -

    In the following example we will show compatibility with Astropy objects.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = 0 * apyu.Gyr
    >>> pot.potential(q, t=t)
    Quantity[...](Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    .. skip: end
    """
    return api.potential(pot, q, t)


# =============================================================================
# Gradient


@dispatch
def gradient(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the gradient.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> print(pot.gradient(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> print(pot.gradient(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> print(pot.gradient(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    q = q.astype(float)  # TODO: better casting
    grad = pot._gradient(q, pspt.t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad)


@dispatch
def gradient(pot: AbstractPotential, q: Any, t: Any, /) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    q : Any
        The position to compute the gradient of the potential. See
        `parse_to_quantity` for more details.

    t : Any
        The time at which to compute the gradient of the potential. See
        :meth:`unxt.Quantity.from_` for more details.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1., 2, 3], [4, 5, 6]])
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    - - -

    `~galax.potential.gradient` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and `astropy.units.Quantity`, which
    are interpreted like their jax'ed counterparts `~coordinax.AbstractPos3D`
    and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = 0 *  apyu.Gyr
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation([[1, 4], [2, 5], [3, 6]], unit="kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    Instead of passing a `~astropy.coordinates.Representation` (in this case a
    `~astropy.coordinates.CartesianRepresentation`), we can instead pass a
    `astropy.units.Quantity`, which is interpreted as a Cartesian position:

    >>> q = [1., 2, 3] * apyu.kpc
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~numpy.ndarray`), it is assumed to be in the same unit system as the
    potential.

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    .. skip: end

    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    q = q.astype(float)  # TODO: better casting
    t = u.Quantity.from_(t, pot.units["time"])
    grad = pot._gradient(q, t)  # noqa: SLF001
    return cx.vecs.CartesianAcc3D.from_(grad)


@dispatch
def gradient(pot: AbstractPotential, q: Any, /, *, t: Any) -> cx.vecs.CartesianAcc3D:
    """Compute the gradient at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the gradient of.
    q : Any
        The position to compute the gradient of the potential.
    t : Any, keyword-only
        The time at which to compute the gradient of the potential.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the gradient at a position (and time, if any parameters are
    time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the gradient at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [0.086 0.172 0.258]>

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    """
    return api.gradient(pot, q, t)


# =============================================================================
# Laplacian


@dispatch
def laplacian(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the laplacian.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> pot.laplacian(w)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> pot.laplacian(w)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.laplacian(w)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')
    """  # noqa: E501
    q = parse_to_quantity(pspt.q, units=pot.units)
    q = q.astype(float)  # TODO: better casting
    return pot._laplacian(q, pspt.t)  # noqa: SLF001


@dispatch
def laplacian(pot: AbstractPotential, q: Any, t: Any, /) -> u.Quantity["1/s^2"]:
    """Compute the laplacian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    q : Any
        The position to compute the laplacian of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the laplacian of the potential.  If
        unitless (i.e. is an `~jax.Array`), it is assumed to be in the unit
        system of the potential.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    - - -

    `~galax.potential.laplacian` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and `astropy.units.Quantity`, which
    are interpreted like their jax'ed counterparts `~coordinax.AbstractPos3D`
    and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import astropy.units as apyu
    >>> import astropy.coordinates as apyc

    We can compute the potential energy at a position (and time, if any
    parameters are time-dependent):

    >>> q = apyc.CartesianRepresentation(apyu.Quantity([1, 2, 3], "kpc"))
    >>> t = 0 * apyu.Gyr
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation(apyu.Quantity([[1, 4], [2, 5], [3, 6]], "kpc"))
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = [1., 2, 3] * apyu.kpc
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is a `~numpy.ndarray`), it is assumed to be in the same unit system as the
    potential.

    >>> import numpy as np
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    .. skip: end
    """  # noqa: E501
    q = parse_to_quantity(q, unit=pot.units["length"])
    q = q.astype(float)  # TODO: better casting
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._laplacian(q, t)  # noqa: SLF001


@dispatch
def laplacian(pot: AbstractPotential, q: Any, /, *, t: Any) -> u.Quantity["1/s^2"]:
    """Compute the laplacian at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the laplacian of.
    q : Any
        The position to compute the laplacian of the potential.
    t : Any, keyword-only
        The time at which to compute the laplacian of the potential.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the laplacian at a position (and time, if any parameters are
    time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the laplacian at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.laplacian(q, t)
    Quantity[...](Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.laplacian(q, t)
    Quantity[...](Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')
    """  # noqa: E501
    return api.laplacian(pot, q, t)


# =============================================================================
# Density


@dispatch
def density(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> u.Quantity["mass density"]:
    """Compute the density at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the density of.
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the density.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the density:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> pot.density(w)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> pot.density(w)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.density(w)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')
    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    return pot._density(q, pspt.t)  # noqa: SLF001


@dispatch
def density(pot: AbstractPotential, q: Any, t: Any, /) -> u.Quantity["mass density"]:
    """Compute the density at the given position(s).

    Parameters
    ----------
    q : Any
        The position to compute the density of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the density of the potential. See
        :meth:`unxt.Quantity.from_` for more details.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the density at a position (and time, if any parameters are
    time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    - - -

    meth:`~galax.potential.AbstractPotential.density` also supports Astropy
    objects, like `astropy.coordinates.BaseRepresentation` and
    `astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts `~coordinax.AbstractPos3D` and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import numpy as np
    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    We can compute the density at a position (and time, if any parameters are
    time-dependent):

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = 0 * apyu.Gyr
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`, we can
    instead pass a `astropy.units.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.density(q, t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.density(q, t)
    Quantity['mass density'](Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    .. skip: end
    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._density(q, t)  # noqa: SLF001


@dispatch
def density(pot: AbstractPotential, q: Any, /, *, t: Any) -> u.Quantity["mass density"]:
    """Compute the density when `t` is keyword-only.

    Examples
    --------
    All these examples are covered by the case where `t` is positional.
    :mod:`plum` dispatches on positional arguments only, so it necessary to
    redispatch here. See the other examples in the positional-only case.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.density(q, t=t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    - - -

    `~galax.potential.density` also supports Astropy objects.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = apyu.Quantity(0, "Gyr")
    >>> pot.density(q, t=t)
    Quantity['mass density'](Array(0., dtype=float64), unit='solMass / kpc3')

    .. skip: end

    """
    return api.density(pot, q, t)


# =============================================================================
# Hessian


@dispatch
def hessian(
    pot: AbstractPotential,
    pspt: gc.AbstractOnePhaseSpacePosition | cx.FourVector,
    /,
) -> gt.BtQuSz33:
    """Compute the hessian of the potential at the given position(s).

    Parameters
    ----------
    pspt : `~galax.coordinates.AbstractOnePhaseSpacePosition`
        The phase-space + time position to compute the hessian of the potential.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the hessian:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> pot.hessian(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> pot.hessian(w)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                  unit='1 / Myr2')

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.hessian(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')
    """
    q = parse_to_quantity(pspt.q, units=pot.units)
    q = q.astype(float)  # TODO: better casting
    return pot._hessian(q, pspt.t)  # noqa: SLF001


@dispatch
def hessian(pot: AbstractPotential, q: Any, t: Any, /) -> HessianVec:
    """Compute the hessian of the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the hessian of.
    q : Any
        The position to compute the hessian of the potential. See
        `parse_to_quantity` for more details.
    t : Any
        The time at which to compute the hessian of the potential. See
        :meth:`~unxt.array.Quantity.from_` for more details.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    We can compute the hessian at a position (and time, if any parameters are
    time-dependent):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.hessian(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the hessian at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.hessian(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.hessian(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

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

    - - -

    `~galax.potential.hessian` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and `astropy.units.Quantity`, which
    are interpreted like their jax'ed counterparts `~coordinax.AbstractPos3D`
    and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import numpy as np
    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    We can compute the hessian at a position (and time, if any parameters are
    time-dependent):

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = 0 * apyu.Gyr
    >>> pot.hessian(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the hessian at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.hessian(q, t)
    Quantity[...](Array([[[ 0.00800845, -0.00152542, -0.00266948],
                          [-0.00152542,  0.00228813, -0.01067794],
                          [-0.00266948, -0.01067794, -0.01029658]],
                          [[ 0.00436863, -0.00161801, -0.00258882],
                          [-0.00161801,  0.00097081, -0.00647205],
                          [-0.00258882, -0.00647205, -0.00533944]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`, we can
    instead pass a `astropy.units.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.hessian(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.hessian(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    .. skip: end
    """
    q = parse_to_quantity(q, unit=pot.units["length"])
    q = q.astype(float)  # TODO: better casting
    t = u.Quantity.from_(t, pot.units["time"])
    return pot._hessian(q, t)  # noqa: SLF001


@dispatch
def hessian(pot: AbstractPotential, q: Any, /, *, t: Any) -> HessianVec:
    """Compute the hessian when `t` is keyword-only.

    Examples
    --------
    All these examples are covered by the case where `t` is positional.
    :mod:`plum` dispatches on positional arguments only, so it necessary to
    redispatch here. See the other examples in the positional-only case.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.hessian(q, t=t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    """
    return api.hessian(pot, q, t)


# =============================================================================
# Acceleration


@dispatch
def acceleration(
    pot: AbstractPotential,
    /,
    *args: Any,  # defer to `gradient`
    **kwargs: Any,  # defer to `gradient`
) -> cx.vecs.CartesianAcc3D:
    """Compute the acceleration due to the potential at the given position(s).

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the acceleration of.
    *args : Any
        The phase-space + time position to compute the acceleration. See
        `~galax.potential.gradient` for more details.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> print(pot.acceleration(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> print(pot.acceleration(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[-0.086 -0.172 -0.258]
         [-0.027 -0.033 -0.04 ]]>

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> print(pot.acceleration(w))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    We can compute the potential energy at a position (and time, which may be a
    keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> print(pot.acceleration(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> print(pot.acceleration(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[-0.086 -0.172 -0.258]
         [-0.027 -0.033 -0.04 ]]>

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> print(pot.acceleration(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> print(pot.acceleration(q, t))
    <CartesianAcc3D (d2_x[kpc / Myr2], d2_y[kpc / Myr2], d2_z[kpc / Myr2])
        [[-0.086 -0.172 -0.258]
         [-0.027 -0.033 -0.04 ]]>

    """
    return -api.gradient(pot, *args, **kwargs)


# =============================================================================
# Tidal Tensor


@dispatch
def tidal_tensor(pot: AbstractPotential, *args: Any, **kwargs: Any) -> gt.BtQuSz33:
    """Compute the tidal tensor.

    See https://en.wikipedia.org/wiki/Tidal_tensor

    .. note::

        This is in cartesian coordinates with a Euclidean metric. Also, this
        isn't correct for GR.

    Parameters
    ----------
    pot : `~galax.potential.AbstractPotential`
        The potential to compute the tidal tensor of.
    *args, **kwargs : Any
        The arguments to pass to `~galax.potential.hessian`.

    Examples
    --------
    For this example we will use a simple potential, the Kepler potential.

    First some imports:

    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the tidal tensor:

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"),
    ...                           t=u.Quantity(0, "Gyr"))

    >>> pot.tidal_tensor(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                           p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                           t=u.Quantity([0, 1], "Gyr"))
    >>> pot.tidal_tensor(w)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a `~galax.coordinates.AbstractOnePhaseSpacePosition`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.tidal_tensor(w)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can compute the tidal tensor at a position and time:

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions / times:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

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

    :mod:`plum` dispatches on positional arguments only, so it necessary to
    redispatch when `t` is a keyword argument.

    >>> pot = gp.KeplerPotential(m_tot=u.Quantity(1e12, "Msun"), units="galactic")

    >>> q = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t=t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    - - -

    `~galax.potential.tidal_tensor` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and `astropy.units.Quantity`, which
    are interpreted like their jax'ed counterparts `~coordinax.AbstractPos3D`
    and `~unxt.Quantity`.

    .. invisible-code-block: python

        from galax._interop.optional_deps import OptDeps

    .. skip: start if(not OptDeps.ASTROPY.installed, reason="requires Astropy")

    >>> import numpy as np
    >>> import astropy.coordinates as apyc
    >>> import astropy.units as apyu

    We can compute the tidal tensor at a position (and time, if any parameters
    are time-dependent):

    >>> q = apyc.CartesianRepresentation([1, 2, 3], unit="kpc")
    >>> t = 0 * apyu.Gyr
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[[ 0.00800845, -0.00152542, -0.00266948],
                          [-0.00152542,  0.00228813, -0.01067794],
                          [-0.00266948, -0.01067794, -0.01029658]],
                          [[ 0.00436863, -0.00161801, -0.00258882],
                          [-0.00161801,  0.00097081, -0.00647205],
                          [-0.00258882, -0.00647205, -0.00533944]]], dtype=float64),
                    unit='1 / Myr2')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`, we can
    instead pass a `astropy.units.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.tidal_tensor(q, t)
    Quantity[...](Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.

    .. skip: end

    """
    J = hessian(pot, *args, **kwargs)  # (*batch, 3, 3)
    batch_shape, arr_shape = batched_shape(J, expect_ndim=2)  # (*batch), (3, 3)
    traced = (
        expand_batch_dims(jnp.eye(3), ndim=len(batch_shape))
        * expand_arr_dims(jnp.trace(J, axis1=-2, axis2=-1), ndim=len(arr_shape))
        / 3
    )
    return J - traced


# =============================================================================
# Local Circular Velocity


# TODO: should this be moved to `galax.dynamics`?
@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, x: gt.LengthSz3, /, t: gt.TimeSz0
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position.

    Parameters
    ----------
    pot : AbstractPotential
        The Potential.
    x : Quantity[float, (*batch, 3), "length"]
        Position(s) to estimate the circular velocity.
    t : Quantity[float, (), "time"]
        Time at which to compute the circular velocity.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20.0, "kpc"),
    ...                       units="galactic")
    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> gp.local_circular_velocity(pot, x, t=u.Quantity(0.0, "Gyr"))
    Quantity['speed'](Array(0.16894332, dtype=float64), unit='kpc / Myr')

    """
    r = jnp.linalg.vector_norm(x, axis=-1)
    dPhi_dxyz = convert(pot.gradient(x, t=t), u.Quantity)
    dPhi_dr = jnp.sum(dPhi_dxyz * x / r[..., None], axis=-1)
    return jnp.sqrt(r * jnp.abs(dPhi_dr))


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, x: gt.LengthSz3, /, *, t: gt.TimeSz0
) -> gt.BBtRealQuSz0:
    return api.local_circular_velocity(pot, x, t)


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, q: cx.vecs.AbstractPos3D, /, t: gt.TimeSz0
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import coordinax as cx

    >>> pot = gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20.0, "kpc"),
    ...                       units="galactic")
    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "kpc")
    >>> gp.local_circular_velocity(pot, x, t=u.Quantity(0.0, "Gyr"))
    Quantity['speed'](Array(0.16894332, dtype=float64), unit='kpc / Myr')

    """
    return api.local_circular_velocity(pot, convert(q, u.Quantity), t)


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, q: cx.vecs.AbstractPos3D, /, *, t: gt.TimeSz0
) -> gt.BBtRealQuSz0:
    return api.local_circular_velocity(pot, q, t)


@dispatch
@partial(jax.jit, inline=True)
def local_circular_velocity(
    pot: AbstractPotential, w: gc.AbstractOnePhaseSpacePosition, /
) -> gt.BBtRealQuSz0:
    """Estimate the circular velocity at the given position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(20.0, "kpc"),
    ...                       units="galactic")
    >>> q = gc.PhaseSpacePosition(q=u.Quantity([8.0, 0.0, 0.0], "kpc"),
    ...                           p=u.Quantity([0.0, 0.0, 0.0], "km/s"),
    ...                           t=u.Quantity(0.0, "Gyr"))
    >>> gp.local_circular_velocity(pot, q)
    Quantity['speed'](Array(0.16894332, dtype=float64), unit='kpc / Myr')

    """
    return api.local_circular_velocity(pot, w.q, w.t)
