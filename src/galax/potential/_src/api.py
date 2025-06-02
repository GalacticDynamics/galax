"""Functional API for galax potentials."""

__all__ = [
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
    "local_circular_velocity",
    "spherical_mass_enclosed",
    "dpotential_dr",
    "d2potential_dr2",
]

from typing import Any

from jaxtyping import Array
from plum import dispatch

import unxt as u

import galax._custom_types as gt


@dispatch.abstract
def potential(*args: Any, **kwargs: Any) -> Any:
    """Compute the potential energy at the given position(s).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the potential energy at a given
    coordinate:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.potential(w)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> pot.potential(w)
    Quantity(Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    This function is very flexible and can accept a broad variety of inputs.
    Let's work up the type ladder:

    - `jax.Array`s: which is interpreted as a `coordinax.vecs.CartesianPos3D`
      position in the same unit system as the potential. For performance reasons
      the output is a `jax.Array`. Be careful!

    >>> xyz = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.potential(xyz, t)
    Array([-1.20227527, -0.5126519 ], dtype=float64)

    >>> pot.potential(xyz, t=t)
    Array([-1.20227527, -0.5126519 ], dtype=float64)

    - A `unxt.Quantity`, which is interpreted as a
      `coordinax.vecs.CartesianPos3D` position:

    >>> xyz = u.Quantity([1., 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.potential(xyz, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    >>> pot.potential(xyz, t=t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - `coordinax.vecs.AbstractPos3D`:

    >>> q = cx.CartesianPos3D.from_(xyz)
    >>> pot.potential(q, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    >>> pot.potential(q, t=t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    >>> qs = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.potential(qs, t)
    Quantity(Array([-1.20227527, -0.5126519 ], dtype=float64), unit='kpc2 / Myr2')

    - `coordinax.vecs.FourVector`:

    >>> tq = cx.FourVector(q=q, t=t)
    >>> pot.potential(tq)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - `coordinax.vecs.Space`:

    >>> space = cx.vecs.Space(length=q)
    >>> pot.potential(space, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    >>> space = cx.vecs.Space(length=tq)
    >>> pot.potential(space)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - `coordinax.frames.Coordinate`:

    >>> coord = cx.frames.Coordinate({"length": q}, frame=gc.frames.simulation_frame)
    >>> pot.potential(coord, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - `galax.coordinates.PhaseSpacePosition`:

    >>> p = u.Quantity([4, 5, 6], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> pot.potential(w, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> pot.potential(w)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')


    ## Astropy Support

    `galax.potential.potential` also supports Astropy objects, like
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
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.potential(q, t)
    Quantity(Array([-0.55372734, -0.46647294], dtype=float64), unit='kpc2 / Myr2')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`, we can
    instead pass a `astropy.units.Quantity`, which is interpreted as a Cartesian
    position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.potential(q, t)
    Quantity(Array(-1.20227527, dtype=float64), unit='kpc2 / Myr2')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.potential(q, t)
    Array([-1.20227527, -0.5126519 ], dtype=float64)

    .. skip: end

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def gradient(*args: Any, **kwargs: Any) -> Any:
    """Compute the gradient of the potential at the given position(s).

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> print(pot.gradient(w))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions and times:

    >>> wt = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                              p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                              t=u.Quantity([0, 1], "Gyr"))
    >>> print(pot.gradient(wt))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    The function is very flexible and can accept a broad variety of inputs.
    Let's work up the type ladder:

    - `jax.Array`s: which is interpreted as a `coordinax.vecs.CartesianPos3D`
      position in the same unit system as the potential. For performance reasons
      the output is a `jax.Array`. Be careful!

    >>> xyz = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.gradient(q, t)
    Array([[0.08587681, 0.17175361, 0.25763042],
           [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64)

    >>> pot.gradient(q, t=t)
    Array([[0.08587681, 0.17175361, 0.25763042],
           [0.02663127, 0.03328908, 0.0399469 ]], dtype=float64)

    - A `unxt.Quantity`, which is interpreted as a
      `coordinax.vecs.CartesianPos3D` position:

    >>> xyz = u.Quantity([1., 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.gradient(xyz, t)
    Quantity(Array([0.08587681, 0.17175361, 0.25763042], dtype=float64), unit='kpc / Myr2')

    >>> pot.gradient(xyz, t=t)
    Quantity(Array([0.08587681, 0.17175361, 0.25763042], dtype=float64), unit='kpc / Myr2')

    - `coordinax.vecs.AbstractPos3D`:

    >>> q = cx.CartesianPos3D.from_(xyz)
    >>> print(pot.gradient(q, t=t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions:

    >>> qs = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> print(pot.gradient(qs, t=t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    - `coordinax.vecs.FourVector`:

    >>> w = cx.FourVector(q=q, t=t)
    >>> print(pot.gradient(w))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    - `coordinax.vecs.Space`:

    >>> w = cx.vecs.Space(length=q)
    >>> print(pot.gradient(w, t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    - `coordinax.frames.Coordinate`:

    >>> w = cx.frames.Coordinate({"length": q},
    ...                          frame=gc.frames.simulation_frame)
    >>> print(pot.gradient(w, t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    - `galax.coordinates.PhaseSpacePosition`:

    >>> p = u.Quantity([4, 5, 6], "km/s")
    >>> w = gc.PhaseSpacePosition(q=q, p=p)
    >>> print(pot.gradient(w, t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> print(pot.gradient(wt))  # re-using the previous example
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>


    ## Astropy Support

    `galax.potential.gradient` also supports Astropy objects, like
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
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [0.086 0.172 0.258]>

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation([[1, 4], [2, 5], [3, 6]], unit="kpc")
    >>> print(pot.gradient(q, t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[0.086 0.172 0.258]
         [0.027 0.033 0.04 ]]>

    Instead of passing a `~astropy.coordinates.Representation` (in this case a
    `~astropy.coordinates.CartesianRepresentation`), we can instead pass a
    `astropy.units.Quantity`, which is interpreted as a Cartesian position:

    >>> q = [1., 2, 3] * apyu.kpc
    >>> print(pot.gradient(q, t))
    Quantity(Array([0.08587681, 0.17175361, 0.25763042], dtype=float64), unit='kpc / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~numpy.ndarray`), it is assumed to be in the same unit system as the
    potential.

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> print(pot.gradient(q, t))
    [[0.08587681 0.17175361 0.25763042]
     [0.02663127 0.03328908 0.0399469 ]]

    .. skip: end

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def laplacian(*args: Any, **kwargs: Any) -> u.Quantity["1/s^2"] | Array:
    """Compute the laplacian of the potential at the given position(s).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.laplacian(w)
    Quantity(Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> pot.laplacian(w)
    Quantity(Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    This function is very flexible and can accept a broad variety of inputs. For
    example, instead of passing a
    `galax.coordinates.PhaseSpaceCoordinate`, we can instead pass a
    `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.laplacian(w)
    Quantity(Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    Or using a `~coordinax.AbstractPos3D` and time `unxt.Quantity` (which can be
    positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.laplacian(q, t=t)
    Quantity(Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.laplacian(q, t=t)
    Quantity(Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a
    `unxt.Quantity`, which is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.laplacian(q, t)
    Quantity(Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.laplacian(q, t)
    Array([2.77555756e-17, 0.00000000e+00], dtype=float64)

    - - -

    :func:`galax.potential.laplacian` also supports Astropy objects, like
    `astropy.coordinates.BaseRepresentation` and
    `astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts `~coordinax.AbstractPos3D` and
    `~unxt.Quantity`.

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
    Quantity(Array(2.77555756e-17, dtype=float64), unit='1 / Myr2')

    We can also compute the potential energy at multiple positions:

    >>> q = apyc.CartesianRepresentation(apyu.Quantity([[1, 4], [2, 5], [3, 6]], "kpc"))
    >>> pot.laplacian(q, t)
    Quantity(Array([2.77555756e-17, 0.00000000e+00], dtype=float64), unit='1 / Myr2')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a
    `unxt.Quantity`, which is interpreted as a Cartesian position:

    >>> q = [1., 2, 3] * apyu.kpc
    >>> pot.laplacian(q, t)
    Array(2.77555756e-17, dtype=float64)

    Again, this can be batched.  If the input position object has no units (i.e.
    is a `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> import numpy as np
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> pot.laplacian(q, t)
    Array([2.77555756e-17, 0.00000000e+00], dtype=float64)

    .. skip: end

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def density(*args: Any, **kwargs: Any) -> u.Quantity["mass density"] | Array:
    """Compute the density at the given position(s).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the density:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.density(w)
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> pot.density(w)
    Quantity(Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    This function is very flexible and can accept a broad variety of inputs. For
    example, instead of passing a
    `galax.coordinates.PhaseSpaceCoordinate`, we can instead pass a
    `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.density(w)
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    Or using a `~coordinax.AbstractPos3D` and time `unxt.Quantity` (which can be
    positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.density(q, t=t)
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.density(q, t=t)
    Quantity(Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a
    `unxt.Quantity`, which is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.density(q, t=t)
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.density(q, t)
    Array([0., 0.], dtype=float64)

    - - -

    meth:`galax.potential.AbstractPotential.density` also supports Astropy
    objects, like `astropy.coordinates.BaseRepresentation` and
    `astropy.units.Quantity`, which are interpreted like their jax'ed
    counterparts `~coordinax.AbstractPos3D` and
    `~unxt.Quantity`.

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
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    We can also compute the density at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.density(q, t)
    Quantity(Array([0., 0.], dtype=float64), unit='solMass / kpc3')

    Instead of passing a `astropy.coordinates.CartesianRepresentation`,
    we can instead pass a `astropy.units.Quantity`, which is interpreted
    as a Cartesian position:

    >>> q = [1, 2, 3] * apyu.kpc
    >>> pot.density(q, t)
    Quantity(Array(0., dtype=float64), unit='solMass / kpc3')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.density(q, t)
    Array([0., 0.], dtype=float64)

    .. skip: end

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def hessian(*args: Any, **kwargs: Any) -> Any:
    """Compute the hessian of the potential at the given position(s).

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the hessian:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.hessian(w)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> pot.hessian(w)
    Quantity(Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                  unit='1 / Myr2')

    This function is very flexible and can accept a broad variety of inputs:

    - `coordinax.vecs.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.hessian(w)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')

    - `coordinax.vecs.AbstractPos3D` and time `unxt.Quantity` (which can be
      positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.hessian(q, t=t)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')

    We can also compute the hessian at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.hessian(q, t=t)
    Quantity(Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    - A `unxt.Quantity`, which is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.hessian(q, t=t)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                  unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> import jax.numpy as jnp
    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.hessian(q, t=t)
    Array([[[ 0.06747463, -0.03680435, -0.05520652],
            [-0.03680435,  0.01226812, -0.11041304],
            [-0.05520652, -0.11041304, -0.07974275]],
           [[ 0.00250749, -0.00518791, -0.00622549],
            [-0.00518791,  0.00017293, -0.00778186],
            [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64)

    - `jax.Array`, which are Cartesian positions, assumed to be in the same
      units as the potential:

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = jnp.asarray(0)
    >>> pot.hessian(q, t=t)
    Array([[[ 0.06747463, -0.03680435, -0.05520652],
            [-0.03680435,  0.01226812, -0.11041304],
            [-0.05520652, -0.11041304, -0.07974275]],
           [[ 0.00250749, -0.00518791, -0.00622549],
            [-0.00518791,  0.00017293, -0.00778186],
            [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64)

    - - -

    `galax.potential.hessian` also supports Astropy objects, like
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
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the hessian at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.hessian(q, t)
    Quantity(Array([[[ 0.00800845, -0.00152542, -0.00266948],
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
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  Also, If the input position object has no units
    (i.e. is an `~numpy.ndarray`), it is assumed to be in the same unit system
    as the potential.

    >>> q = np.array([[1, 2, 3], [4, 5, 6]])
    >>> pot.hessian(q, t)
    Array([[[ 0.06747463, -0.03680435, -0.05520652],
            [-0.03680435,  0.01226812, -0.11041304],
            [-0.05520652, -0.11041304, -0.07974275]],
           [[ 0.00250749, -0.00518791, -0.00622549],
            [-0.00518791,  0.00017293, -0.00778186],
            [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64)

    .. skip: end

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def acceleration(*args: Any, **kwargs: Any) -> Any:
    """Compute the acceleration due to the potential at the given position(s).

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    We can construct a potential and compute the potential energy:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> print(pot.acceleration(w))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> print(pot.acceleration(w))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[-0.086 -0.172 -0.258]
         [-0.027 -0.033 -0.04 ]]>

    This function is very flexible and can accept a broad variety of inputs. For
    example, instead of passing a `galax.coordinates.PhaseSpaceCoordinate`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> print(pot.acceleration(w))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    Or using a `~coordinax.AbstractPos3D` and time `unxt.Quantity` (which can be
    positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> print(pot.acceleration(q, t=t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [-0.086 -0.172 -0.258]>

    We can also compute the potential energy at multiple positions:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> print(pot.acceleration(q, t=t))
    <CartesianAcc3D (x[kpc / Myr2], y[kpc / Myr2], z[kpc / Myr2])
        [[-0.086 -0.172 -0.258]
         [-0.027 -0.033 -0.04 ]]>

    Instead of passing a `~coordinax.AbstractPos3D` (in this case a
    `~coordinax.CartesianPos3D`), we can instead pass a `unxt.Quantity`, which
    is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.acceleration(q, t=t)
    Quantity(Array([-0.08587681, -0.17175361, -0.25763042], dtype=float64), unit='kpc / Myr2')

    """  # noqa: E501
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def tidal_tensor(*args: Any, **kwargs: Any) -> gt.BBtQuSz33 | gt.BBtSz33:
    """Compute the tidal tensor.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import galax.potential as gp
    >>> import galax.coordinates as gc

    Then we can construct a potential and compute the tidal tensor:

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.tidal_tensor(w)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the potential energy at multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))
    >>> pot.tidal_tensor(w)
    Quantity(Array([[[ 0.06747463, -0.03680435, -0.05520652],
                          [-0.03680435,  0.01226812, -0.11041304],
                          [-0.05520652, -0.11041304, -0.07974275]],
                          [[ 0.00250749, -0.00518791, -0.00622549],
                          [-0.00518791,  0.00017293, -0.00778186],
                          [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64),
                    unit='1 / Myr2')

    This function is very flexible and can accept a broad variety of inputs. For
    example, instead of passing a
    `galax.coordinates.PhaseSpaceCoordinate`, we can instead pass a
    `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.tidal_tensor(w)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Or using a `~coordinax.AbstractPos3D` and time `unxt.Quantity` (which can be
    positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t=t)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions / times:

    >>> q = cx.CartesianPos3D.from_([[1, 2, 3], [4, 5, 6]], "kpc")
    >>> pot.tidal_tensor(q, t=t)
    Quantity(Array([[[ 0.06747463, -0.03680435, -0.05520652],
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
    >>> pot.tidal_tensor(q, t=t)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.  If the input position object has no units (i.e.
    is an `~jax.Array`), it is assumed to be in the same unit system as the
    potential.

    >>> q = jnp.asarray([[1, 2, 3], [4, 5, 6]])
    >>> t = 0
    >>> pot.tidal_tensor(q, t=t)
    Array([[[ 0.06747463, -0.03680435, -0.05520652],
            [-0.03680435,  0.01226812, -0.11041304],
            [-0.05520652, -0.11041304, -0.07974275]],
           [[ 0.00250749, -0.00518791, -0.00622549],
            [-0.00518791,  0.00017293, -0.00778186],
            [-0.00622549, -0.00778186, -0.00268042]]], dtype=float64)

    :mod:`plum` dispatches on positional arguments only, so it necessary to
    redispatch when `t` is a keyword argument.

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> q = cx.CartesianPos3D.from_(u.Quantity([1, 2, 3], "kpc"))
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.tidal_tensor(q, t=t)
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    - - -

    `galax.potential.tidal_tensor` also supports Astropy objects, like
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
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    We can also compute the tidal tensor at multiple positions:

    >>> q = apyc.CartesianRepresentation(x=[1, 2], y=[4, 5], z=[7, 8], unit="kpc")
    >>> pot.tidal_tensor(q, t)
    Quantity(Array([[[ 0.00800845, -0.00152542, -0.00266948],
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
    Quantity(Array([[ 0.06747463, -0.03680435, -0.05520652],
                         [-0.03680435,  0.01226812, -0.11041304],
                         [-0.05520652, -0.11041304, -0.07974275]], dtype=float64),
                    unit='1 / Myr2')

    Again, this can be batched.

    .. skip: end

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def local_circular_velocity(*args: Any, **kwargs: Any) -> gt.BBtQuSz0:
    """Estimate the circular velocity at the given position.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([8.0, 0.0, 0.0], "kpc"),
    ...                             p=u.Quantity([0.0, 0.0, 0.0], "km/s"),
    ...                             t=u.Quantity(0.0, "Gyr"))
    >>> gp.local_circular_velocity(pot, w)
    Quantity(Array(0.16894332, dtype=float64), unit='kpc / Myr')

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "kpc")
    >>> gp.local_circular_velocity(pot, x, t=u.Quantity(0.0, "Gyr"))
    Quantity(Array(0.16894332, dtype=float64), unit='kpc / Myr')

    >>> x = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> gp.local_circular_velocity(pot, x, t=u.Quantity(0.0, "Gyr"))
    Quantity(Array(0.16894332, dtype=float64), unit='kpc / Myr')

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def dpotential_dr(*args: Any, **kwargs: Any) -> gt.BtQuSz0:
    """Compute the radial derivative of the potential at the given position(s).

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.dpotential_dr(w)
    Quantity(Array(0.32132158, dtype=float64), unit='kpc / Myr2')

    We can also compute the radial derivative of the potential at multiple
    positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))

    >>> pot.dpotential_dr(w)
    Quantity(Array([0.32132158, 0.05842211], dtype=float64), unit='kpc / Myr2')

    This function is very flexible and can accept a broad variety of inputs. For
    example, instead of passing a `galax.coordinates.PhaseSpaceCoordinate`, we
    can instead pass a `~vector.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=u.Quantity(0, "Gyr"))
    >>> pot.dpotential_dr(w)
    Quantity(Array(0.32132158, dtype=float64), unit='kpc / Myr2')

    Or using a `~coordinax.AbstractPos3D` and time `unxt.Quantity` (which can be
    positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.dpotential_dr(q, t=t)
    Quantity(Array(0.32132158, dtype=float64), unit='kpc / Myr2')

    """
    raise NotImplementedError  # pragma: no cover


@dispatch.abstract
def d2potential_dr2(*args: Any, **kwargs: Any) -> gt.BBtQorVSz0:
    """Compute the second radial derivative of the potential.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.KeplerPotential(m_tot=1e12, units="galactic")

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([1, 2, 3], "kpc"),
    ...                             p=u.Quantity([4, 5, 6], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))

    >>> pot.d2potential_dr2(w)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    We can also compute the second radial derivative of the potential at
    multiple positions and times:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([[1, 2, 3], [4, 5, 6]], "kpc"),
    ...                             p=u.Quantity([[4, 5, 6], [7, 8, 9]], "km/s"),
    ...                             t=u.Quantity([0, 1], "Gyr"))

    >>> pot.d2potential_dr2(w)
    Quantity(Array([-0.17175361, -0.01331563], dtype=float64), unit='1 / Myr2')

    This function is very flexible and can accept a broad variety of inputs.
    Let's work down the type ladder:

    - `galax.coordinates.PhaseSpacePosition`:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([1, 2, 3], "kpc"),
    ...                           p=u.Quantity([4, 5, 6], "km/s"))
    >>> t = u.Quantity(0, "Gyr")
    >>> pot.d2potential_dr2(w, t)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `coordinax.Coordinate`:

    >>> coord = cx.Coordinate({"length": cx.vecs.FourVector.from_([0, 1, 2, 3], "kpc")},
    ...                       frame=gc.frames.simulation_frame)
    >>> pot.d2potential_dr2(coord, t)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `coordinax.Space`:

    >>> space = cx.Space({"length": cx.vecs.FourVector.from_([0, 1, 2, 3], "kpc")})
    >>> pot.d2potential_dr2(space, t)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `coordinax.vecs.FourVector`:

    >>> w = cx.FourVector(q=u.Quantity([1, 2, 3], "kpc"), t=t)
    >>> pot.d2potential_dr2(w)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `coordinax.vecs.AbstractPos3D` and time `unxt.Quantity` (which can be
      positional or a keyword argument):

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "kpc")
    >>> pot.d2potential_dr2(q, t)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `unxt.Quantity` which is interpreted as a Cartesian position:

    >>> q = u.Quantity([1., 2, 3], "kpc")
    >>> pot.d2potential_dr2(q, t)
    Quantity(Array(-0.17175361, dtype=float64), unit='1 / Myr2')

    - `jax.Array` which is a Cartesian position assumed to be in the same unit
      system as the potential:

    >>> x = jnp.asarray([1, 2, 3])
    >>> t = jnp.array(0)
    >>> pot.d2potential_dr2(x, t)
    Array(-0.17175361, dtype=float64)

    """
    raise NotImplementedError  # pragma: no cover


# TODO: change the name
@dispatch.abstract
def spherical_mass_enclosed(*args: Any, **kwargs: Any) -> gt.BtQuSz0:
    r"""Compute the mass enclosed within a spherical shell, assuming spherical symmetry.

    This assumes the potential is spherical, which is often NOT correct.

    $$ M(r) = \frac{r^2}{G} \left| \frac{d\Phi}{dr} \right| $$

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential()

    Let's work down the type ladder:

    - `galax.coordinates.PhaseSpaceCoordinate`:

    >>> w = gc.PhaseSpaceCoordinate(q=u.Quantity([8, 0, 0], "kpc"),
    ...                             p=u.Quantity([0, 0, 0], "km/s"),
    ...                             t=u.Quantity(0, "Gyr"))
    >>> gp.spherical_mass_enclosed(pot, w).uconvert("Msun")
    Quantity(Array(9.99105233e+10, dtype=float64), unit='solMass')

    - `galax.coordinates.PhaseSpacePosition`:

    >>> w = gc.PhaseSpacePosition(q=u.Quantity([8, 0, 0], "kpc"),
    ...                           p=u.Quantity([0, 0, 0], "km/s"))
    >>> t = u.Quantity(0, "Gyr")
    >>> gp.spherical_mass_enclosed(pot, w, t).uconvert("Msun")
    Quantity(Array(9.99105233e+10, dtype=float64), unit='solMass')

    - `coordinax.Coordinate`:

    >>> coord = cx.Coordinate({"length": cx.vecs.FourVector.from_([0, 8, 0, 0], "kpc")},
    ...                       frame=gc.frames.simulation_frame)
    >>> gp.spherical_mass_enclosed(pot, coord).uconvert("Msun")
    Quantity(Array(9.99105233e+10, dtype=float64), unit='solMass')

    - `coordinax.vecs.FourVector`:

    >>> vec4 = cx.vecs.FourVector(q=u.Quantity([8, 0, 0], "kpc"), t=t)
    >>> gp.spherical_mass_enclosed(pot, vec4).uconvert("Msun")
    Quantity(Array(9.99105233e+10, dtype=float64), unit='solMass')

    - `coordinax.AbstractPos3D`:

    >>> q = cx.CartesianPos3D.from_([[8, 0, 0], [9, 0, 0]], "kpc")
    >>> gp.spherical_mass_enclosed(pot, q, t).uconvert("Msun")
    Quantity(Array([9.99105233e+10, 1.10435505e+11], dtype=float64), unit='solMass')

    - `unxt.AbstractQuantity`:

    >>> x = u.Quantity([8, 0, 0], "kpc")
    >>> gp.spherical_mass_enclosed(pot, x, t).uconvert("Msun")
    Quantity(Array(9.99105233e+10, dtype=float64), unit='solMass')

    >>> xs = u.Quantity([[8, 0, 0], [10, 0, 0]], "kpc")
    >>> ts = u.Quantity([0, 1], "Gyr")
    >>> gp.spherical_mass_enclosed(pot, xs, ts).uconvert("Msun")
    Quantity(Array([9.99105233e+10, 1.20586103e+11], dtype=float64), unit='solMass')

    - `jax.Array`:

    >>> x = jnp.asarray([8, 0, 0])
    >>> t = jnp.asarray(0)
    >>> gp.spherical_mass_enclosed(pot, x, t)
    Array(9.99105233e+10, dtype=float64)

    """
    raise NotImplementedError  # pragma: no cover
