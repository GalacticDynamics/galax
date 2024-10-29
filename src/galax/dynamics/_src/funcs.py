"""galax: Galactic Dynamix in Jax."""

__all__ = [
    "specific_angular_momentum",
    "lagrange_points",
    "tidal_radius",
]


from functools import partial

import jax
from jaxtyping import Float, Shaped
from plum import convert, dispatch

import coordinax as cx
import quaxed.numpy as jnp
from unxt import Quantity

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from galax.potential._src.funcs import d2potential_dr2

# ===================================================================
# Specific angular momentum


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: gt.LengthBatchVec3, v: gt.SpeedBatchVec3, /
) -> Shaped[Quantity, "*batch 3"]:
    """Compute the specific angular momentum.

    Arguments:
    ---------
    x: Quantity[Any, (3,), "length"]
        3d Cartesian position (x, y, z).
    v: Quantity[Any, (3,), "speed"]
        3d Cartesian velocity (v_x, v_y, v_z).

    Returns
    -------
    Quantity[Any, (3,), "angular momentum"]
        Specific angular momentum.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.dynamics as gd

    >>> x = Quantity([8.0, 0.0, 0.0], "m")
    >>> v = Quantity([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity['diffusivity'](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    """
    return jnp.linalg.cross(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(
    x: cx.AbstractPos3D, v: cx.AbstractVel3D, /
) -> gt.BatchQVec3:
    """Compute the specific angular momentum.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import galax.dynamics as gd

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> gd.specific_angular_momentum(x, v)
    Quantity['diffusivity'](Array([ 0.,  0., 64.], dtype=float64), unit='m2 / s')

    """
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    x = convert(x.represent_as(cx.CartesianPos3D), Quantity)
    v = convert(v.represent_as(cx.CartesianVel3D, x), Quantity)
    return specific_angular_momentum(x, v)


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(w: cx.Space) -> gt.BatchQVec3:
    """Compute the specific angular momentum.

    Examples
    --------
    >>> import coordinax as cx
    >>> w = cx.Space(length=cx.CartesianPos3D.from_([[[7., 0, 0], [8, 0, 0]]], "m"),
    ...              speed=cx.CartesianVel3D.from_([[[0., 5, 0], [0, 6, 0]]], "m/s"))

    >>> specific_angular_momentum(w)
    Quantity['diffusivity'](Array([[[ 0.,  0., 35.], [ 0.,  0., 48.]]], dtype=float64), unit='m2 / s')

    """  # noqa: E501
    # TODO: keep as a vector.
    #       https://github.com/GalacticDynamics/vector/issues/27
    return specific_angular_momentum(w["length"], w["speed"])


@dispatch
@partial(jax.jit, inline=True)
def specific_angular_momentum(w: gc.AbstractBasePhaseSpacePosition) -> gt.BatchQVec3:
    r"""Compute the specific angular momentum.

    .. math::

        \boldsymbol{{L}} = \boldsymbol{{q}} \times \boldsymbol{{p}}

    Returns
    -------
    L : Quantity[float, (*batch,3)]
        Array of angular momentum vectors in Cartesian coordinates.

    Examples
    --------
    We assume the following imports

    >>> from unxt import Quantity
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd

    We can compute the angular momentum of a single object

    >>> w = gc.PhaseSpacePosition(q=Quantity([1., 0, 0], "au"),
    ...                           p=Quantity([0, 2., 0], "au/yr"),
    ...                           t=Quantity(0, "yr"))
    >>> gd.specific_angular_momentum(w)
    Quantity[...](Array([0., 0., 2.], dtype=float64), unit='AU2 / yr')
    """
    return specific_angular_momentum(w.q, w.p)


# ===================================================================
# Orbital angular frequency


@dispatch
@partial(jax.jit, inline=True)
def _orbital_angular_frequency(
    x: gt.LengthBatchVec3, v: gt.SpeedBatchVec3, /
) -> Shaped[Quantity["frequency"], "*batch"]:
    """Compute the orbital angular frequency about the origin.

    Arguments:
    ---------
    x: Quantity[Any, (3,), "length"]
        3d Cartesian position (x, y, z).
    v: Quantity[Any, (3,), "speed"]
        3d Cartesian velocity (v_x, v_y, v_z).

    Returns
    -------
    Quantity[Any, (3,), "frequency"]
        Angular velocity.

    Examples
    --------
    >>> from unxt import Quantity

    >>> x = Quantity([8.0, 0.0, 0.0], "m")
    >>> v = Quantity([0.0, 8.0, 0.0], "m/s")
    >>> _orbital_angular_frequency(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')
    """
    r = jnp.linalg.vector_norm(x, axis=-1, keepdims=True)
    omega = jnp.linalg.cross(x, v) / r**2
    return jnp.linalg.vector_norm(omega, axis=-1)


@dispatch
@partial(jax.jit, inline=True)
def _orbital_angular_frequency(
    x: cx.AbstractPos3D, v: cx.AbstractVel3D, /
) -> Shaped[Quantity["frequency"], "*batch"]:
    """Compute the orbital angular frequency about the origin.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> x = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "m")
    >>> v = cx.CartesianVel3D.from_([0.0, 8.0, 0.0], "m/s")
    >>> _orbital_angular_frequency(x, v)
    Quantity['frequency'](Array(1., dtype=float64), unit='1 / s')

    """
    # TODO: more directly using the vectors
    x = convert(x.represent_as(cx.CartesianPos3D), Quantity)
    v = convert(v.represent_as(cx.CartesianVel3D, x), Quantity)
    return _orbital_angular_frequency(x, v)


# ===================================================================


@partial(jax.jit, inline=True)
def tidal_radius(
    potential: gp.AbstractBasePotential,
    x: gt.LengthBatchVec3,
    v: gt.SpeedBatchVec3,
    /,
    prog_mass: gt.MassBatchableScalar,
    t: gt.TimeBatchableScalar,
) -> Float[Quantity["length"], "*batch"]:
    """Compute the tidal radius of a cluster in the potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractBasePotential`
        The gravitational potential of the host.
    x: Quantity[float, (3,), "length"]
        3d position (x, y, z).
    v: Quantity[float, (3,), "speed"]
        3d velocity (v_x, v_y, v_z).
    prog_mass : Quantity[float, (), "mass"]
        Cluster mass.
    t: Quantity[float, (), "time"]
        Time.

    Returns
    -------
    Quantity[float, (), "length"]
        Tidal radius of the cluster.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from galax.potential import NFWPotential

    >>> pot = NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> prog_mass = Quantity(1e4, "Msun")

    >>> tidal_radius(pot, x, v, prog_mass=prog_mass, t=Quantity(0, "Myr"))
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')
    """
    omega = _orbital_angular_frequency(x, v)
    d2phi_dr2 = d2potential_dr2(potential, x, t)
    return jnp.cbrt(potential.constants["G"] * prog_mass / (omega**2 - d2phi_dr2))


# ===================================================================


@partial(jax.jit, inline=True)
def lagrange_points(
    potential: gp.AbstractBasePotential,
    x: gt.LengthVec3,
    v: gt.SpeedVec3,
    prog_mass: gt.MassScalar,
    t: gt.TimeScalar,
) -> tuple[gt.LengthVec3, gt.LengthVec3]:
    """Compute the lagrange points of a cluster in a host potential.

    Parameters
    ----------
    potential : `galax.potential.AbstractBasePotential`
        The gravitational potential of the host.
    x: Quantity[float, (3,), "length"]
        Cartesian 3D position ($x$, $y$, $z$)
    v: Quantity[float, (3,), "speed"]
        Cartesian 3D velocity ($v_x$, $v_y$, $v_z$)
    prog_mass: Quantity[float, (), "mass"]
        Cluster mass.
    t: Quantity[float, (), "time"]
        Time.

    Returns
    -------
    L_1, L_2: Quantity[float, (3,), "length"]
        The lagrange points L_1 and L_2.

    Examples
    --------
    >>> from unxt import Quantity
    >>> import galax.potential as gp

    >>> pot = gp.MilkyWayPotential()
    >>> x = Quantity([8.0, 0.0, 0.0], "kpc")
    >>> v = Quantity([0.0, 220.0, 0.0], "km/s")
    >>> prog_mass = Quantity(1e4, "Msun")
    >>> t = Quantity(0.0, "Gyr")

    >>> L1, L2 = lagrange_points(pot, x, v, prog_mass, t)
    >>> L1
    Quantity['length'](Array([7.97070926, 0.        , 0.        ], dtype=float64), unit='kpc')
    >>> L2
    Quantity['length'](Array([8.02929074, 0.        , 0.        ], dtype=float64), unit='kpc')
    """  # noqa: E501
    r_hat = cx.normalize_vector(x)
    r_t = tidal_radius(potential, x, v, prog_mass, t)
    L_1 = x - r_hat * r_t  # close
    L_2 = x + r_hat * r_t  # far
    return L_1, L_2
