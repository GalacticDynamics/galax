"""Cluster functions."""

__all__ = [
    "tidal_radius",
    "AbstractRadiusMethod",
    # specific methods
    "Hoerner1957",
    "tidal_radius_hoerner1957",
    "King1962PointMass",
    "tidal_radius_king1962_pointmass",
    "King1962",
    "tidal_radius_king1962",
]

from functools import partial
from typing import Any, NoReturn, final

import jax
from plum import dispatch

import coordinax as cx
import quaxed.numpy as jnp
from unxt.quantity import BareQuantity

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from galax.dynamics._src.api import omega

#####################################################################
# tidal radius


@dispatch.abstract
def tidal_radius(*args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.NFWPotential(m=1e12, r_s=20.0, units="galactic")

    >>> x = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc")
    >>> v = u.Quantity(jnp.asarray([8.0, 0.0, 0.0]), "kpc/Myr")
    >>> t = u.Quantity(0, "Myr")
    >>> mass = u.Quantity(1e4, "Msun")

    >>> gd.cluster.tidal_radius(pot, x, v, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> q = cx.CartesianPos3D.from_(x)
    >>> p = cx.CartesianVel3D.from_(v)
    >>> gd.cluster.tidal_radius(pot, q, p, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> space = cx.Space(length=q, speed=p)
    >>> gd.cluster.tidal_radius(pot, space, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> coord = cx.Coordinate(space, frame=gc.frames.SimulationFrame())
    >>> gd.cluster.tidal_radius(pot, coord, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    >>> w = gc.PhaseSpaceCoordinate(q=q, p=p, t=t)
    >>> gd.cluster.tidal_radius(pot, w, mass=mass)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    Now with different methods:

    >>> import galax.dynamics.cluster as gdc

    The default is King (1962):

    >>> gd.cluster.tidal_radius(gdc.radius.King1962, pot, x, v, mass=mass, t=t)
    Quantity['length'](Array(0.06362008, dtype=float64), unit='kpc')

    Also available is von Hoerner (1957):

    >>> gd.cluster.tidal_radius(gdc.radius.Hoerner1957, pot, x, mass=mass, t=t)
    Quantity[...](Array([136.40324281], dtype=float64), unit='')

    And King (1962) with a point mass:

    >>> rperi = jnp.linalg.vector_norm(x, axis=-1)
    >>> gd.cluster.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                         rperi=rperi, mass=mass, t=t, e=0.5)
    Quantity[...](Array([113.19103012], dtype=float64), unit='')

    >>> gd.cluster.tidal_radius(gdc.radius.King1962PointMass, pot,
    ...                         rperi=q, mass=mass, t=t, e=0.5)
    Quantity[...](Array([113.19103012], dtype=float64), unit='')

    """
    raise NotImplementedError  # pragma: no cover


class AbstractRadiusMethod:
    """Abstract base class for tidal radius flags.

    Examples
    --------
    >>> import galax.dynamics.cluster as gdc

    >>> try: gdc.radius.AbstractRadiusMethod()
    ... except TypeError as e: print(e)
    Cannot instantiate AbstractRadiusMethod

    """

    def __new__(cls) -> NoReturn:
        msg = "Cannot instantiate AbstractRadiusMethod"
        raise TypeError(msg)


@dispatch
def tidal_radius(*args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Compute radius, defaulting to King (1962) tidal radius."""
    return tidal_radius(King1962, *args, **kwargs)


#####################################################################
# von Hoerner (1957) tidal radius


@final
class Hoerner1957(AbstractRadiusMethod):
    pass


@dispatch
def tidal_radius(_: type[Hoerner1957], /, *args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_hoerner1957(*args, **kwargs)


# ---------------------------


def tidal_radius_hoerner1957(
    pot: gp.AbstractPotential,
    x: gt.BBtRealQuSz3,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    r"""Calculate the tidal radius of a star cluster based on von Hoerner (1957).

    Von Hoerner (1957) derived a theoretical tidal radius for a star cluster in
    a galaxy, accounting for the balance between the cluster's self-gravity and
    the galactic tidal field. The formula given is:

    $$ r_t = R \left(\frac{M_c}{2M_g}\right)^{1/3} $$

    where $R$ is the cluster's galactocentric distance, $M_c$ is the cluster's
    mass, and $M_g$ is the mass of the host galaxy enclosed at the distance $R$.

    This formula gives the instantaneous limiting radius for a cluster moving
    radially in a galactic potential. However, even von Hoerner recognized that
    this is an oversimplification and suggests using the limiting radius at the
    cluster's perigalactic point, as tidal stripping is strongest there and
    internal relaxation is too slow to regenerate lost stars between successive
    passages.

    ## Reference:

    von Hoerner, S. 1957, ApJ, 125, 451 ADS:
    https://ui.adsabs.harvard.edu/abs/1957ApJ...125..451V

    """
    # TODO: a way to select different mass calculator
    return jnp.cbrt(gp.spherical_mass_enclosed(pot, x, t) / (2 * mass))


#####################################################################


@final
class King1962PointMass(AbstractRadiusMethod):
    pass


@dispatch
def tidal_radius(
    _: type[King1962PointMass], /, *args: Any, **kwargs: Any
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962_pointmass(*args, **kwargs)


# ---------------------------


def tidal_radius_king1962_pointmass(
    pot: gp.AbstractPotential,
    /,
    *,
    rperi: gt.BBtRealQuSz0 | cx.vecs.AbstractPos3D,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
    e: float = 0.0,
) -> gt.BBtRealQuSz0:
    r"""Calculate the tidal radius of a star cluster based on King (1962).

    $$ r_t = R \left(\frac{M_c}{(3+e)M_g}\right)^{1/3} $$

    where $R$ is the cluster's galactocentric distance, $M_c$ is the cluster's
    mass, $M_g$ is the mass of the host galaxy enclosed at the distance $R$,
    and $e$ is the orbital eccentricity of the cluster.

    ## Reference:

    King, Ivan. 1957, Astronomical Journal:
    https://ui.adsabs.harvard.edu/abs/1962AJ.....67..471K/

    """
    # Parse to the quantity
    if isinstance(rperi, cx.vecs.AbstractPos3D):
        x = rperi
    else:
        x = BareQuantity(jnp.zeros((*rperi.shape, 3)), rperi.unit)
        x = x.at[..., 0].set(rperi)

    # TODO: a way to select different mass calculator
    return jnp.cbrt(gp.spherical_mass_enclosed(pot, x, t) / ((3 + e) * mass))


#####################################################################


@final
class King1962(AbstractRadiusMethod):
    r"""Calculate the tidal radius of a star cluster based on King (1962).

    $$ r_t^3 = \frac{G M_c}{\Omega^2 - \frac{d^2\Phi}{dr^2}}

    """


@dispatch
def tidal_radius(_: type[King1962], /, *args: Any, **kwargs: Any) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962(*args, **kwargs)


# ---------------------------


@dispatch
@partial(jax.jit)
def tidal_radius_king1962(
    pot: gp.AbstractPotential,
    x: gt.BBtRealQuSz3 | cx.vecs.AbstractPos3D,
    v: gt.BBtRealQuSz3 | cx.vecs.AbstractVel3D,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute from `unxt.Quantity` or `coordinax.vecs.AbstractVector`s."""
    d2phi_dr2 = pot.d2potential_dr2(x, t)
    return jnp.cbrt(pot.constants["G"] * mass / (omega(x, v) ** 2 - d2phi_dr2))


@dispatch
def tidal_radius_king1962(
    pot: gp.AbstractPotential,
    space: cx.Space,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    q, p = space["length"], space["speed"]
    return tidal_radius_king1962(pot, q, p, mass=mass, t=t)


@dispatch
def tidal_radius_king1962(
    pot: gp.AbstractPotential,
    coord: cx.frames.AbstractCoordinate,
    /,
    *,
    mass: gt.MassBBtSz0,
    t: gt.TimeBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962(pot, coord.data, mass=mass, t=t)


@dispatch
def tidal_radius_king1962(
    pot: gp.AbstractPotential, w: gc.PhaseSpaceCoordinate, /, *, mass: gt.MassBBtSz0
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962(pot, w.q, w.p, mass=mass, t=w.t)
