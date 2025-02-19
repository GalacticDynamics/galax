"""Functions related to computing cluster radii.

This is public API.

"""

__all__ = [
    "tidal_radius",
    "AbstractTidalRadiusMethod",
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
import unxt as u
from unxt.quantity import BareQuantity

import galax.coordinates as gc
import galax.potential as gp
import galax.typing as gt
from galax.dynamics._src.api import omega
from galax.utils._unxt import AllowValue


class AbstractTidalRadiusMethod:
    """Abstract base class for tidal radius flags.

    Examples
    --------
    >>> import galax.dynamics.cluster as gdc

    >>> try: gdc.radius.AbstractTidalRadiusMethod()
    ... except TypeError as e: print(e)
    Cannot instantiate AbstractTidalRadiusMethod

    """

    def __new__(cls) -> NoReturn:
        msg = "Cannot instantiate AbstractTidalRadiusMethod"
        raise TypeError(msg)


@dispatch
def tidal_radius(
    pot: gp.AbstractPotential, *args: Any, **kwargs: Any
) -> gt.BBtRealQuSz0:
    """Compute radius, defaulting to King (1962) tidal radius."""
    return tidal_radius_king1962(pot, *args, **kwargs)


#####################################################################
# von Hoerner (1957) tidal radius


@final
class Hoerner1957(AbstractTidalRadiusMethod):
    pass


@dispatch
def tidal_radius(
    _: type[Hoerner1957], pot: gp.AbstractPotential, *args: Any, **kw: Any
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_hoerner1957(pot, *args, **kw)


# ---------------------------


def tidal_radius_hoerner1957(
    pot: gp.AbstractPotential,
    xyz: gt.BBtRealQuSz3,
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
    r = jnp.linalg.vector_norm(xyz, axis=-1)
    return jnp.cbrt(mass / (2 * gp.spherical_mass_enclosed(pot, xyz, t))) * r


#####################################################################


@final
class King1962PointMass(AbstractTidalRadiusMethod):
    r"""Tidal radius from King (1962) with a point mass.

    $$ r_t = R \left(\frac{M_c}{(3+e)M_g}\right)^{1/3} $$

    """


@dispatch
def tidal_radius(
    _: type[King1962PointMass], pot: gp.AbstractPotential, **kw: Any
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962_pointmass(pot, **kw)


# ---------------------------


def tidal_radius_king1962_pointmass(
    pot: gp.AbstractPotential,
    /,
    *,
    rperi: gt.BBtRealQuSz0 | cx.vecs.AbstractPos3D,
    mass: gt.MassBBtSz0,
    t: gt.BBtRealQuSz0 | float,
    e: float = 0.0,
) -> gt.BBtRealQuSz0:
    r"""Calculate the tidal radius of a star cluster based on King (1962).

    $$ r_t = R_{peri} \left(\frac{M_c}{(3+e)M_g}\right)^{1/3} $$

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

    r = jnp.linalg.vector_norm(x, axis=-1)
    m_encl = gp.spherical_mass_enclosed(pot, x, t)  # TODO: flag to sel mass calculator
    return jnp.cbrt(mass / ((3 + e) * m_encl)) * r


#####################################################################


@final
class King1962(AbstractTidalRadiusMethod):
    r"""Calculate the tidal radius of a star cluster based on King (1962).

    $$ r_t^3 = \frac{G M_c}{\Omega^2 - \frac{d^2\Phi}{dr^2}} $$

    where $G$ is the gravitational constant, $M_c$ is the cluster mass, $\Omega$
    is the cluster's angular velocity, and $\frac{d^2\Phi}{dr^2}$ is the second
    derivative of the potential at the cluster's position.

    """


@dispatch
def tidal_radius(
    _: type[King1962], pot: gp.AbstractPotential, *args: Any, **kw: Any
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962(pot, *args, **kw)


# ---------------------------


@dispatch
@partial(jax.jit)
def tidal_radius_king1962(
    pot: gp.AbstractPotential,
    xyz: gt.BBtRealSz3,
    v: gt.BBtRealSz3,
    /,
    *,
    mass: gt.BBtSz0 | gt.BBtQuSz0,  # cluster mass
    t: gt.BBtSz0 | gt.BBtQuSz0,
) -> gt.BBtRealSz0:
    """Compute from `jax.Array`."""
    mass = u.ustrip(AllowValue, pot.units["mass"], mass)
    t = u.ustrip(AllowValue, pot.units["time"], t)

    d2phi_dr2 = pot.d2potential_dr2(xyz, t)
    GM = pot.constants["G"].value * mass  # TODO: G unit handling
    return jnp.cbrt(GM / (omega(xyz, v) ** 2 - d2phi_dr2))


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
    pot: gp.AbstractPotential,
    w: gc.AbstractPhaseSpaceCoordinate,
    /,
    *,
    mass: gt.MassBBtSz0,
) -> gt.BBtRealQuSz0:
    """Compute the tidal radius of a cluster in the potential."""
    return tidal_radius_king1962(pot, w.q, w.p, mass=mass, t=w.t.squeeze())
