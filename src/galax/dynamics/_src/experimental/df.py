"""Experimental dynamics."""

__all__: list[str] = []

import abc
import functools as ft
from typing import Any, Final, Self, final

import jax
import jax.random as jr
from jax.stages import ArgInfo
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray

import coordinax as cx
import quaxed.numpy as jnp
from dataclassish import field_items
from dataclassish.converters import Unless, dataclass, field

import galax._custom_types as gt
import galax.potential as gp
from galax.dynamics._src.api import omega
from galax.dynamics._src.cluster.api import tidal_radius

##############################################################################


class AbstractKinematicDF(metaclass=abc.ABCMeta):
    """Abstract class for kinematic distribution functions."""

    @abc.abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        pot: gp.AbstractPotential,
        /,
        t: gt.LikeSz0,
        x: gt.Sz3,
        v: gt.Sz3,
        Msat: gt.LikeSz0,
    ) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
        raise NotImplementedError

    # ================================================
    # JAX stuff

    def tree_flatten(
        self,
    ) -> tuple[dict[str, gt.Sz0], None]:
        return (dict(field_items(self)), None)

    @classmethod
    def tree_unflatten(cls, _: Any, children: dict[str, gt.Sz0]) -> Self:
        return cls(**children, _skip_convert=True)  # type: ignore[call-arg]


##############################################################################

default_kr_bar: Final = 2.0
default_kvphi_bar: Final = 0.3
default_kz_bar: Final = 0.0
default_kvz_bar: Final = 0.0
default_sigma_kr: Final = 0.5
default_sigma_kvphi: Final = 0.5
default_sigma_kz: Final = 0.5
default_sigma_kvz: Final = 0.5

array_unless_jitting = Unless(ArgInfo, jnp.asarray)


# TODO: rename to Fardal-like, since it can have different parameters?
@final
@ft.partial(
    register_dataclass,
    data_fields=[
        "kr_bar",
        "kvphi_bar",
        "kz_bar",
        "kvz_bar",
        "sigma_kr",
        "sigma_kvphi",
        "sigma_kz",
        "sigma_kvz",
    ],
    meta_fields=[],
)
@dataclass
class Fardal2015DF(AbstractKinematicDF):
    """Fardal Stream Distribution Function.

    A class for representing the Fardal+2015 distribution function for
    generating stellar streams based on Fardal et al. 2015
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract

    Examples
    --------
    >>> import jax.random as jr
    >>> import galax.potential as gp

    >>> df = Fardal2015DF()
    >>> df
    Fardal2015DF(kr_bar=Array(2., dtype=float64, ...),
        kvphi_bar=Array(0.3, dtype=float64, ...),
        kz_bar=Array(0., dtype=float64, ...),
        kvz_bar=Array(0., dtype=float64, ...),
        sigma_kr=Array(0.5, dtype=float64, ...),
        sigma_kvphi=Array(0.5, dtype=float64, ...),
        sigma_kz=Array(0.5, dtype=float64, ...),
        sigma_kvz=Array(0.5, dtype=float64, ...))

    >>> key = jr.key(0)
    >>> pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    >>> t = 0
    >>> x = jnp.array([15.0, 0.0, 0.0])
    >>> v = jnp.array([0.0, 220.0, 0.0])
    >>> Msat = 1e5
    >>> df.sample(key, pot, t, x, v, Msat)
    (Array([1.49982843e+01, 0.00000000e+00, 2.79332017e-04], ...),
     Array([ 0.00000000e+00,  2.20012478e+02, -1.66486551e-02], ...),
     Array([1.50017157e+01, 0.00000000e+00, 2.79332017e-04], ...),
     Array([ 0.00000000e+00,  2.19987522e+02, -1.66486551e-02], ...))

    """

    kr_bar: gt.Sz0 = field(default=default_kr_bar, converter=array_unless_jitting)
    kvphi_bar: gt.Sz0 = field(default=default_kvphi_bar, converter=array_unless_jitting)
    kz_bar: gt.Sz0 = field(default=default_kz_bar, converter=array_unless_jitting)
    kvz_bar: gt.Sz0 = field(default=default_kvz_bar, converter=array_unless_jitting)
    sigma_kr: gt.Sz0 = field(default=default_sigma_kr, converter=array_unless_jitting)
    sigma_kvphi: gt.Sz0 = field(
        default=default_sigma_kvphi, converter=array_unless_jitting
    )
    sigma_kz: gt.Sz0 = field(default=default_sigma_kz, converter=array_unless_jitting)
    sigma_kvz: gt.Sz0 = field(default=default_sigma_kvz, converter=array_unless_jitting)

    @ft.partial(jax.jit)
    def sample(
        self: "Fardal2015DF",
        key: PRNGKeyArray,
        pot: gp.AbstractPotential,
        /,
        t: gt.LikeSz0,
        x: gt.Sz3,
        v: gt.Sz3,
        Msat: gt.LikeSz0,
    ) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
        # ---------------------------------

        key1, key2, key3, key4 = jr.split(key, 4)

        Omega = omega(x, v)  # orbital angular frequency about the origin
        r_tidal = tidal_radius(pot, x, v, mass=Msat, t=t)  # tidal radius
        v_circ = Omega * r_tidal  # relative velocity

        # unit vectors
        r_hat = cx.vecs.normalize_vector(x)
        z_hat = cx.vecs.normalize_vector(jnp.linalg.cross(x, v))
        phi_vec = v - jnp.sum(v * r_hat) * r_hat
        phi_hat = cx.vecs.normalize_vector(phi_vec)

        # k vals
        shape = r_tidal.shape
        kr_samp = self.kr_bar + jr.normal(key1, shape) * self.sigma_kr
        kvphi_samp = kr_samp * (
            self.kvphi_bar + jr.normal(key2, shape) * self.sigma_kvphi
        )
        kz_samp = self.kz_bar + jr.normal(key3, shape) * self.sigma_kz
        kvz_samp = self.kvz_bar + jr.normal(key4, shape) * self.sigma_kvz

        # Leading arm
        x_lead = x - r_tidal * (kr_samp * r_hat - kz_samp * z_hat)
        v_lead = v - v_circ * (kvphi_samp * phi_hat - kvz_samp * z_hat)

        # Trailing arm
        x_trail = x + r_tidal * (kr_samp * r_hat + kz_samp * z_hat)
        v_trail = v + v_circ * (kvphi_samp * phi_hat + kvz_samp * z_hat)

        return x_lead, v_lead, x_trail, v_trail
