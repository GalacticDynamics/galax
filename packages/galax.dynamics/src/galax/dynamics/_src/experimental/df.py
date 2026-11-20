"""Experimental dynamics."""

__all__: tuple[str, ...] = ()

import abc
import functools as ft

from jaxtyping import PRNGKeyArray
from typing import Any, Final, Self, final

import jax
import jax.random as jr
from jax.stages import ArgInfo
from jax.tree_util import register_dataclass

import coordinax as cx
import quaxed.numpy as jnp
from dataclassish import field_items
from dataclassish.converters import Unless, dataclass, field

import galax.dynamics.custom_types as gt
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

    def tree_flatten(self) -> tuple[dict[str, gt.Sz0], None]:
        return (dict(field_items(self)), None)  # type: ignore[call-overload]

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
    (Array([1.49977501e+01, 0.00000000e+00, 7.49832945e-04], dtype=float64),
     Array([ 0.00000000e+00,  2.19966471e+02, -4.78647352e-04], dtype=float64),
     Array([1.50037597e+01, 0.00000000e+00, 2.49694925e-04], dtype=float64),
     Array([0.00000000e+00, 2.19977919e+02, 6.34205795e-03], dtype=float64))

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
        v_circ = Omega * r_tidal  # type: ignore[operator] # relative velocity

        # unit vectors
        r_hat = cx.vecs.normalize_vector(x)
        z_hat = cx.vecs.normalize_vector(jnp.linalg.cross(x, v))
        phi_vec = v - jnp.sum(v * r_hat) * r_hat
        phi_hat = cx.vecs.normalize_vector(phi_vec)

        # k vals, a separate draw for each arm (index 0 trailing, 1 leading), as
        # in gala and the agama code of Chen+24.
        shape = (2, *r_tidal.shape)  # type: ignore[attr-defined]
        kr_samp = self.kr_bar + jr.normal(key1, shape) * self.sigma_kr
        kvphi_samp = kr_samp * (
            self.kvphi_bar + jr.normal(key2, shape) * self.sigma_kvphi
        )
        kz_samp = self.kz_bar + jr.normal(key3, shape) * self.sigma_kz
        kvz_samp = self.kvz_bar + jr.normal(key4, shape) * self.sigma_kvz

        # Leading arm
        x_lead = x - r_tidal * (kr_samp[1] * r_hat - kz_samp[1] * z_hat)
        v_lead = v - v_circ * (kvphi_samp[1] * phi_hat - kvz_samp[1] * z_hat)

        # Trailing arm
        x_trail = x + r_tidal * (kr_samp[0] * r_hat + kz_samp[0] * z_hat)
        v_trail = v + v_circ * (kvphi_samp[0] * phi_hat + kvz_samp[0] * z_hat)

        return x_lead, v_lead, x_trail, v_trail


##############################################################################

# Chen+24 Eq. 4: (Dr / r_t, phi, theta, Dv / v_esc, alpha, beta), angles in deg.
default_chen_mean: Final = (1.6, -30.0, 0.0, 1.0, 20.0, 0.0)
default_chen_cov: Final = (
    (0.1225, 0.0, 0.0, 0.0, -4.9, 0.0),
    (0.0, 529.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 144.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-4.9, 0.0, 0.0, 0.0, 400.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 484.0),
)


@final
@ft.partial(register_dataclass, data_fields=["mean", "cov"], meta_fields=[])
@dataclass
class Chen2024DF(AbstractKinematicDF):
    """Chen Stream Distribution Function.

    The Chen+2024 distribution function for generating stellar streams
    https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract
    Each arm is its own independent draw.

    Examples
    --------
    >>> import jax.random as jr
    >>> import galax.potential as gp

    >>> df = Chen2024DF()
    >>> pot = gp.NFWPotential(m=1e12, r_s=15, units="galactic")
    >>> x = jnp.array([15.0, 0.0, 0.0])
    >>> v = jnp.array([0.0, 220.0, 0.0])
    >>> x_lead, v_lead, x_trail, v_trail = df.sample(jr.key(0), pot, 0, x, v, 1e5)
    >>> x_lead.shape
    (3,)

    """

    mean: jax.Array = field(default=default_chen_mean, converter=array_unless_jitting)
    cov: jax.Array = field(default=default_chen_cov, converter=array_unless_jitting)

    @ft.partial(jax.jit)
    def sample(
        self: "Chen2024DF",
        key: PRNGKeyArray,
        pot: gp.AbstractPotential,
        /,
        t: gt.LikeSz0,
        x: gt.Sz3,
        v: gt.Sz3,
        Msat: gt.LikeSz0,
    ) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
        # Release frame: x_new along r, z_new along L, y_new in-plane.
        x_hat = cx.vecs.normalize_vector(x)
        z_hat = cx.vecs.normalize_vector(jnp.linalg.cross(x, v))
        y_hat = cx.vecs.normalize_vector(v - jnp.sum(v * x_hat) * x_hat)

        r_tidal = tidal_radius(pot, x, v, mass=Msat, t=t)
        GM = pot.constants["G"].value * Msat

        def offsets(key: PRNGKeyArray, /) -> tuple[gt.Sz3, gt.Sz3]:
            """Position and velocity offsets (along x_new, y_new, z_new)."""
            # "svd": "cholesky" fails on the singular covariance.
            posvel = jr.multivariate_normal(key, self.mean, self.cov, method="svd")
            Dr = posvel[0] * r_tidal  # type: ignore[operator]
            Dv = posvel[3] * jnp.sqrt(2 * GM / Dr)  # v_esc at Dr
            phi, theta, alpha, beta = jnp.deg2rad(posvel[jnp.array([1, 2, 4, 5])])
            dx = Dr * jnp.stack(
                [
                    jnp.cos(theta) * jnp.cos(phi),
                    jnp.cos(theta) * jnp.sin(phi),
                    jnp.sin(theta),
                ]
            )
            dv = Dv * jnp.stack(
                [
                    jnp.cos(beta) * jnp.cos(alpha),
                    jnp.cos(beta) * jnp.sin(alpha),
                    jnp.sin(beta),
                ]
            )
            return dx, dv

        key_trail, key_lead = jr.split(key)
        frame = jnp.stack([x_hat, y_hat, z_hat])  # (3, 3)

        dx, dv = offsets(key_trail)
        x_trail, v_trail = x + dx @ frame, v + dv @ frame

        # Leading arm: the same DF rotated by 180 deg about z_new.
        dx, dv = offsets(key_lead)
        flip = jnp.array([-1.0, -1.0, 1.0])
        x_lead, v_lead = x + (flip * dx) @ frame, v + (flip * dv) @ frame

        return x_lead, v_lead, x_trail, v_trail


def sample_releases(
    df: AbstractKinematicDF,
    key: PRNGKeyArray,
    pot: gp.AbstractPotential,
    ts: jax.Array,
    xs: gt.Sz3,
    vs: gt.Sz3,
    Msat: jax.Array,
) -> tuple[gt.Sz3, gt.Sz3, gt.Sz3, gt.Sz3]:
    """Sample ``n`` releases at once, one key per release (``jr.split(key, n)``).

    ``ts`` and ``Msat`` have shape ``(n,)``; ``xs`` and ``vs`` have shape
    ``(n, 3)``. Returns ``(x_lead, v_lead, x_trail, v_trail)``, each ``(n, 3)``.
    The key-to-draws map depends only on ``n``, never on parameter values.
    """
    keys = jr.split(key, jnp.shape(ts)[0])
    return jax.vmap(df.sample, in_axes=(0, None, 0, 0, 0, 0))(
        keys, pot, ts, xs, vs, Msat
    )
