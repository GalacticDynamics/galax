"""Functions related to computing cluster relaxation times.

This is public API.

"""

__all__ = [
    "relaxation_time",
    "AbstractRelaxationTimeMethod",
    # specific methods
    "Baumgardt1998",
    "relaxation_time_baumgardt1998",
    "Spitzer1987HalfMass",
    "half_mass_relaxation_time_spitzer1987",
    "Spitzer1987Core",
    "core_relaxation_time_spitzer1987",
]

from functools import partial
from typing import Annotated as Antd, NoReturn, final
from typing_extensions import Doc

import jax
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u

import galax.typing as gt


class AbstractRelaxationTimeMethod:
    """Abstract base class for relaxation time flags.

    Examples
    --------
    >>> import galax.dynamics.cluster as gdc

    >>> try: gdc.relax_time.AbstractRelaxationTimeMethod()
    ... except TypeError as e: print(e)
    Cannot instantiate AbstractRelaxationTimeMethod

    """

    def __new__(cls) -> NoReturn:
        msg = "Cannot instantiate AbstractRelaxationTimeMethod"
        raise TypeError(msg)


@dispatch
def relaxation_time(
    M: u.AbstractQuantity,
    r_hm: u.AbstractQuantity,
    m_avg: u.AbstractQuantity,
    /,
    *,
    G: u.AbstractQuantity,
) -> u.AbstractQuantity:
    """Compute relaxation time, defaulting to Baumgardt (1998) formula."""
    return relaxation_time_baumgardt1998(M, r_hm, m_avg, G=G)


######################################################################
# Baumgardt (1998) relaxation time
# TODO: I don't think this is the original reference


@final
class Baumgardt1998(AbstractRelaxationTimeMethod):
    r"""Relaxation time from Baumgardt (1998).

    $$ t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \\ln(0.4 N)} $$

    """


@dispatch
def relaxation_time(
    _: type[Baumgardt1998],
    M: u.AbstractQuantity,
    r_hm: u.AbstractQuantity,
    m_avg: u.AbstractQuantity,
    /,
    *,
    G: u.AbstractQuantity,
) -> u.AbstractQuantity:
    """Compute relaxation time using Baumgardt (1998) formula."""
    return relaxation_time_baumgardt1998(M, r_hm, m_avg, G=G)


# ---------------------------


@partial(jax.jit)
def relaxation_time_baumgardt1998(
    M: Antd[u.AbstractQuantity, Doc("mass of the cluster")],
    r_hm: Antd[u.AbstractQuantity, Doc("half-mass radius of the cluster")],
    m_avg: Antd[u.AbstractQuantity, Doc("average stellar mass")],
    /,
    G: Antd[u.AbstractQuantity, Doc("gravitational constant")],
) -> u.AbstractQuantity:
    r"""Compute the cluster's relaxation time.

    Baumgardt 1998 Equation 1.

    $$
        t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \ln(0.4 N)}
    $$

    where $N$ is the number of stars in the cluster, $M_c$ is the mass of the
    cluster, $r_{hm}$ is the half-mass radius of the cluster, $m_{avg}$ is the
    average stellar mass, and $G$ is the gravitational constant.

    """
    N = M / m_avg
    return 0.138 * jnp.sqrt(M * r_hm**3 / G / m_avg**2) / jnp.log(0.4 * N)


######################################################################
# Spitzer 1987 relaxation time


@final
class Spitzer1987HalfMass(AbstractRelaxationTimeMethod):
    r"""Half-mass relaxation time from Spitzer (1987).

    $$ t_{rh} \approx \frac{0.17 N}{\ln(\Lambda)} \sqrt{\frac{r_h^3}{G M}} $$

    """


@final
class Spitzer1987Core(AbstractRelaxationTimeMethod):
    r"""Core relaxation time from Spitzer (1987).

    $$ t_{rc} \approx \frac{0.34 N}{\ln(\Lambda)} \sqrt{\frac{r_c^3}{G M_c}} $$

    """


@dispatch
def relaxation_time(
    _: type[Spitzer1987HalfMass],
    M: u.AbstractQuantity,
    r_hm: u.AbstractQuantity,
    m_avg: u.AbstractQuantity,
    /,
    *,
    G: u.AbstractQuantity,
    lnLambda: gt.RealScalarLike,
) -> u.AbstractQuantity:
    """Compute relaxation time using Spitzer (1987) formula."""
    return half_mass_relaxation_time_spitzer1987(M, r_hm, m_avg, G=G, lnLambda=lnLambda)


@dispatch
def relaxation_time(
    _: type[Spitzer1987Core],
    M_core: u.AbstractQuantity,
    r_core: u.AbstractQuantity,
    m_avg: u.AbstractQuantity,
    /,
    *,
    G: u.AbstractQuantity,
    lnLambda: gt.RealScalarLike,
) -> u.AbstractQuantity:
    """Compute relaxation time using Spitzer (1987) formula."""
    return core_relaxation_time_spitzer1987(
        M_core, r_core, m_avg, G=G, lnLambda=lnLambda
    )


# ---------------------------


def _relaxation_time_spitzer1987(
    M: u.AbstractQuantity,
    r: u.AbstractQuantity,
    m_avg: u.AbstractQuantity,
    prefactor: float,
    lnLambda: gt.RealScalarLike,
    G: u.AbstractQuantity,
) -> u.AbstractQuantity:
    N = M / m_avg
    return jnp.sqrt(r**3 / G / M) * prefactor * N / lnLambda


@partial(jax.jit)
def half_mass_relaxation_time_spitzer1987(
    M: Antd[u.AbstractQuantity, Doc("mass of the cluster")],
    r_hm: Antd[u.AbstractQuantity, Doc("half-mass radius of the cluster")],
    m_avg: Antd[u.AbstractQuantity, Doc("average stellar mass")],
    /,
    G: Antd[u.AbstractQuantity, Doc("gravitational constant")],
    lnLambda: Antd[gt.RealScalarLike, Doc("Coulomb logarithm")],
) -> u.AbstractQuantity:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 1.

    .. math::

        t_r = \frac{0.1 N}{\ln(0.4 N)} \frac{r_{hm}^3}{G M}

    """
    return _relaxation_time_spitzer1987(
        M, r_hm, m_avg, prefactor=0.17, lnLambda=lnLambda, G=G
    )


@partial(jax.jit)
def core_relaxation_time_spitzer1987(
    Mc: Antd[u.AbstractQuantity, Doc("mass of the cluster")],
    r_c: Antd[u.AbstractQuantity, Doc("core radius of the cluster")],
    m_avg: Antd[u.AbstractQuantity, Doc("average stellar mass")],
    /,
    G: Antd[u.AbstractQuantity, Doc("gravitational constant")],
    lnLambda: Antd[gt.RealScalarLike, Doc("Coulomb logarithm")],
) -> u.AbstractQuantity:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 2.

    .. math::

        t_r = \frac{0.2 N}{\ln(0.4 N)} \frac{r_c^3}{G M_c}

    """
    return _relaxation_time_spitzer1987(
        Mc, r_c, m_avg, prefactor=0.34, lnLambda=lnLambda, G=G
    )
