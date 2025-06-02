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

import functools as ft
from typing import Annotated as Antd, Any, NoReturn, TypeAlias, TypeVar, cast, final
from typing_extensions import Doc

import equinox as eqx
import jax
from plum import dispatch

import quaxed.numpy as jnp
from unxt.quantity import is_any_quantity

import galax._custom_types as gt

BBtAorQSz0: TypeAlias = gt.BBtSz0 | gt.BBtQuSz0
T = TypeVar("T", bound=gt.BBtSz0 | gt.BBtQuSz0)


def _check_types_match(obj: T, comparator: object, /, name: str) -> T:
    out = eqx.error_if(
        obj,
        (is_any_quantity(obj) and not is_any_quantity(comparator))
        or (not is_any_quantity(obj) and is_any_quantity(comparator)),
        f"{name} must be of type {'Quantity' if is_any_quantity(obj) else 'Array'}",
    )
    return cast(T, out)


#####################################################################


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


@dispatch.multi(
    (gt.BBtSz0, gt.BBtSz0, gt.BBtSz0),
    (gt.BBtQuSz0, gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    M: BBtAorQSz0, r_hm: BBtAorQSz0, m_avg: BBtAorQSz0, /, **kw: Any
) -> BBtAorQSz0:
    """Compute relaxation time, defaulting to Baumgardt (1998) formula."""
    return relaxation_time_baumgardt1998(M, r_hm, m_avg, **kw)


######################################################################
# Baumgardt (1998) relaxation time
# TODO: I don't think this is the original reference


@final
class Baumgardt1998(AbstractRelaxationTimeMethod):
    r"""Relaxation time from Baumgardt (1998).

    $$ t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \\ln(0.4 N)} $$

    """


@dispatch.multi(
    (type[Baumgardt1998], gt.BBtSz0, gt.BBtSz0, gt.BBtSz0),
    (type[Baumgardt1998], gt.BBtQuSz0, gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Baumgardt1998],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    /,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Baumgardt (1998) formula."""
    return relaxation_time_baumgardt1998(M, r_hm, m_avg, **kw)


# ---------------------------


@dispatch.multi(
    (gt.BBtSz0, gt.BBtSz0, gt.BBtSz0),
    (gt.BBtQuSz0, gt.BBtQuSz0, gt.BBtQuSz0),
)
@ft.partial(jax.jit)
def relaxation_time_baumgardt1998(
    M: Antd[BBtAorQSz0, Doc("mass of the cluster")],
    r_hm: Antd[BBtAorQSz0, Doc("half-mass radius of the cluster")],
    m_avg: Antd[BBtAorQSz0, Doc("average stellar mass")],
    /,
    *,
    G: Antd[BBtAorQSz0, Doc("gravitational constant")],
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Baumgardt 1998 Equation 1.

    $$
        t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \ln(0.4 N)}
    $$

    where $N$ is the number of stars in the cluster, $M_c$ is the mass of the
    cluster, $r_{hm}$ is the half-mass radius of the cluster, $m_{avg}$ is the
    average stellar mass, and $G$ is the gravitational constant.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.5, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")

    >>> gdc.relax_time.relaxation_time_baumgardt1998(M, r_hm, m_avg, G=G).uconvert("Myr")
    Quantity(Array(129.63033763, dtype=float64, ...), unit='Myr')

    The function also works with raw JAX arrays, in which case the
    inputs are assumed to be in compatible units:

    >>> gdc.relax_time.relaxation_time_baumgardt1998(M.value, r_hm.value, m_avg.value, G=0.00449)
    Array(129.63033763, dtype=float64, ...)

    """  # noqa: E501
    G = _check_types_match(G, M, name="G")
    N = M / m_avg
    return 0.138 * jnp.sqrt(N * r_hm**3 / (G * m_avg)) / jnp.log(0.4 * N)


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


@dispatch.multi(
    (type[Spitzer1987HalfMass], gt.BBtSz0, gt.BBtSz0, gt.BBtSz0),
    (type[Spitzer1987HalfMass], gt.BBtQuSz0, gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Spitzer1987HalfMass],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    /,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Spitzer (1987) formula."""
    return half_mass_relaxation_time_spitzer1987(M, r_hm, m_avg, **kw)


@dispatch.multi(
    (type[Spitzer1987Core], gt.BBtSz0, gt.BBtSz0, gt.BBtSz0),
    (type[Spitzer1987Core], gt.BBtQuSz0, gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Spitzer1987Core],
    M_core: BBtAorQSz0,
    r_core: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    /,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Spitzer (1987) formula."""
    return core_relaxation_time_spitzer1987(M_core, r_core, m_avg, **kw)


# ---------------------------


def _relaxation_time_spitzer1987(
    M: BBtAorQSz0,
    r: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    prefactor: float,
    lnLambda: gt.RealScalarLike,
    G: BBtAorQSz0,
) -> BBtAorQSz0:
    r = _check_types_match(r, M, name="r")
    m_avg = _check_types_match(m_avg, M, name="m_avg")
    G = _check_types_match(G, M, name="G")
    N = M / m_avg
    return jnp.sqrt(r**3 / G / M) * prefactor * N / lnLambda


@ft.partial(jax.jit)
def half_mass_relaxation_time_spitzer1987(
    M: Antd[BBtAorQSz0, Doc("mass of the cluster")],
    r_hm: Antd[BBtAorQSz0, Doc("half-mass radius of the cluster")],
    m_avg: Antd[BBtAorQSz0, Doc("average stellar mass")],
    /,
    *,
    G: Antd[BBtAorQSz0, Doc("gravitational constant")],
    lnLambda: Antd[gt.RealScalarLike, Doc("Coulomb logarithm")],
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 1.

    .. math::

        t_r = \frac{0.1 N}{\ln(0.4 N)} \frac{r_{hm}^3}{G M}

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.5, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")
    >>> lnLambda = 10

    >>> gdc.relax_time.half_mass_relaxation_time_spitzer1987(M, r_hm, m_avg, G=G, lnLambda=lnLambda).uconvert("Myr")
    Quantity(Array(143.51613833, dtype=float64, ...), unit='Myr')

    The function also works with raw JAX arrays, in which case the
    inputs are assumed to be in compatible units:

    >>> gdc.relax_time.half_mass_relaxation_time_spitzer1987(M.value, r_hm.value, m_avg.value, G=0.00449, lnLambda=lnLambda)
    Array(143.51613833, dtype=float64, ...)

    """  # noqa: E501
    return _relaxation_time_spitzer1987(
        M, r_hm, m_avg, prefactor=0.17, lnLambda=lnLambda, G=G
    )


@ft.partial(jax.jit)
def core_relaxation_time_spitzer1987(
    Mc: Antd[BBtAorQSz0, Doc("mass of the cluster")],
    r_c: Antd[BBtAorQSz0, Doc("core radius of the cluster")],
    m_avg: Antd[BBtAorQSz0, Doc("average stellar mass")],
    /,
    *,
    G: Antd[BBtAorQSz0, Doc("gravitational constant")],
    lnLambda: Antd[gt.RealScalarLike, Doc("Coulomb logarithm")],
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 2.

    .. math::

        t_r = \frac{0.2 N}{\ln(0.4 N)} \frac{r_c^3}{G M_c}

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(2e3, "Msun")
    >>> r_hm = u.Quantity(0.1, "pc")
    >>> m_avg = u.Quantity(0.5, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")
    >>> lnLambda = 10

    >>> gdc.relax_time.core_relaxation_time_spitzer1987(M, r_hm, m_avg, G=G, lnLambda=lnLambda).uconvert("Myr")
    Quantity(Array(1.43516138, dtype=float64, ...), unit='Myr')

    The function also works with raw JAX arrays, in which case the
    inputs are assumed to be in compatible units:

    >>> gdc.relax_time.core_relaxation_time_spitzer1987(M.value, r_hm.value, m_avg.value, G=0.00449, lnLambda=lnLambda)
    Array(1.43516138, dtype=float64, ...)

    """  # noqa: E501
    return _relaxation_time_spitzer1987(
        Mc, r_c, m_avg, prefactor=0.34, lnLambda=lnLambda, G=G
    )
