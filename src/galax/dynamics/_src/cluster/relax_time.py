"""Functions related to computing cluster relaxation times.

This is public API.

"""

__all__ = [
    "relaxation_time",
    "AbstractRelaxationTimeMethod",
    # specific methods
    "Baumgardt1998",
    "relaxation_time_baumgardt1998",
    "SpitzerHart1971",
    "relaxation_time_spitzer_hart_1971",
    "Spitzer1987HalfMass",
    "half_mass_relaxation_time_spitzer1987",
    "Spitzer1987Core",
    "core_relaxation_time_spitzer1987",
]

import functools as ft
from typing import Any, NoReturn, TypeAlias, TypeVar, cast, final

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
    return cast("T", out)


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
# 10.1086/150855: Spitzer and Hart (1971) relaxation time
# Also in 10.1093/mnras/268.1.257


@final
class SpitzerHart1971(AbstractRelaxationTimeMethod):
    r"""Relaxation time from Spitzer and Hart (1971).

    $$
    t_{\mathrm{rh}} = \frac{0.138 \sqrt{N} r_{\mathrm{h}}^{3/2}}
                           {\sqrt{M G} \ln(\gamma N)}
    $$

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.42, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")
    >>> trh = gdc.relaxation_time(gdc.relax_time.SpitzerHart1971, M, r_hm,
    ...     m_avg=m_avg, gamma=0.11, G=G)
    >>> print(trh)
    Quantity['time'](176.21612725, unit='Myr')

    """


@dispatch.multi(
    (type[SpitzerHart1971], gt.BBtSz0, gt.BBtSz0),
    (type[SpitzerHart1971], gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[SpitzerHart1971],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    /,
    m_avg: BBtAorQSz0,
    **kw: Any,
) -> BBtAorQSz0:
    r"""Compute relaxation time using Spitzer and Hart (1971) formula.

    $$
    t_{\mathrm{rh}} = \frac{0.138 \sqrt{N} r_{\mathrm{h}}^{3/2}}
                           {\sqrt{M G} \ln(\gamma N)}
    $$

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.42, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")
    >>> trh = gdc.relaxation_time(gdc.relax_time.SpitzerHart1971, M, r_hm,
    ...     m_avg=m_avg, gamma=0.11, G=G)
    >>> print(trh)
    Quantity['time'](176.21612725, unit='Myr')

    """
    return relaxation_time_spitzer_hart_1971(M, r_hm, m_avg=m_avg, **kw)


# ---------------------------


@ft.partial(jax.jit)
def relaxation_time_spitzer_hart_1971(
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    /,
    *,
    m_avg: float = 0.42,
    gamma: float = 0.11,
    G: float = 0.00449,
) -> BBtAorQSz0:
    r"""Compute relaxation time using Spitzer and Hart (1971) formula.

    $$ t_{\mathrm{rh}} = \frac{0.138 \sqrt{N} r_{\mathrm{h}}^{3/2}}
                              {\sqrt{M G} \ln(\gamma N)}
    $$

    where:

    - $N = m / \bar{M}$ is the mean number of stars in the cluster,
    - $r_h$ is the half-mass radius of the cluster,
    - $\bar{M}$ is the mean stellar mass. For a Chabrier (2005) IMF between 0.08
      and 100 $M_{\odot}$ this is approximately 0.42 $M_{\odot}$,
    - $G$ is the gravitational constant,
    - $\ln(\gamma N)$ is the Coulomb logarithm. For equal-mass clusters (Giersz
      & Heggie 1994) $\gamma \sim 0.11$.

    Parameters
    ----------
    M : BBtAorQSz0
        Mass of the cluster.
    r_hm : BBtAorQSz0
        Half-mass radius of the cluster.
    m_avg : float, optional
        Mean stellar mass. Default is 0.42 (Chabrier 2005 IMF between 0.08 and
        100 $M_{\odot}$).
    gamma : float, optional
        Coulomb logarithm term. Default is 0.11 (Giersz & Heggie 1994).
    G : float, optional
        Gravitational constant.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.dynamics.cluster as gdc

    >>> M = u.Quantity(1e4, "Msun")
    >>> r_hm = u.Quantity(2, "pc")
    >>> m_avg = u.Quantity(0.42, "Msun")
    >>> G = u.Quantity(0.00449, "pc3 / (Myr2 Msun)")
    >>> trh = gdc.relax_time.relaxation_time_spitzer_hart_1971(
    ...     M, r_hm, m_avg=m_avg, gamma=0.11, G=G)
    >>> print(trh)
    Quantity['time'](176.21612725, unit='Myr')

    """
    N = M / m_avg
    return 0.138 * jnp.sqrt(N * r_hm**3 / (G * m_avg)) / jnp.log(gamma * N)


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
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    /,
    *,
    G: BBtAorQSz0,
    lnLambda: gt.RealScalarLike,
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 1.

    .. math::

        t_r = \frac{0.1 N}{\ln(0.4 N)} \frac{r_{hm}^3}{G M}

    Parameters
    ----------
    M : BBtAorQSz0
        Mass of the cluster.
    r_hm : BBtAorQSz0
        Half-mass radius of the cluster.
    m_avg : BBtAorQSz0
        Average stellar mass.
    G : BBtAorQSz0
        Gravitational constant.
    lnLambda : RealScalarLike
        Coulomb logarithm.

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
    Mc: BBtAorQSz0,
    r_c: BBtAorQSz0,
    m_avg: BBtAorQSz0,
    /,
    *,
    G: BBtAorQSz0,
    lnLambda: gt.RealScalarLike,
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Spitzer 1987 Equation 2.

    .. math::

        t_r = \frac{0.2 N}{\ln(0.4 N)} \frac{r_c^3}{G M_c}

    Parameters
    ----------
    Mc : BBtAorQSz0
        Mass of the cluster.
    r_c : BBtAorQSz0
        Core radius of the cluster.
    m_avg : BBtAorQSz0
        Average stellar mass.
    G : BBtAorQSz0
        Gravitational constant.
    lnLambda : RealScalarLike
        Coulomb logarithm.

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
    M: BBtAorQSz0, r_hm: BBtAorQSz0, m_avg: BBtAorQSz0, /, *, G: BBtAorQSz0
) -> BBtAorQSz0:
    r"""Compute the cluster's relaxation time.

    Baumgardt 1998 Equation 1.

    $$
        t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \ln(0.4 N)}
    $$

    where $N$ is the number of stars in the cluster, $M_c$ is the mass of the
    cluster, $r_{hm}$ is the half-mass radius of the cluster, $m_{avg}$ is the
    average stellar mass, and $G$ is the gravitational constant.

    Parameters
    ----------
    M : BBtAorQSz0
        Mass of the cluster.
    r_hm : BBtAorQSz0
        Half-mass radius of the cluster.
    m_avg : BBtAorQSz0
        Average stellar mass.
    G : BBtAorQSz0
        Gravitational constant.

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
