"""Functions related to computing cluster relaxation times.

This is public API.

"""

__all__ = [
    "relaxation_time",
    "AbstractRelaxationTimeMethod",
    # specific methods
    "Baumgardt1998",
    "SpitzerHart1971",
    "Spitzer1987HalfMass",
    "Spitzer1987Core",
]

import abc
import functools as ft
from dataclasses import KW_ONLY
from typing import Annotated as Antd, Any, TypeAlias, TypeVar, cast, final
from typing_extensions import Doc

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity
from xmmutablemap import ImmutableMap

import galax._custom_types as gt
from galax.potential._src.base import default_constants

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


class AbstractRelaxationTimeMethod(eqx.Module):  # type: ignore[misc]
    """Abstract base class for relaxation time flags."""

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> BBtAorQSz0:
        pass


@dispatch.multi((gt.BBtSz0, gt.BBtSz0), (gt.BBtQuSz0, gt.BBtQuSz0))
def relaxation_time(
    M: BBtAorQSz0, r_hm: BBtAorQSz0, /, *, m_avg: BBtAorQSz0, **kw: Any
) -> BBtAorQSz0:
    """Compute relaxation time, defaulting to Baumgardt (1998) formula."""
    return relaxation_time(Baumgardt1998, M, r_hm, m_avg=m_avg, **kw)


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
    >>> trh = gdc.relax_time.SpitzerHart1971(m_avg=m_avg, gamma=0.11)(M, r_hm)
    >>> print(trh.uconvert("Myr"))
    Quantity['time'](176.0495246, unit='Myr')

    """

    m_avg: u.AbstractQuantity | ArrayLike
    """Average stellar mass."""

    gamma: float = 0.11
    """Constant in the Coulomb logarithm."""

    _: KW_ONLY
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def __call__(
        self,
        M: Antd[BBtAorQSz0, Doc("mass of the cluster")],
        r_hm: Antd[BBtAorQSz0, Doc("half-mass radius of the cluster")],
        /,
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

        Examples
        --------
        >>> import unxt as u
        >>> import galax.dynamics.cluster as gdc

        >>> M = u.Quantity(1e4, "Msun")
        >>> r_hm = u.Quantity(2, "pc")
        >>> m_avg = u.Quantity(0.42, "Msun")
        >>> trh = gdc.relax_time.SpitzerHart1971(m_avg=m_avg, gamma=0.11)(M, r_hm)
        >>> print(trh.uconvert("Myr"))
        Quantity['time'](176.0495246, unit='Myr')

        """
        G = self.constants["G"]  # TODO: unit detection
        N = M / self.m_avg
        coulomb_log = jnp.log(self.gamma * N)
        return 0.138 * jnp.sqrt(N * r_hm**3 / (G * self.m_avg)) / coulomb_log


@dispatch.multi(
    (type[SpitzerHart1971], gt.BBtSz0, gt.BBtSz0),
    (type[SpitzerHart1971], gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[SpitzerHart1971],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    /,
    m_avg: u.AbstractQuantity | ArrayLike,
    gamma: float = 0.11,
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
    ...     m_avg=m_avg, gamma=0.11)
    >>> print(trh.uconvert("Myr"))
    Quantity['time'](176.0495246, unit='Myr')

    """
    return SpitzerHart1971(m_avg=m_avg, gamma=gamma, **kw)(M, r_hm)


######################################################################
# Spitzer 1987 relaxation time


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


@final
class Spitzer1987HalfMass(AbstractRelaxationTimeMethod):
    r"""Half-mass relaxation time from Spitzer (1987).

    $$ t_{rh} \approx \frac{0.17 N}{\ln(\Lambda)} \sqrt{\frac{r_h^3}{G M}} $$

    """

    m_avg: u.AbstractQuantity | ArrayLike
    """Average stellar mass."""

    lnLambda: gt.RealScalarLike  # noqa: N815
    """Coulomb logarithm."""

    _: KW_ONLY
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def __call__(
        self,
        M: Antd[BBtAorQSz0, Doc("mass of the cluster")],
        r_hm: Antd[BBtAorQSz0, Doc("half-mass radius of the cluster")],
        /,
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
        >>> lnLambda = 10

        >>> func = gdc.relax_time.Spitzer1987HalfMass(m_avg, lnLambda=lnLambda)
        >>> func(M, r_hm).uconvert("Myr")
        Quantity(Array(143.38045171, dtype=float64), unit='Myr')

        The function also works with raw JAX arrays, in which case the
        inputs are assumed to be in compatible units:

        >>> func = gdc.relax_time.Spitzer1987HalfMass(m_avg.value, lnLambda=lnLambda, constants={"G": 0.00449})
        >>> func(M.value, r_hm.value)
        Array(143.51613833, dtype=float64, ...)

        """  # noqa: E501
        return _relaxation_time_spitzer1987(
            M,
            r_hm,
            self.m_avg,
            prefactor=0.17,
            lnLambda=self.lnLambda,
            G=self.constants["G"],
        )


@dispatch.multi(
    (type[Spitzer1987HalfMass], gt.BBtSz0, gt.BBtSz0),
    (type[Spitzer1987HalfMass], gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Spitzer1987HalfMass],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    /,
    *,
    m_avg: BBtAorQSz0,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Spitzer (1987) formula."""
    return Spitzer1987HalfMass(m_avg=m_avg, **kw)(M, r_hm)


# ---------------------------------------------------------


@final
class Spitzer1987Core(AbstractRelaxationTimeMethod):
    r"""Core relaxation time from Spitzer (1987).

    $$ t_{rc} \approx \frac{0.34 N}{\ln(\Lambda)} \sqrt{\frac{r_c^3}{G M_c}} $$

    """

    m_avg: u.AbstractQuantity | ArrayLike
    """Average stellar mass."""

    lnLambda: gt.RealScalarLike  # noqa: N815
    """Coulomb logarithm."""

    _: KW_ONLY
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def __call__(
        self,
        Mc: Antd[BBtAorQSz0, Doc("mass of the cluster")],
        r_c: Antd[BBtAorQSz0, Doc("core radius of the cluster")],
        /,
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
        >>> lnLambda = 10

        >>> func = gdc.relax_time.Spitzer1987Core(m_avg, lnLambda=lnLambda)
        >>> func(M, r_hm).uconvert("Myr")
        Quantity(Array(1.43380452, dtype=float64), unit='Myr')

        The function also works with raw JAX arrays, in which case the
        inputs are assumed to be in compatible units:

        >>> func = gdc.relax_time.Spitzer1987Core(m_avg.value, lnLambda=lnLambda, constants={"G": 0.00449})
        >>> func(M.value, r_hm.value)
        Array(1.43516138, dtype=float64, ...)

        """  # noqa: E501
        return _relaxation_time_spitzer1987(
            Mc,
            r_c,
            self.m_avg,
            prefactor=0.34,
            lnLambda=self.lnLambda,
            G=self.constants["G"],
        )


@dispatch.multi(
    (type[Spitzer1987Core], gt.BBtSz0, gt.BBtSz0),
    (type[Spitzer1987Core], gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Spitzer1987Core],
    M_core: BBtAorQSz0,
    r_core: BBtAorQSz0,
    /,
    *,
    m_avg: BBtAorQSz0,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Spitzer (1987) formula."""
    return Spitzer1987Core(m_avg=m_avg, **kw)(M_core, r_core)


######################################################################
# Baumgardt (1998) relaxation time
# TODO: I don't think this is the original reference


@final
class Baumgardt1998(AbstractRelaxationTimeMethod):
    r"""Relaxation time from Baumgardt (1998).

    $$ t_r = \frac{0.138 \sqrt{M_c} r_{hm}^{3/2}}{\sqrt{G} m_{avg} \\ln(0.4 N)} $$

    """

    m_avg: u.AbstractQuantity | ArrayLike
    """Average stellar mass."""
    _: KW_ONLY
    constants: ImmutableMap[str, u.AbstractQuantity | ArrayLike] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    @ft.partial(jax.jit)
    def __call__(
        self,
        M: Antd[BBtAorQSz0, Doc("mass of the cluster")],
        r_hm: Antd[BBtAorQSz0, Doc("half-mass radius of the cluster")],
        /,
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

        >>> gdc.relax_time.Baumgardt1998(m_avg)(M, r_hm).uconvert("Myr")
        Quantity(Array(129.50777927, dtype=float64), unit='Myr')

        The function also works with raw JAX arrays, in which case the
        inputs are assumed to be in compatible units:

        >>> func = gdc.relax_time.Baumgardt1998(m_avg.value, constants={"G": 0.00449})
        >>> func(M.value, r_hm.value)
        Array(129.63033763, dtype=float64, ...)

        """
        G = _check_types_match(self.constants["G"], M, name="G")
        N = M / self.m_avg
        return 0.138 * jnp.sqrt(N * r_hm**3 / (G * self.m_avg)) / jnp.log(0.4 * N)


@dispatch.multi(
    (type[Baumgardt1998], gt.BBtSz0, gt.BBtSz0),
    (type[Baumgardt1998], gt.BBtQuSz0, gt.BBtQuSz0),
)
def relaxation_time(
    _: type[Baumgardt1998],
    M: BBtAorQSz0,
    r_hm: BBtAorQSz0,
    /,
    *,
    m_avg: BBtAorQSz0,
    **kw: Any,
) -> BBtAorQSz0:
    """Compute relaxation time using Baumgardt (1998) formula."""
    return Baumgardt1998(m_avg=m_avg, **kw)(M, r_hm)
