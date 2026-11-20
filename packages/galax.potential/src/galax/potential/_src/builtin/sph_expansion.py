"""Spherical-harmonic expansions and expansions in a radial basis."""

__all__ = [
    "AbstractRadialBasisExpansionPotential",
    "AbstractSphericalHarmonicExpansionPotential",
    "harmonic_sum",
    "sum_basis",
]

import abc
import functools as ft

from collections.abc import Callable
from jaxtyping import Array, Float

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.harmonic.ylm import iter_Ylm
from galax.potential._src.params.base import AbstractParameter

RadialFactors = Callable[
    [int], tuple[Float[Array, "l+1 *batch"], Float[Array, "l+1 *batch"]]
]


def harmonic_sum(lmax: int, factors: RadialFactors, uvec: gt.BtSz3, /) -> gt.BBtSz0:
    r"""Sum the harmonics against per-``l`` radial factors.

    .. math::

        \sum_{l \le l_\max} \sum_{m \le l} [C_{lm} \Re Y_l^m + S_{lm} \Im Y_l^m]

    ``factors(l)`` returns the radial factors :math:`(C_{l\cdot}, S_{l\cdot})`,
    each of shape ``(l+1, *batch)``. It is called once per ``l``, inside the
    loop, so that a child can fuse its own radial contraction per ``l`` (see
    `sum_basis`) instead of materializing an ``(l, m, *batch)`` grid.
    """
    # `iter_Ylm` yields m-major; this needs l-major, since the radial factors are
    # shared across the `m` of a given `l`. Bucketing first (rather than folding a
    # dict) keeps every term on the accumulator below a plain running sum, with no
    # intermediate collection kept around for XLA to materialize.
    by_l: list[list[tuple[int, gt.BtFloatSz0, gt.BtFloatSz0]]] = [
        [] for _ in range(lmax + 1)
    ]
    for l, m, cY, sY in iter_Ylm(lmax, uvec):
        by_l[l].append((m, cY, sY))

    total: Array | float = 0.0
    for l, ms in enumerate(by_l):
        Cl, Sl = factors(l)
        for m, cY, sY in ms:
            total = total + Cl[m] * cY + Sl[m] * sY

    return total  # type: ignore[return-value]


def basis_factors(
    radial: Float[Array, "nmax+1 lmax+1 *batch"],
    Snlm: Float[Array, "n l m"],
    Tnlm: Float[Array, "n l m"],
    /,
) -> RadialFactors:
    """Return the per-``l`` radial factors of a basis expansion, for `harmonic_sum`."""

    def factors(l: int) -> tuple[Array, Array]:
        # One matvec per l: (nmax+1, m) against (nmax+1, *batch).
        Sl = jnp.einsum("nm,n...->m...", Snlm[:, l, : l + 1], radial[:, l])
        Tl = jnp.einsum("nm,n...->m...", Tnlm[:, l, : l + 1], radial[:, l])
        return Sl, Tl

    return factors


def sum_basis(
    lmax: int,
    radial: Float[Array, "nmax+1 lmax+1 *batch"],
    Snlm: Float[Array, "n l m"],
    Tnlm: Float[Array, "n l m"],
    uvec: gt.BtSz3,
    /,
) -> gt.BBtSz0:
    r"""Sum :math:`\sum_{nlm} R_{nl}(S_{nlm}\Re Y_l^m + T_{nlm}\Im Y_l^m)`.

    Used by every `AbstractRadialBasisExpansionPotential` for its potential, and
    by `SCFPotential` for its density (the radial factor :math:`R_{nl}` is then
    `rho_nl` rather than the potential basis).

    Two things about the order of operations here are load-bearing, and
    both were measured rather than reasoned.

    **Each term is added directly onto a running total**, rather than
    scattering the harmonics into an ``(l, m, *batch)`` grid and
    contracting that with one `einsum`. The values are the same, but the
    grid must be *materialized* before it can be contracted, while a
    running sum never holds more than one term's worth of temporaries.

    **The contraction over** ``n`` **is done once per** ``l``, not once per
    ``(l, m)``. Both are the same arithmetic; the difference is that
    ``radial[:, l]`` -- which is ``(nmax+1, *batch)``, tens of megabytes at
    a large batch -- is then read once for each ``l`` rather than once for
    every ``m <= l``. Per-pair contraction was actually *slower* than the
    grid it replaced for the density at ``nmax = 24``, which is what
    surfaced this.

    Measured on the potential over 200,000 positions, against the grid:

    =================  ======  =======
    ``(nmax, lmax)``     grid   folded
    =================  ======  =======
    ``(6, 4)``          47 ms    18 ms
    ``(12, 6)``        129 ms    58 ms
    ``(24, 6)``        169 ms    85 ms
    ``(24, 12)``       967 ms   148 ms
    =================  ======  =======

    Values agree with the grid form to 3.4e-13 absolute at the largest
    order tested, and the potential is bit-identical at ``lmax = 6``.
    """
    return harmonic_sum(lmax, basis_factors(radial, Snlm, Tnlm), uvec)


class AbstractSphericalHarmonicExpansionPotential(AbstractSinglePotential):
    r"""A potential expanded in spherical harmonics.

    .. math::

        \Phi(r, \theta, \phi, t) = \sum_{l \le l_\max} \sum_{m \le l}
            \left[ C_{lm}(r, t)\, \Re Y_l^m(\theta, \phi)
                  + S_{lm}(r, t)\, \Im Y_l^m(\theta, \phi) \right]

    with :math:`Y_l^m` the orthonormal spherical harmonics, evaluated in
    Cartesian form so that derivatives are finite on the z-axis. The angular
    sum is shared (`harmonic_sum`); a subclass says how the radial factors
    :math:`C_{lm}, S_{lm}` are made, by implementing `_harmonic_terms`. In a
    radial basis (`AbstractRadialBasisExpansionPotential`) they are sums over
    basis functions; elsewhere they could be power laws or splined profiles.
    """

    lmax: eqx.AbstractVar[int]

    @abc.abstractmethod
    def _harmonic_terms(
        self, xyz: gt.BBtSz3, p: dict[str, Array], /
    ) -> tuple[RadialFactors, gt.BtSz3, Array | float]:
        """``(factors, unit direction, prefactor)`` of the potential at ``xyz``.

        ``xyz`` is in the potential's length unit and ``p`` holds the
        parameters at the requested time. The potential is ``prefactor *
        harmonic_sum(lmax, factors, unit direction)``.
        """

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        if jnp.ndim(t) > 0:  # a time per position: coefficients differ per time
            return jnp.vectorize(self._potential, signature="(3),()->()")(xyz, t)  # type: ignore[no-any-return]
        p = self.params(t)
        factors, uvec, prefactor = self._harmonic_terms(xyz, p)
        return prefactor * harmonic_sum(self.lmax, factors, uvec)


class AbstractRadialBasisExpansionPotential(
    AbstractSphericalHarmonicExpansionPotential
):
    r"""A spherical-harmonic expansion whose radial factors are sums over a basis.

    .. math::

        C_{lm}(r, t) = \sum_{n \le n_\max} S_{nlm}(t)\, u_{nl}(r), \qquad
        S_{lm}(r, t) = \sum_{n \le n_\max} T_{nlm}(t)\, u_{nl}(r)

    A subclass supplies the radial basis :math:`u_{nl}` by implementing
    `_radial_basis` and declares the ``Snlm`` and ``Tnlm`` parameters (shape
    ``(nmax+1, lmax+1, lmax+1)``; any parameter, including a time-dependent
    one) and the ``nmax`` and ``lmax`` fields, which this class fills in from
    the coefficient shape. `SCFPotential` has an analytic basis;
    `TabulatedRadialBasisPotential` takes one as a table.
    """

    Snlm: eqx.AbstractVar[AbstractParameter]
    Tnlm: eqx.AbstractVar[AbstractParameter]
    nmax: eqx.AbstractVar[int]

    def __post_init__(self) -> None:
        # NOTE: must call super() -- it applies the unit system.
        super().__post_init__()
        shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        object.__setattr__(self, "nmax", shape[0] - 1)
        object.__setattr__(self, "lmax", shape[1] - 1)

    def __check_init__(self) -> None:
        s_shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        t_shape = self.Tnlm(u.Q(0.0, "Gyr")).shape
        if s_shape != t_shape:
            msg = (
                f"Snlm and Tnlm must have the same shape. Got {s_shape} and {t_shape}."
            )
            raise ValueError(msg)
        if len(s_shape) != 3 or s_shape[1] != s_shape[2]:
            msg = (
                "Snlm and Tnlm must have shape (nmax+1, lmax+1, lmax+1). "
                f"Got {s_shape}."
            )
            raise ValueError(msg)

    @abc.abstractmethod
    def _radial_basis(
        self, xyz: gt.BBtSz3, p: dict[str, Array], /
    ) -> tuple[Float[Array, "nmax+1 lmax+1 *batch"], gt.BtSz3, Array | float]:
        """``(u_nl at |xyz|, unit direction, prefactor)``; see `_harmonic_terms`."""

    def _harmonic_terms(
        self, xyz: gt.BBtSz3, p: dict[str, Array], /
    ) -> tuple[RadialFactors, gt.BtSz3, Array | float]:
        radial, uvec, prefactor = self._radial_basis(xyz, p)
        return basis_factors(radial, p["Snlm"], p["Tnlm"]), uvec, prefactor
