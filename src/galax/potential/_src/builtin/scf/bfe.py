"""Self-Consistent Field basis functions."""

__all__ = ["SCFPotential", "phi_nl", "rho_nl"]

import functools as ft
from dataclasses import KW_ONLY

from jaxtyping import Array, Float
from typing import final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .gegenbauer import gegenbauer_all
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.builtin.multipole import (
    iter_Ylm,
    scaled_radius_and_direction,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

SQRT_FOURPI = 3.544907701811031
"""``sqrt(4 * pi)``, matching the literal in gala's ``bfe_helper.cpp``."""


def _nl_axes(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Broadcastable ``n``, ``l`` and Gegenbauer table for the given ``s``."""
    nbatch = jnp.ndim(s)
    ls = jnp.arange(lmax + 1, dtype=s.dtype)
    # n gets one axis for l plus one per batch axis; l gets one per batch axis.
    n = jnp.expand_dims(
        jnp.arange(nmax + 1, dtype=s.dtype), tuple(range(1, 2 + nbatch))
    )
    l = jnp.expand_dims(ls, tuple(range(1, 1 + nbatch)))
    cn = gegenbauer_all(nmax, 2 * ls + 1.5, (s - 1) / (s + 1))
    return n, l, cn


def _s_pow_l(
    lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{lmax}+1 *batch"]:
    r""":math:`s^l` for :math:`l = 0 \ldots l_{max}`, stacked on a leading axis.

    Built with *integer* exponents rather than as ``s ** l`` against a float
    ``l`` array. The two agree in value, but the float form is not twice
    differentiable at :math:`s = 0`: the derivative of :math:`s^0` is
    :math:`0 \cdot s^{-1}`, which JAX evaluates as ``0 * inf`` and returns
    ``nan``, so the hessian of every SCF potential was NaN at the origin even
    though the value and gradient were finite. ``lmax`` is static, so the
    comprehension unrolls at trace time and `jax` uses `lax.integer_pow`,
    whose derivative at zero is exact.
    """
    return jnp.stack([s**i for i in range(lmax + 1)])  # type: ignore[no-any-return]


@ft.partial(jax.jit, static_argnums=(0, 1))
def phi_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial potential expansion terms.

    $$ \phi_{nl}(s) = -\sqrt{4\pi} \frac{s^l}{(1+s)^{2l+1}} C_n^{2l+3/2}(\xi) $$

    with $\xi = (s-1)/(s+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import phi_nl

    The $n = l = 0$ term is the Hernquist profile up to normalization:

    >>> bool(jnp.allclose(phi_nl(0, 0, jnp.asarray(1.0)),
    ...                   -3.544907701811031 / 2))
    True

    """
    _, l, cn = _nl_axes(nmax, lmax, s)
    prefactor = -SQRT_FOURPI * _s_pow_l(lmax, s) / (1 + s) ** (2 * l + 1)
    return prefactor * cn


@ft.partial(jax.jit, static_argnums=(0, 1))
def rho_nl(
    nmax: int, lmax: int, s: Float[Array, "*batch"], /
) -> Float[Array, "{nmax}+1 {lmax}+1 *batch"]:
    r"""Radial density expansion terms.

    $$ \rho_{nl}(s) = \sqrt{4\pi} \frac{K_{nl}}{2\pi}
                      \frac{s^l}{s(1+s)^{2l+3}} C_n^{2l+3/2}(\xi) $$

    with $K_{nl} = \frac{1}{2}n(n+4l+3) + (l+1)(2l+1)$.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from galax.potential.scf import rho_nl

    >>> rho_nl(0, 0, jnp.asarray(1.0)).shape
    (1, 1)

    """
    n, l, cn = _nl_axes(nmax, lmax, s)
    knl = 0.5 * n * (n + 4 * l + 3) + (l + 1) * (2 * l + 1)
    prefactor = (
        SQRT_FOURPI
        * (knl / (2 * jnp.pi))
        * _s_pow_l(lmax, s)
        / (s * (1 + s) ** (2 * l + 3))
    )
    return prefactor * cn  # type: ignore[no-any-return]


@final
class SCFPotential(AbstractSinglePotential):
    r"""Self-Consistent Field (SCF) basis function expansion potential.

    The method of Hernquist & Ostriker (1992) and Lowing et al. (2011), with
    all coefficients real.

    $$ \Phi(r,\theta,\phi) = \frac{G M}{r_s} \sum_{nlm} \phi_{nl}(s)
       \left[ S_{nlm} \Re Y_l^m + T_{nlm} \Im Y_l^m \right] $$

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    The monopole term alone is the Hernquist potential:

    >>> Snlm = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
    >>> pot = gp.SCFPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(10.0, "kpc"),
    ...                       Snlm=Snlm, Tnlm=jnp.zeros_like(Snlm),
    ...                       units="galactic")
    >>> pot.nmax, pot.lmax
    (0, 0)

    """

    m_tot: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Scale mass."
    )
    r_s: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Scale radius."
    )
    Snlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\cos(m\phi)$ terms, shape "
        r"``(nmax+1, lmax+1, lmax+1)``.",
    )
    Tnlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Expansion coefficients for the $\sin(m\phi)$ terms, shape "
        r"``(nmax+1, lmax+1, lmax+1)``.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    nmax: int = eqx.field(init=False, static=True, repr=False)
    lmax: int = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        # NOTE: must call super() -- it applies the unit system. (Do not copy
        # nfw/triaxial.py's __post_init__, which omits this.)
        super().__post_init__()
        shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        object.__setattr__(self, "nmax", shape[0] - 1)
        object.__setattr__(self, "lmax", shape[1] - 1)

    def __check_init__(self) -> None:
        s_shape = self.Snlm(u.Q(0.0, "Gyr")).shape
        t_shape = self.Tnlm(u.Q(0.0, "Gyr")).shape
        if s_shape != t_shape:
            msg = (
                "Snlm and Tnlm must have the same shape. "
                f"Got {s_shape} and {t_shape}."
            )
            raise ValueError(msg)
        if len(s_shape) != 3 or s_shape[1] != s_shape[2]:
            msg = (
                "Snlm and Tnlm must have shape (nmax+1, lmax+1, lmax+1). "
                f"Got {s_shape}."
            )
            raise ValueError(msg)

    # ==========================================================================

    def _summation(
        self,
        radial: Float[Array, "{nmax}+1 {lmax}+1 *batch"],
        Snlm: Float[Array, "n l m"],
        Tnlm: Float[Array, "n l m"],
        uvec: gt.BtSz3,
        /,
    ) -> gt.BBtSz0:
        r"""Sum :math:`\sum_{nlm} R_{nl}(S_{nlm}\Re Y_l^m + T_{nlm}\Im Y_l^m)`.

        Shared by `_potential` and `_density`, which differ only in the radial
        factor :math:`R_{nl}` -- `phi_nl` or `rho_nl` -- and the prefactor
        outside.

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
        # `iter_Ylm` yields m-major; this needs l-major, since the radial
        # contraction is shared across the `m` of a given `l`. Bucketing first
        # (rather than folding a dict) keeps every term on the accumulator
        # below a plain running sum, with no intermediate collection kept
        # around for XLA to materialize.
        by_l: list[list[tuple[int, gt.BtFloatSz0, gt.BtFloatSz0]]] = [
            [] for _ in range(self.lmax + 1)
        ]
        for l, m, cY, sY in iter_Ylm(self.lmax, uvec):
            by_l[l].append((m, cY, sY))

        total = jnp.zeros(jnp.shape(uvec)[:-1])
        for l, ms in enumerate(by_l):
            # One matvec per l: (nmax+1, m) against (nmax+1, *batch).
            Sl = jnp.einsum("nm,n...->m...", Snlm[:, l, : l + 1], radial[:, l])
            Tl = jnp.einsum("nm,n...->m...", Tnlm[:, l, : l + 1], radial[:, l])
            for m, cY, sY in ms:
                total = total + Sl[m] * cY + Tl[m] * sY

        return total  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        ud = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        Snlm = self.Snlm(t, ustrip=ud)
        Tnlm = self.Tnlm(t, ustrip=ud)

        s, uvec = scaled_radius_and_direction(xyz, r_s)
        phinl = phi_nl(self.nmax, self.lmax, s)
        summation = self._summation(phinl, Snlm, Tnlm, uvec)

        return self.constants["G"].value * m_tot / r_s * summation  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.Q.from_(t, self.units["time"])

        ud = self.units["dimensionless"]
        m_tot = self.m_tot(t, ustrip=self.units["mass"])
        r_s = self.r_s(t, ustrip=self.units["length"])
        Snlm = self.Snlm(t, ustrip=ud)
        Tnlm = self.Tnlm(t, ustrip=ud)

        s, uvec = scaled_radius_and_direction(xyz, r_s)
        rhonl = rho_nl(self.nmax, self.lmax, s)
        summation = self._summation(rhonl, Snlm, Tnlm, uvec)

        return m_tot / r_s**3 * summation  # type: ignore[no-any-return]
