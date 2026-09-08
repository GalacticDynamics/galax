r"""Zhao (1996) double power-law potential, with interpolated special functions.

`ZhaoPotential` evaluates the incomplete beta function of Zhao's Eq. 43 by
summing a 64-term series on every call, because its power-law indices are
ordinary parameters and so may vary. Most uses fix them: for a halo whose
$(\alpha, \beta, \gamma)$ never change, the model needs just two fixed
functions of one variable, and those can be fitted once at construction.

`InterpolatedZhaoPotential` does that, with the Chebyshev fit in
`galax.potential._src.special`. The mass and scale radius stay ordinary
(possibly time-dependent) parameters -- ``m`` enters only as a linear
normalization and ``r_s`` only through Zhao's $\chi$ (Eq. 5), so neither
affects the fit.

Everything else -- the equations, the analytic gradient/hessian/laplacian --
is shared with `ZhaoPotential`, so this differs only in how two numbers get
computed.
"""

__all__ = ["InterpolatedZhaoPotential"]

import functools as ft

from typing import Any, NamedTuple, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from .zhao import (
    _r_to_chi,
    d2potential_dr2_from,
    density_from_norm,
    gradient_from_radial,
    hessian_from_radial,
    mass_enclosed_from_bz,
    norm_from_bz_half,
    potential_from_bz,
)
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.special import ChebyshevIncompleteBeta
from galax.potential._src.utils import r_spherical

DimL = u.dimension("length")


class _Fit(NamedTuple):
    """The two fitted incomplete beta functions Zhao's Eq. 7 needs, plus B(.,.,1/2)."""

    interior: ChebyshevIncompleteBeta
    exterior: ChebyshevIncompleteBeta
    bz_half: float


@ft.cache
def _fitted(alpha: float, beta: float, gamma: float, n: int) -> _Fit:
    """Fit the two beta functions for these indices; memoized on them.

    The fit depends only on the power-law indices, which are static, so it is
    a constant of the model rather than part of its pytree -- and repeating a
    set of indices (a fit per orbit, say) costs nothing after the first.
    """
    # Zhao Eq. 42, in plain Python: this runs while tracing, where a jitted
    # helper would inline into the trace and hand back tracers.
    c0, p0, q0 = alpha * (beta - gamma), alpha * (2.0 - gamma), alpha * (beta - 3.0)
    interior = ChebyshevIncompleteBeta(c0 - q0, q0, n)
    exterior = ChebyshevIncompleteBeta(c0 - p0, p0, n)
    # chi(r_s) = 1/2 for any alpha (Eq. 5), so the normalization is a constant.
    return _Fit(interior, exterior, interior.at_half)


@final
class InterpolatedZhaoPotential(AbstractSinglePotential):
    r"""Zhao (1996) potential with the incomplete beta functions interpolated.

    Numerically equivalent to `ZhaoPotential` -- same equations, same analytic
    derivatives -- but with Zhao's Eq. 43 replaced by a Chebyshev fit made once
    at construction, which is worth roughly an order of magnitude per
    evaluation.

    The power-law indices are fixed at construction (they set what is fitted),
    so unlike `ZhaoPotential` they are plain numbers rather than parameters and
    cannot vary with time. ``m`` and ``r_s`` are ordinary parameters, as usual.

    Parameters
    ----------
    m, r_s
        As `ZhaoPotential`.
    alpha, beta, gamma
        As `ZhaoPotential`, but static: plain floats, not parameters.
    n_coeffs
        Chebyshev coefficients per panel. The default reaches ~1e-13 over the
        whole valid index range; see `ChebyshevIncompleteBeta`.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> kw = dict(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(10.0, "kpc"),
    ...           alpha=1.0, beta=4.0, gamma=1.0, units="galactic")
    >>> pot = gp.InterpolatedZhaoPotential(**kw)
    >>> exact = gp.ZhaoPotential(**kw)

    It agrees with `ZhaoPotential` to near machine precision:

    >>> xyz = u.Quantity([8.0, 0.0, 0.0], "kpc")
    >>> got = pot.potential(xyz, 0)
    >>> bool(jnp.isclose(got, exact.potential(xyz, 0),
    ...                  atol=u.Quantity(1e-12, got.unit)))
    True

    >>> got = pot.gradient(xyz, 0).x
    >>> bool(jnp.isclose(got, exact.gradient(xyz, 0).x,
    ...                  atol=u.Quantity(1e-12, got.unit)))
    True

    """

    m: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass", doc="Mass enclosed within the scale radius."
    )
    r_s: AbstractParameter = ParameterField(dimensions="length", doc="Scale radius.")  # type: ignore[assignment]

    alpha: float = eqx.field(static=True, converter=float)
    beta: float = eqx.field(static=True, converter=float)
    gamma: float = eqx.field(static=True, converter=float)
    n_coeffs: int = eqx.field(static=True, default=24, converter=int)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Fit now rather than on first evaluation, so that indices the fit
        # cannot represent are rejected here, where the traceback is useful.
        # The result is memoized, so `_fit` below is then free.
        _ = self._fit

    @property
    def _fit(self) -> "_Fit":
        """The fitted beta functions; see `_fitted`."""
        return _fitted(self.alpha, self.beta, self.gamma, self.n_coeffs)

    # ===========================================

    def _params(self, t: gt.BBtQorVSz0, /) -> gt.Params:
        """Build the parameter dict the shared `zhao` functions take."""
        t = u.Q.from_(t, self.units["time"])
        return {
            "G": self.constants["G"].value,
            "m": self.m(t, ustrip=self.units["mass"]),
            "r_s": self.r_s(t, ustrip=self.units["length"]),
            "alpha": jnp.asarray(self.alpha),
            "beta": jnp.asarray(self.beta),
            "gamma": jnp.asarray(self.gamma),
        }

    def _norm(self, p: gt.Params, /) -> gt.FloatSz0:
        return norm_from_bz_half(p, jnp.asarray(self._fit.bz_half))  # type: ignore[no-any-return]

    def _mass_enclosed(self, p: gt.Params, r: gt.BBtSz0, /) -> gt.BtFloatSz0:
        chi = _r_to_chi(p, r)
        return mass_enclosed_from_bz(p, self._norm(p), self._fit.interior(chi))  # type: ignore[no-any-return]

    # ===========================================

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        p = self._params(t)
        r = r_spherical(xyz, self.units["length"])
        chi = _r_to_chi(p, r)
        rho = density_from_norm(p, r, self._norm(p))
        fit = self._fit
        return potential_from_bz(  # type: ignore[no-any-return]
            p, r, chi, fit.interior(chi), fit.exterior(1.0 - chi), rho
        )

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BtFloatSz0:
        p = self._params(t)
        r = r_spherical(xyz, self.units["length"])
        return density_from_norm(p, r, self._norm(p))  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _laplacian(self, xyz: gt.BBtQorVSz3, /, t: gt.BBtQorVSz0) -> gt.BBtFloatSz0:
        return 4.0 * jnp.pi * self.constants["G"].value * self._density(xyz, t)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        p = self._params(t)
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        r = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        dphi_dr = p["G"] * self._mass_enclosed(p, r) / r**2
        return gradient_from_radial(dphi_dr, xyz, r)  # type: ignore[no-any-return]

    @vectorize_method(signature="(3),()->(3,3)")
    @ft.partial(jax.jit)
    def _hessian(
        self, xyz: gt.FloatQuSz3 | gt.FloatSz3, t: gt.QuSz0 | gt.Sz0, /
    ) -> gt.Sz33:
        p = self._params(t)
        xyz = u.ustrip(AllowValue, self.units[DimL], xyz)
        r = jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        m_enc = self._mass_enclosed(p, r)
        rho = density_from_norm(p, r, self._norm(p))
        dphi_dr = p["G"] * m_enc / r**2
        d2phi_dr2 = d2potential_dr2_from(p, r, m_enc, rho)
        return hessian_from_radial(dphi_dr, d2phi_dr2, xyz, r)  # type: ignore[no-any-return]

    # ===========================================

    @classmethod
    def from_zhao(
        cls, pot: Any, /, *, n_coeffs: int = 24
    ) -> "InterpolatedZhaoPotential":
        """Build from a `ZhaoPotential`, whose indices must be constant.

        Examples
        --------
        >>> import unxt as u
        >>> import galax.potential as gp

        >>> exact = gp.ZhaoPotential(m=u.Quantity(1e12, "Msun"),
        ...                          r_s=u.Quantity(10.0, "kpc"),
        ...                          alpha=1.0, beta=4.0, gamma=1.0,
        ...                          units="galactic")
        >>> gp.InterpolatedZhaoPotential.from_zhao(exact).beta
        4.0

        """
        t = u.Q(0.0, pot.units["time"])
        udim = pot.units["dimensionless"]
        return cls(
            m=pot.m,
            r_s=pot.r_s,
            alpha=float(pot.alpha(t, ustrip=udim)),
            beta=float(pot.beta(t, ustrip=udim)),
            gamma=float(pot.gamma(t, ustrip=udim)),
            n_coeffs=n_coeffs,
            units=pot.units,
            constants=pot.constants,
        )
