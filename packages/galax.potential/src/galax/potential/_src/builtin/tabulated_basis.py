"""Spherical basis-function expansion with a tabulated radial basis."""

__all__ = ["TabulatedRadialBasisPotential"]

from dataclasses import KW_ONLY

from jaxtyping import Array, Float
from typing import Any, final

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from galax.potential._src.base import default_constants
from galax.potential._src.builtin.multipole import scaled_radius_and_direction
from galax.potential._src.builtin.sph_expansion import (
    AbstractRadialBasisExpansionPotential,
)
from galax.potential._src.harmonic.spline import eval_log_spline, fit_log_spline
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class TabulatedRadialBasisPotential(AbstractRadialBasisExpansionPotential):
    r"""Spherical basis-function expansion whose radial basis is tabulated.

    .. math::

        \Phi(r, \theta, \phi, t) = \sum_{nlm} u_{nl}(r)
            \left[ S_{nlm}(t)\, \Re Y_l^m(\theta, \phi)
                 + T_{nlm}(t)\, \Im Y_l^m(\theta, \phi) \right]

    with :math:`Y_l^m` the orthonormal spherical harmonics of `SCFPotential`
    (evaluated in Cartesian form, so derivatives are finite on the z-axis)
    and :math:`u_{nl}(r)` given at radial knots. `SCFPotential` is the special
    case of an analytic Hernquist basis; this class takes any basis known only
    as a table: an empirical basis from an N-body code such as EXP, or the
    numerical solution of a Sturm-Liouville problem.

    The basis is interpolated with a cubic spline in :math:`\ln r` (the
    not-a-knot spline of `MultipoleProfilePotential`), so the potential is
    :math:`C^2` inside the knot range and its gradient and hessian come from
    autodiff. Accuracy follows the knot spacing :math:`h` in :math:`\ln r`:
    :math:`O(h^4)` for the potential, :math:`O(h^3)` for the force and
    :math:`O(h^2)` for the hessian. With 600 knots over six decades (h =
    0.023) and ``nmax = lmax = 4``, a tabulated Hernquist basis reproduces
    `SCFPotential` to better than 1e-6 in the potential, 3e-5 in the force
    and 1e-3 in the hessian, between 0.01 and 1000 kpc.

    **Outside the knot range** each term continues as the solution of
    Laplace's equation: :math:`u_{nl} \propto r^{-(l+1)}` beyond the
    outermost knot, as EXP's spherical basis does (exact when the basis meets
    the vacuum boundary condition there), and :math:`u_{nl} \propto r^l`
    inside the innermost, so the potential and force stay finite at the
    origin. Both are joined to the table in value, not in slope.

    The coefficients may depend on time (any `AbstractParameter`), which is how
    a time series of snapshots from a simulation becomes a potential: for
    example a `CustomParameter` that interpolates the snapshots linearly.

    Build one with `from_knots`, which fits the spline's knot derivatives,
    rather than by passing ``du_nl`` directly.

    Examples
    --------
    A monopole Hernquist basis, tabulated from `SCFPotential`'s own radial
    function, reproduces `SCFPotential`:

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> from galax.potential.scf import phi_nl
    >>> m, r_s = 1e12, 10.0
    >>> G = 4.498502151469554e-12  # kpc^3 / (Msun Myr^2)
    >>> r = jnp.logspace(-2, 3, 400)
    >>> table = G * m / r_s * jnp.moveaxis(phi_nl(0, 0, r / r_s), -1, 0)
    >>> S = jnp.ones((1, 1, 1))
    >>> pot = gp.TabulatedRadialBasisPotential.from_knots(
    ...     u.Q(r, "kpc"), u.Q(table, "kpc2/Myr2"), S, jnp.zeros_like(S),
    ...     units="galactic")
    >>> scf = gp.SCFPotential(m_tot=u.Q(m, "Msun"), r_s=u.Q(r_s, "kpc"),
    ...                       Snlm=S, Tnlm=jnp.zeros_like(S), units="galactic")
    >>> x = u.Q([8.0, 1.0, 2.0], "kpc")
    >>> bool(jnp.isclose(pot.potential(x, 0.0), scf.potential(x, 0.0),
    ...                  rtol=1e-8, atol=u.Q(0.0, "kpc2/Myr2")))
    True

    """

    r_knots: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Radial knots, positive and strictly increasing."
    )
    u_nl: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="specific energy",
        doc=r"Radial basis :math:`u_{nl}` at the knots, "
        r"shape ``(n_r, nmax+1, lmax+1)``.",
    )
    du_nl: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="specific energy",
        doc=r":math:`du_{nl}/d\ln r` at the knots, same shape as ``u_nl``.",
    )
    Snlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Coefficients of :math:`\Re Y_l^m`, shape ``(nmax+1, lmax+1, lmax+1)``.",
    )
    Tnlm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=r"Coefficients of :math:`\Im Y_l^m`, shape ``(nmax+1, lmax+1, lmax+1)``.",
    )

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, u.AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    nmax: int = eqx.field(init=False, static=True, repr=False)
    lmax: int = eqx.field(init=False, static=True, repr=False)

    def __check_init__(self) -> None:
        # The coefficient shapes are checked by AbstractRadialBasisExpansionPotential.
        t0 = u.Q(0.0, "Gyr")
        s_shape = self.Snlm(t0).shape
        r_knots = self.r_knots(t0)
        if r_knots.ndim != 1 or r_knots.shape[0] < 4:
            msg = (
                "r_knots must be 1-D with at least 4 entries, "
                f"got shape {r_knots.shape}."
            )
            raise ValueError(msg)
        expected = (r_knots.shape[0], s_shape[0], s_shape[1])
        for name in ("u_nl", "du_nl"):
            got = getattr(self, name)(t0).shape
            if got != expected:
                msg = (
                    f"{name} must have shape (n_r, nmax+1, lmax+1) = {expected}, "
                    f"got {got}."
                )
                raise ValueError(msg)
        if isinstance(r_knots.value, jax.core.Tracer):
            return
        if not bool(jnp.all(r_knots.value > 0)) or not bool(
            jnp.all(jnp.diff(r_knots.value) > 0)
        ):
            msg = "r_knots must be positive and strictly increasing."
            raise ValueError(msg)

    @classmethod
    def from_knots(
        cls,
        r_knots: Any,
        u_nl: Any,
        Snlm: Any,
        Tnlm: Any,
        /,
        *,
        units: Any,
        **kwargs: Any,
    ) -> "TabulatedRadialBasisPotential":
        """Build from a tabulated basis, fitting the spline's knot derivatives.

        ``r_knots`` (length) and ``u_nl`` (specific energy, shape
        ``(n_r, nmax+1, lmax+1)``) are constant; ``Snlm`` and ``Tnlm`` may be
        any parameter, including a time-dependent one. ``kwargs`` go to the
        constructor (``units``, ``constants``).
        """
        units = u.unitsystem(units)
        r = u.ustrip(AllowValue, units["length"], r_knots)
        un = u.ustrip(AllowValue, units["specific energy"], u_nl)
        du = fit_log_spline(jnp.log(r), un)
        return cls(
            r_knots=u.Q(r, units["length"]),
            u_nl=u.Q(un, units["specific energy"]),
            du_nl=u.Q(du, units["specific energy"]),
            Snlm=Snlm,
            Tnlm=Tnlm,
            units=units,
            **kwargs,
        )

    def _radial_basis(
        self, xyz: gt.BBtSz3, p: dict[str, Array], /
    ) -> tuple[Float[Array, "{nmax}+1 {lmax}+1 *batch"], gt.BtSz3, Array | float]:
        r, uvec = scaled_radius_and_direction(xyz, jnp.ones((), dtype=xyz.dtype))
        log_r, lq = jnp.log(p["r_knots"]), jnp.log(r)
        radial = eval_log_spline(  # (*batch, n, l)
            log_r, p["u_nl"], p["du_nl"], jnp.clip(lq, log_r[0], log_r[-1])
        )
        # Outside the knots: r^-(l+1) beyond the last, r^l inside the first.
        ell = jnp.arange(self.lmax + 1, dtype=radial.dtype)
        out = -(ell + 1) * jnp.maximum(lq - log_r[-1], 0.0)[..., None]
        inn = ell * jnp.minimum(lq - log_r[0], 0.0)[..., None]
        radial = radial * jnp.exp(out + inn)[..., None, :]
        return jnp.moveaxis(radial, (-2, -1), (0, 1)), uvec, 1.0
