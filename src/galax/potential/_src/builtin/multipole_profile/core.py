"""Multipole profile potentials."""

__all__ = ["AbstractMultipoleProfilePotential", "MultipoleProfilePotential"]

from dataclasses import KW_ONLY

from collections.abc import Callable
from jaxtyping import Array, Float
from typing import Any, final

import jax.core
from equinox import field

import quaxed.numpy as jnp
import unxt as u
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .expansion import MultipoleProfileMixin
from .funcs import build_expansion, radial_grid
from .project import default_angular_resolution, lm_keys
from galax.potential._src.base import AbstractPotential, default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.constant import ConstantParameter
from galax.potential._src.params.field import ParameterField


class AbstractMultipoleProfilePotential(MultipoleProfileMixin, AbstractSinglePotential):
    r"""Abstract potential whose multipole coefficients are radial profiles.

    Unlike `galax.potential.MultipolePotential`, whose :math:`S_{lm}`,
    :math:`T_{lm}` are constants multiplying a fixed power of :math:`r` (an
    exact solid harmonic, hence zero density away from the origin), the
    coefficients here are *functions* :math:`\Phi_{lm}(r)` obtained by solving
    Poisson's equation from a density, so the potential represents real mass.

    Each radial profile is stored as knot values plus knot derivatives with
    respect to :math:`\log r`, so evaluation needs no spline solve and the
    coefficients remain ordinary `ParameterField`\ s -- which is what allows a
    caller to supply time-dependent ones. Building an expansion *on* a time
    grid is https://github.com/GalacticDynamics/galax/issues/849

    ``_density`` is reconstructed from the stored :math:`\rho_{lm}`
    profiles, so it is consistent with the expansion itself rather than with
    whatever density the expansion was built from.
    """

    r_knots: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="Radial spline knots, log-spaced."
    )
    phi_lm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="specific energy",
        doc=r"$\Phi_{lm}$ at the knots, shape ``(n_r, n_modes)``.",
    )
    dphi_lm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="specific energy",
        doc=r"$d\Phi_{lm}/d\ln r$ at the knots, shape ``(n_r, n_modes)``.",
    )
    rho_residual_lm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass density",
        doc=r"$\rho_{lm}$ minus the inner power law, shape ``(n_r, n_modes)``.",
    )
    drho_residual_lm: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass density",
        doc="Residual derivative w.r.t. $\\ln r$, shape ``(n_r, n_modes)``.",
    )
    rho_amplitude: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="mass density",
        doc=r"$\rho_{lm}(r_0)$, the inner power-law amplitude, shape ``(n_modes,)``.",
    )
    rho_alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc="Inner power-law exponent, shape ``(n_modes,)``.",
    )

    _: KW_ONLY
    l_max: int = field(static=True)
    lm_keys: tuple[tuple[int, int], ...] = field(static=True)
    symmetry: str | None = field(static=True, default=None)

    def _params(self, t: gt.BBtQorVSz0, /) -> gt.Params:
        """Evaluate the coefficients at ``t``, stripped to this unit system."""
        t = u.Q.from_(t, self.units["time"])
        usys = self.units
        return {
            "r_knots": self.r_knots(t, ustrip=usys["length"]),
            "phi_lm": self.phi_lm(t, ustrip=usys["specific energy"]),
            "dphi_lm": self.dphi_lm(t, ustrip=usys["specific energy"]),
            "rho_residual_lm": self.rho_residual_lm(t, ustrip=usys["mass density"]),
            "drho_residual_lm": self.drho_residual_lm(t, ustrip=usys["mass density"]),
            "rho_amplitude": self.rho_amplitude(t, ustrip=usys["mass density"]),
            "rho_alpha": self.rho_alpha(t, ustrip=usys["dimensionless"]),
        }


def _check_time_independent(pot: AbstractPotential, /) -> None:
    """Raise if any parameter of ``pot`` varies with time."""
    varying = sorted(
        name
        for name, param in pot.parameters.items()
        if not isinstance(param, ConstantParameter)
    )
    if varying:
        msg = (
            f"cannot build a multipole profile from {type(pot).__name__}: its "
            f"time-dependent parameter(s) {varying} cannot be tracked by an "
            "expansion built at a single time. See "
            "https://github.com/GalacticDynamics/galax/issues/849"
        )
        raise ValueError(msg)


def _from_density(
    cls: type["MultipoleProfilePotential"],
    rho: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
    /,
    *,
    r_min: Any,
    r_max: Any,
    n_r: int = 512,
    l_max: int = 8,
    n_theta: int | None = None,
    n_phi: int | None = None,
    symmetry: str | None = None,
    t: Any = None,
    units: Any,
    constants: Any = default_constants,
) -> "MultipoleProfilePotential":
    """Build an expansion of an arbitrary density (see the class docstring).

    The expansion is defined only on ``[r_min, r_max]``. Outside this range,
    cubic Hermite extrapolation in log :math:`r` is used, with no analytic
    tail. Results degrade rapidly outside the bracket and can change sign
    or grow unbounded, and at exactly :math:`r = 0` the gradient is NaN. See
    the class docstring for details.

    Parameters
    ----------
    rho : Callable
        Density function to project. Takes ``(xyz, t)`` as bare arrays in
        this unit system and returns bare *mass density* values in
        ``units["mass density"]`` -- unit-stripped, but not dimensionless:
        the values are a physical density, and are projected and fed into
        the Poisson solve together with :math:`G`. ``xyz`` has shape
        ``(..., 3)`` and the callable must broadcast over the leading axes,
        matching `galax`'s own ``_density`` contract.
    r_min, r_max : Quantity
        Inner and outer radii bracketing the region of interest. These are
        build-time grid configuration and must be concrete: like ``n_r``,
        ``l_max`` and ``symmetry``, they are not traceable under `jax.jit`.
    n_r : int, optional
        Number of radial knots (default 512). Must be >= 4.

        The radial rule is a trapezoid in :math:`r`, which converges as
        :math:`O(h^2)` -- measured ratios of 4.07, 4.03, 4.02, 4.01 and 4.00
        on successive doublings. Over a realistic five-decade bracket the
        relative error in the potential runs

        ===== =======
        n_r   error
        ===== =======
        64    2.0e-2
        128   5.0e-3
        256   1.2e-3
        512   3.1e-4
        1024  7.7e-5
        2048  1.9e-5
        ===== =======

        Build cost is linear in ``n_r`` and paid once; evaluation cost does
        not depend on it. Size the grid from the table for the accuracy you
        need. A better-than-trapezoid rule is
        https://github.com/GalacticDynamics/galax/issues/848
    l_max : int, optional
        Maximum multipole order (default 8).
    n_theta, n_phi : int, optional
        Angular quadrature resolution. Defaults follow `default_angular_resolution`.
    symmetry : str or None, optional
        Symmetry assumption ("spherical", "axisymmetric", "triaxial", or None).
    t : Quantity, optional
        Time at which to evaluate the density. Defaults to 0 Gyr.
    units : AbstractUnitSystem
        Unit system for all inputs and outputs.
    constants : Mapping, optional
        Physical constants (default from `default_constants`).

    Returns
    -------
    MultipoleProfilePotential
        The expansion.

    Raises
    ------
    ValueError
        If ``n_r < 4``, ``l_max < 0``, ``n_theta`` or ``n_phi`` is given and
        is ``< 1``, ``r_min <= 0``, or ``r_min >= r_max``.
    """
    if n_r < 4:
        msg = (
            f"n_r must be >= 4 (got {n_r}); the boundary slopes are fitted over "
            "the innermost and outermost three knots, which are not distinct "
            "windows below four"
        )
        raise ValueError(msg)
    if l_max < 0:
        msg = (
            f"l_max must be >= 0 (got {l_max}); a negative order selects no "
            "modes at all and fails later inside the Legendre recurrence"
        )
        raise ValueError(msg)

    usys = u.unitsystem(units)
    consts = ImmutableMap(constants)

    keys = lm_keys(l_max, symmetry)  # validates `symmetry`
    default_theta, default_phi = default_angular_resolution(l_max)
    n_theta = default_theta if n_theta is None else n_theta
    n_phi = default_phi if n_phi is None else n_phi
    for name, n in (("n_theta", n_theta), ("n_phi", n_phi)):
        if n < 1:
            msg = f"{name} must be >= 1 (got {n}); it is a quadrature node count"
            raise ValueError(msg)

    def to_len(q: Any) -> Float[Array, "..."]:
        return jnp.asarray(  # type: ignore[no-any-return]
            u.ustrip(usys["length"], u.Q.from_(q, usys["length"]))
        )

    # Compare as Python floats, not 0-d arrays. A 0-d array is fine in an
    # `if`, but a *traced* one raises `TracerBoolConversionError`, which would
    # turn a clear "r_min must be < r_max" into a confusing jax error. The
    # grid bracket is build-time configuration and is required to be concrete
    # (as are `n_r`, `l_max` and `symmetry`), so forcing the conversion here
    # reports that requirement at the point it is violated.
    r_min_val = float(to_len(r_min))
    r_max_val = float(to_len(r_max))
    # The grid is log-spaced, so a non-positive bracket feeds `log` a zero or
    # negative and surfaces much later as an opaque JaxRuntimeError from deep
    # inside the jitted build. Reject it here, where the message can say why.
    if r_min_val <= 0.0:
        msg = f"r_min must be > 0 (got {r_min_val}); the radial grid is log-spaced"
        raise ValueError(msg)
    if r_min_val >= r_max_val:
        msg = f"r_min must be < r_max (got r_min={r_min_val}, r_max={r_max_val})"
        raise ValueError(msg)

    r_knots = radial_grid(n_r, jnp.asarray(r_min_val), jnp.asarray(r_max_val))
    t_ = jnp.asarray(
        u.ustrip(
            usys["time"],
            u.Q.from_(u.Q(0.0, "Gyr") if t is None else t, usys["time"]),
        )
    )

    # `build_expansion` takes `rho_fn` as a jit static argument, so jax hashes
    # it. An equinox bound method (e.g. `some_pot._density`) closes over
    # array-valued parameters and is unhashable, raising "Non-hashable static
    # arguments are not supported". Wrap only in that case: a fresh `lambda`
    # per call is a fresh hash, so wrapping unconditionally would force a full
    # recompilation (~1 s) even when the caller passes a plain module-level
    # function repeatedly.
    try:
        hash(rho)
    except TypeError:
        rho_fn: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]] = (
            lambda xyz, tt: rho(xyz, tt)
        )
    else:
        rho_fn = rho

    coeffs = build_expansion(
        rho_fn,
        r_knots,
        l_max,
        keys,
        n_theta,
        n_phi,
        t_,
        jnp.asarray(consts["G"].decompose(usys).value),
    )

    return cls(
        r_knots=u.Q(r_knots, usys["length"]),
        phi_lm=u.Q(coeffs["phi_lm"], usys["specific energy"]),
        dphi_lm=u.Q(coeffs["dphi_lm"], usys["specific energy"]),
        rho_residual_lm=u.Q(coeffs["rho_residual_lm"], usys["mass density"]),
        drho_residual_lm=u.Q(coeffs["drho_residual_lm"], usys["mass density"]),
        rho_amplitude=u.Q(coeffs["rho_amplitude"], usys["mass density"]),
        rho_alpha=u.Q(coeffs["rho_alpha"], ""),
        l_max=l_max,
        lm_keys=keys,
        symmetry=symmetry,
        units=usys,
        constants=consts,
    )


def _from_potential(
    cls: type["MultipoleProfilePotential"],
    pot: AbstractPotential,
    /,
    *,
    r_min: Any,
    r_max: Any,
    n_r: int = 512,
    l_max: int = 8,
    n_theta: int | None = None,
    n_phi: int | None = None,
    symmetry: str | None = None,
    t: Any = None,
) -> "MultipoleProfilePotential":
    """Build an expansion of another potential's density (see class docstring).

    The expansion is defined only on ``[r_min, r_max]``. Outside this range,
    cubic Hermite extrapolation in log :math:`r` is used, with no analytic
    tail. Results degrade rapidly outside the bracket and can change sign
    or grow unbounded, and at exactly :math:`r = 0` the gradient is NaN. See
    the class docstring for details.

    Parameters
    ----------
    pot : AbstractPotential
        Potential whose density to expand.
    r_min, r_max : Quantity
        Inner and outer radii bracketing the region of interest.
    n_r : int, optional
        Number of radial knots (default 512). Must be >= 4; see
        `MultipoleProfilePotential.from_density` for the convergence table.
    l_max : int, optional
        Maximum multipole order (default 8).
    n_theta, n_phi : int, optional
        Angular quadrature resolution. Defaults follow `default_angular_resolution`.
    symmetry : str or None, optional
        Symmetry assumption ("spherical", "axisymmetric", "triaxial", or None).
    t : Quantity, optional
        Time at which to evaluate the density. Defaults to 0 Gyr.

    Returns
    -------
    MultipoleProfilePotential
        The expansion.

    Raises
    ------
    ValueError
        If ``pot`` is time-dependent, ``n_r < 4``, ``l_max < 0``, ``n_theta``
        or ``n_phi`` is given and is ``< 1``, ``r_min <= 0``, or
        ``r_min >= r_max``.
    """
    _check_time_independent(pot)
    # Passed through unwrapped: `_from_density` wraps it if it is unhashable,
    # which a bound `_density` generally is.
    return _from_density(
        cls,
        pot._density,  # noqa: SLF001
        r_min=r_min,
        r_max=r_max,
        n_r=n_r,
        l_max=l_max,
        n_theta=n_theta,
        n_phi=n_phi,
        symmetry=symmetry,
        t=t,
        units=pot.units,
        constants=pot.constants,
    )


@final
class MultipoleProfilePotential(AbstractMultipoleProfilePotential):
    r"""Potential of an arbitrary density via a multipole profile expansion.

    The density is projected onto spherical harmonics on a log-spaced radial
    grid, the radial Poisson equation is solved per mode, and the resulting
    :math:`\Phi_{lm}(r)` are splined. Evaluation, gradients and the
    reconstructed density are all smooth and `jax.grad`-differentiable,
    including on the z-axis.

    Build one with `from_density` or `from_potential` rather than by passing
    coefficients directly.

    **Limitation: unbounded extrapolation outside** ``[r_min, r_max]``.
    The radial splines use cubic Hermite with extrapolation enabled. Outside
    the grid, the edge cubic continues in log :math:`r` with no analytic tail.
    The potential can change sign and the density can grow unbounded as
    :math:`r \to \infty`, degrading rapidly with no warning. Callers must
    choose ``r_min`` and ``r_max`` to bracket the region they will actually
    evaluate, including any region an orbit integrator may explore. A proper
    fix (continuing the analytic :math:`r^{-(l+1)}` / :math:`r^l` tail) is
    tracked separately.

    **Limitation: the gradient is NaN at the origin.** At exactly
    :math:`\vec{x} = 0` the radial direction is undefined, so the harmonics
    are evaluated on a zero vector and :math:`\log r` underflows: the
    potential and density come back finite but meaningless (a monopole
    Hernquist expansion gives :math:`\Phi \approx -2 \times 10^4` against a
    true :math:`-0.45`) and the gradient comes back **NaN**, where an
    analytic potential returns :math:`0`. This is worse than the
    extrapolation above, because a single NaN propagates through an entire
    vmapped batch of orbits, not just the one that reached the origin. Keep
    the origin out of the evaluation set. Tracked with the analytic tail at
    https://github.com/GalacticDynamics/galax/issues/850

    See Also
    --------
    galax.potential.MultipolePotential : the analytic constant-coefficient
        expansion, whose density is zero away from the origin.
    galax.potential.SCFPotential : a basis-function expansion in both radius
        and angle.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    An expansion of a Hernquist sphere reproduces it:

    >>> hern = gp.HernquistPotential(m_tot=u.Q(1e12, "Msun"),
    ...                              r_s=u.Q(10.0, "kpc"), units="galactic")
    >>> pot = gp.MultipoleProfilePotential.from_potential(
    ...     hern, r_min=u.Q(1e-2, "kpc"), r_max=u.Q(1e4, "kpc"),
    ...     n_r=256, l_max=0, symmetry="spherical")
    >>> pot.l_max, pot.symmetry
    (0, 'spherical')
    """

    from_density = classmethod(_from_density)
    from_potential = classmethod(_from_potential)

    def __check_init__(self) -> None:
        """Validate the mode list, the coefficient shapes and the radial grid.

        ``__init__`` is public -- coefficients computed elsewhere can be loaded
        without a rebuild -- so these invariants are not guaranteed by the
        constructors. Without them an inconsistent instance is accepted here
        and fails much later inside ``searchsorted``, ``log`` or a broadcast,
        where the cause is far from the symptom.

        Shapes are static even under tracing, so those checks hold everywhere.
        The radial-grid *value* checks need concrete arrays and are skipped
        when tracing (the shared potential test-suite builds instances inside
        `jax.jit`).
        """
        expected_keys = lm_keys(self.l_max, self.symmetry)
        if self.lm_keys != expected_keys:
            msg = (
                f"lm_keys must match lm_keys(l_max, symmetry). "
                f"Got {self.lm_keys}, expected {expected_keys}."
            )
            raise ValueError(msg)

        t0 = u.Q(0.0, "Gyr")
        r_knots = self.r_knots(t0)
        if r_knots.ndim != 1:
            msg = f"r_knots must be 1-D, got shape {r_knots.shape}."
            raise ValueError(msg)
        n_r, n_modes = r_knots.shape[0], len(self.lm_keys)
        if n_r < 4:
            msg = (
                f"r_knots must have at least 4 entries, got {n_r}; the boundary "
                "slopes are fitted over the innermost and outermost three."
            )
            raise ValueError(msg)

        for name, expected in (
            ("phi_lm", (n_r, n_modes)),
            ("dphi_lm", (n_r, n_modes)),
            ("rho_residual_lm", (n_r, n_modes)),
            ("drho_residual_lm", (n_r, n_modes)),
            ("rho_amplitude", (n_modes,)),
            ("rho_alpha", (n_modes,)),
        ):
            got = getattr(self, name)(t0).shape
            if got != expected:
                msg = (
                    f"{name} must have shape {expected} for a grid of {n_r} "
                    f"knots and {n_modes} modes, got {got}."
                )
                raise ValueError(msg)

        if isinstance(r_knots.value, jax.core.Tracer):
            return
        rv = r_knots.value
        if not bool(jnp.all(rv > 0)):
            msg = "r_knots must be strictly positive; the grid is log-spaced."
            raise ValueError(msg)
        if not bool(jnp.all(jnp.diff(rv) > 0)):
            msg = "r_knots must be strictly increasing."
            raise ValueError(msg)
