"""Multipole profile potentials."""

__all__ = ["AbstractMultipoleProfilePotential", "MultipoleProfilePotential"]

import math
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
from .build import build_expansion
from .expansion import MultipoleProfileMixin
from galax.potential._src.base import AbstractPotential, default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.harmonic import (
    default_angular_resolution,
    lm_keys,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.constant import ConstantParameter
from galax.potential._src.params.field import ParameterField
from galax.potential._src.symmetry import Symmetry


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
    phi_asympt_powers: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="dimensionless",
        doc=(
            r"Tail exponents $(v, s)$ for the inner and outer continuation, "
            r"shape ``(2, 2, n_modes)``."
        ),
    )
    phi_asympt_scales: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="specific energy",
        doc=(
            r"Tail amplitude $B$ for the inner and outer continuation, "
            r"shape ``(2, 1, n_modes)``."
        ),
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
    symmetry: Symmetry = field(static=True, converter=Symmetry, default=Symmetry.NONE)
    """Symmetry assumption; `None` is accepted as an alias for `Symmetry.NONE`."""

    @property
    def lm_keys(self) -> tuple[tuple[int, int], ...]:
        """The (l, m) modes, determined by ``l_max`` and ``symmetry``.

        Recomputed rather than stored: it is a pure function of two static
        fields, so a stored copy could only ever disagree with them. The
        result is a plain tuple of int pairs, which hashes and compares by
        value, so passing it as a `jax.jit` static argument on the
        evaluation path does not defeat the compilation cache.
        """
        return lm_keys(self.l_max, self.symmetry)  # validates `symmetry`

    def _params(self, t: gt.BBtQorVSz0, /) -> gt.Params:
        """Evaluate the coefficients at ``t``, stripped to this unit system."""
        t = u.Q.from_(t, self.units["time"])
        usys = self.units
        return {
            "r_knots": self.r_knots(t, ustrip=usys["length"]),
            "phi_lm": self.phi_lm(t, ustrip=usys["specific energy"]),
            "dphi_lm": self.dphi_lm(t, ustrip=usys["specific energy"]),
            "phi_asympt_powers": self.phi_asympt_powers(
                t, ustrip=usys["dimensionless"]
            ),
            "phi_asympt_scales": self.phi_asympt_scales(
                t, ustrip=usys["specific energy"]
            ),
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


def _validate_bracket(r_min: float, r_max: float, /) -> None:
    """Reject a radial bracket the log-spaced grid cannot represent.

    Each failure is caught here, where the message can name the argument,
    rather than later as an opaque `JaxRuntimeError` from inside the jitted
    build.
    """
    # Every comparison below is False against a `nan`, so a non-finite bracket
    # would pass both remaining guards and reach `geomspace`, which returns an
    # all-`nan` grid -- exactly the opaque failure they exist to stop.
    if not (math.isfinite(r_min) and math.isfinite(r_max)):
        msg = f"r_min and r_max must be finite (got r_min={r_min}, r_max={r_max})"
        raise ValueError(msg)
    if r_min <= 0.0:
        msg = f"r_min must be > 0 (got {r_min}); the radial grid is log-spaced"
        raise ValueError(msg)
    if r_min >= r_max:
        msg = f"r_min must be < r_max (got r_min={r_min}, r_max={r_max})"
        raise ValueError(msg)


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

    Outside ``[r_min, r_max]`` the potential continues as the power law
    fitted per mode by `galax.potential.harmonic.asymptotic_coeffs`, joined
    to the spline in value *and* slope. It stays bounded and keeps its sign
    at every radius: for a monopole Hernquist on ``[1e-2, 300] kpc`` the
    error is 0.2% at :math:`1.2 r_\max` and 2.6% at :math:`10 r_\max`, where
    the bare edge cubic was 2.5% and then wrong by a factor of 16 -- with the
    wrong sign past :math:`1.9 r_\max`. Inside the grid the result is
    bit-identical to the spline alone, so the continuation costs interior
    accuracy nothing. Accuracy outside is still capped by the spline's
    natural end condition, tracked at
    https://github.com/GalacticDynamics/galax/issues/858

    **Limitation: the density is not continued.** Only :math:`\Phi` gets the
    asymptotic tail. ``_density`` outside the grid is still the edge cubic on
    the :math:`\rho_{lm}` residual and can grow unbounded as
    :math:`r \to \infty`. Choose ``r_min``/``r_max`` to bracket the region
    where the density will be evaluated.

    The origin is covered by the same continuation: the inner tail is
    evaluated on a clamped :math:`\log r`, so :math:`\Phi(0)` is finite (a
    monopole Hernquist expansion gives :math:`-0.4530` against a true
    :math:`-0.4499`) and :math:`\nabla\Phi(0)` is exactly zero, which is the
    analytic answer. The *density* has no such tail and is still meaningless
    at the origin.

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
    (0, <Symmetry.SPHERICAL: 'spherical'>)
    """

    @classmethod
    def from_density(
        cls,
        rho: Callable[[gt.BtSz3, gt.BBtSz0], Float[Array, "..."]],
        /,
        *,
        r_min: Any,
        r_max: Any,
        n_r: int = 512,
        l_max: int = 8,
        n_theta: int | None = None,
        n_phi: int | None = None,
        symmetry: Symmetry | str | None = None,
        t: Any = None,
        units: Any,
        constants: Any = default_constants,
    ) -> "MultipoleProfilePotential":
        """Build an expansion of an arbitrary density (see the class docstring).

        The expansion is fitted on ``[r_min, r_max]``; outside it the
        potential continues as a fitted power law per mode, but the *density*
        is still an unguarded extrapolation. See the class docstring for
        details.

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
        symmetry : Symmetry or str or None, optional
            Symmetry assumption; see `Symmetry`. `None` is an alias for
            `Symmetry.NONE`, which keeps every mode.
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
        _validate_bracket(r_min_val, r_max_val)

        r_knots = jnp.geomspace(r_min_val, r_max_val, n_r)
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
            phi_asympt_powers=u.Q(coeffs["phi_asympt_powers"], ""),
            phi_asympt_scales=u.Q(coeffs["phi_asympt_scales"], usys["specific energy"]),
            rho_residual_lm=u.Q(coeffs["rho_residual_lm"], usys["mass density"]),
            drho_residual_lm=u.Q(coeffs["drho_residual_lm"], usys["mass density"]),
            rho_amplitude=u.Q(coeffs["rho_amplitude"], usys["mass density"]),
            rho_alpha=u.Q(coeffs["rho_alpha"], ""),
            l_max=l_max,
            symmetry=symmetry,
            units=usys,
            constants=consts,
        )

    @classmethod
    def from_potential(
        cls,
        pot: AbstractPotential,
        /,
        *,
        r_min: Any,
        r_max: Any,
        n_r: int = 512,
        l_max: int = 8,
        n_theta: int | None = None,
        n_phi: int | None = None,
        symmetry: Symmetry | str | None = None,
        t: Any = None,
    ) -> "MultipoleProfilePotential":
        """Build an expansion of another potential's density (see class docstring).

        The expansion is fitted on ``[r_min, r_max]``; outside it the
        potential continues as a fitted power law per mode, but the *density*
        is still an unguarded extrapolation. See the class docstring for
        details.

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
        symmetry : Symmetry or str or None, optional
            Symmetry assumption; see `Symmetry`. `None` is an alias for
            `Symmetry.NONE`, which keeps every mode.
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
        # Passed through unwrapped: `from_density` wraps it if it is unhashable,
        # which a bound `_density` generally is.
        return cls.from_density(
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

    def __check_init__(self) -> None:
        """Validate the coefficient shapes and the radial grid.

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
            ("phi_asympt_powers", (2, 2, n_modes)),
            ("phi_asympt_scales", (2, 1, n_modes)),
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
