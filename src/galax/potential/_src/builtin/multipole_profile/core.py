"""Multipole profile potentials."""

__all__ = ["AbstractMultipoleProfilePotential", "MultipoleProfilePotential"]

from dataclasses import KW_ONLY

from collections.abc import Callable
from jaxtyping import Array, Float
from typing import Any, final

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

    Subclasses differ only in where ``_density`` comes from: this one
    reconstructs it from the stored :math:`\rho_{lm}` profiles, while a
    subclass with a closed-form density overrides it with the exact
    expression.
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
    # Restated, matching `AbstractSinglePotential.units` exactly (converter
    # and staticness included), purely so equinox's abstract-var resolution
    # sees a concrete `units` here. `MultipoleProfileMixin` and
    # `AbstractSinglePotential` both declare `units` -- the mixin as
    # `eqx.AbstractVar`, the base class concretely -- and because the mixin
    # is listed first in the MRO (required so its `_potential`/`_density`
    # win over `AbstractPotential`'s abstract stubs), equinox's abstract-var
    # scan (which walks the MRO in reverse) processes the mixin's abstract
    # redeclaration *after* the base class's concrete one, leaving `units`
    # abstract unless it is restated here. A bare re-annotation is not
    # enough: it clears the abstractness but silently drops the converter,
    # so `units` must be given in full.
    units: u.AbstractUnitSystem = field(converter=u.unitsystem, static=True)
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
    or grow unbounded. See the class docstring for details.

    Parameters
    ----------
    rho : Callable
        Density function to project, taking ``(xyz, t)`` in the unit system
        and returning dimensionless density values.
    r_min, r_max : Quantity
        Inner and outer radii bracketing the region of interest.
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
        If ``n_r < 4`` or ``r_min >= r_max``.
    """
    if n_r < 4:
        msg = (
            f"n_r must be >= 4 (got {n_r}); fewer knots give unbounded spline behavior"
        )
        raise ValueError(msg)

    usys = u.unitsystem(units)
    consts = ImmutableMap(constants)

    keys = lm_keys(l_max, symmetry)  # validates `symmetry`
    default_theta, default_phi = default_angular_resolution(l_max)
    n_theta = default_theta if n_theta is None else n_theta
    n_phi = default_phi if n_phi is None else n_phi

    def to_len(q: Any) -> Float[Array, "..."]:
        return jnp.asarray(  # type: ignore[no-any-return]
            u.ustrip(usys["length"], u.Q.from_(q, usys["length"]))
        )

    r_min_val = to_len(r_min)
    r_max_val = to_len(r_max)
    if r_min_val >= r_max_val:
        msg = f"r_min must be < r_max (got r_min={r_min_val}, r_max={r_max_val})"
        raise ValueError(msg)

    r_knots = radial_grid(n_r, r_min_val, r_max_val)
    t_ = jnp.asarray(
        u.ustrip(
            usys["time"],
            u.Q.from_(u.Q(0.0, "Gyr") if t is None else t, usys["time"]),
        )
    )

    # `build_expansion` takes `rho_fn` as a jit static argument, so jax hashes
    # it. An equinox bound method (e.g. `some_pot._density`) closes over
    # array-valued parameters and is unhashable, raising "Non-hashable static
    # arguments are not supported". Wrapping makes any callable acceptable.
    coeffs = build_expansion(
        lambda xyz, tt: rho(xyz, tt),
        r_knots,
        l_max,
        keys,
        n_theta,
        n_phi,
        t_,
        jnp.asarray(consts["G"].decompose(usys).value),
    )

    return cls(  # type: ignore[call-arg]
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
    or grow unbounded. See the class docstring for details.

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
        Time at which to evaluate the density. Defaults to the potential's
        default time.

    Returns
    -------
    MultipoleProfilePotential
        The expansion.

    Raises
    ------
    ValueError
        If ``pot`` is time-dependent, ``n_r < 4``, or ``r_min >= r_max``.
    """
    _check_time_independent(pot)
    return _from_density(
        cls,
        lambda xyz, tt: pot._density(xyz, tt),  # noqa: SLF001
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
        """Validate that lm_keys matches the symmetry."""
        expected_keys = lm_keys(self.l_max, self.symmetry)
        if self.lm_keys != expected_keys:
            msg = (
                f"lm_keys must match lm_keys(l_max, symmetry). "
                f"Got {self.lm_keys}, expected {expected_keys}."
            )
            raise ValueError(msg)
