"""Multipole profile potentials."""

__all__ = ["AbstractMultipoleProfilePotential", "MultipoleProfilePotential"]

from dataclasses import KW_ONLY

from typing import final

from equinox import field

import unxt as u

import galax.potential.custom_types as gt
from .expansion import MultipoleProfileMixin
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.params.base import AbstractParameter
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

    See Also
    --------
    galax.potential.MultipolePotential : the analytic constant-coefficient
        expansion, whose density is zero away from the origin.
    galax.potential.SCFPotential : a basis-function expansion in both radius
        and angle.
    """
