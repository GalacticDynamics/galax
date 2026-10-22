"""Test helpers: separable and scale-safe Kuzmin-Kutuzov potentials, KK Delta."""

__all__ = ["ScaledKuzminKutuzovPotential", "SeparablePotential", "kk_delta"]

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential as gp
from galax.potential.params import AbstractParameter, ParameterField


class ScaledKuzminKutuzovPotential(gp.AbstractSinglePotential):
    r"""Kuzmin-Kutuzov, Delta^2 = a^2 - c^2, evaluated with lengths in units of c.

    `galax.potential.KuzminKutuzovPotential` squares the input coordinates,
    which overflows float32 in SI units (R ~ 1e20 m); this one does not, to
    test that `staeckel_fudge_delta` itself is scale-free there.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @jax.jit
    def _potential(self, xyz, t, /):
        t = u.Q.from_(t, self.units["time"])
        G = self.constants["G"].value
        m = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=self.units["length"])
        c = self.c(t, ustrip=self.units["length"])
        xyz = u.ustrip(AllowValue, self.units["length"], xyz) / c
        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z2 = xyz[..., 2] ** 2
        a2 = (a / c) ** 2
        lam_nu = R2 + a2 * z2 + a2
        denom = jnp.sqrt(R2 + z2 + a2 + 1 + 2 * jnp.sqrt(lam_nu))
        return -G * m / c / denom


class SeparablePotential(gp.AbstractSinglePotential):
    r"""$\Phi = f(R) + g(z)$, so $\partial^2\Phi / \partial R \partial z = 0$.

    $f = \frac{1}{2} R^2$, $g = z^2$, in the potential's units.
    """

    @jax.jit
    def _potential(self, xyz, t, /):
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        return 0.5 * (xyz[..., 0] ** 2 + xyz[..., 1] ** 2) + xyz[..., 2] ** 2


def kk_delta(pot: gp.KuzminKutuzovPotential, t: float = 0.0) -> float:
    """Focal distance Delta = sqrt(a^2 - c^2) at time ``t``, in ``pot.units``."""
    t = u.Q(t, pot.units["time"])
    a = pot.a(t, ustrip=pot.units["length"])
    c = pot.c(t, ustrip=pot.units["length"])
    return float(jnp.sqrt(a**2 - c**2))
