"""Staeckel fudge (Binney 2012; Sanders 2012), ported from gala.

The focal distance follows gala's ``get_staeckel_fudge_delta``
(https://github.com/adrn/gala/pull/649).
"""

__all__: list[str] = []

import functools as ft

from jaxtyping import Array
from typing import Any

import jax
import jax.numpy as jnp
from plum import dispatch

import unxt as u

import galax.coordinates as gc
from ._parse import parse_qpt, parse_w
from galax.potential import AbstractPotential, Symmetry


def _delta_single(pot: Any, xyz: Array, t: Array) -> Array:
    """Focal distance at one point; NaN where Delta^2 < 0 or d2Phi/dRdz == 0.

    ``d2Phi/dRdz == 0`` covers separable potentials and, for a potential
    smooth and symmetric about the plane, z = 0. R = 0 is NaN on its own (0/0).
    As in gala, the cylindrical derivatives are taken at (x, y, z) in the local
    (R, z) basis, from one ``jax.linearize`` of the gradient. The output is
    ``stop_gradient``-ed and the final ``where`` sets the NaNs, so the x/0, 0/0
    and sqrt(<0) intermediates need no guards.
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    R = jnp.hypot(x, y)
    e_R = jnp.stack([x / R, y / R, jnp.zeros_like(z)])
    e_z = jnp.zeros_like(xyz).at[2].set(1)
    grad, hvp = jax.linearize(lambda q: pot._gradient(q, t), xyz)  # noqa: SLF001
    H_eR, H_ez = hvp(e_R), hvp(e_z)
    dPhi_dR = grad[0] * e_R[0] + grad[1] * e_R[1]
    dPhi_dz = grad[2]
    d2Phi_dR2 = H_eR[0] * e_R[0] + H_eR[1] * e_R[1]
    d2Phi_dz2 = H_ez[2]
    d2Phi_dRdz = H_eR[2]

    # Sanders (2012) eq. 9, with the sign convention such that
    # Delta^2 = a^2 - c^2 > 0 for the Kuzmin-Kutuzov potential (as in galpy;
    # Sanders' printed sign differs). Evaluated as Delta^2 / r^2, from R / r,
    # z / r and dPhi / r: every term is then O(1) or of the Hessian's size,
    # where R^2 itself overflows float32 in SI units (R ~ 1e20 m).
    r = jnp.hypot(R, z)
    R_, z_ = R / r, z / r
    gR, gz = dPhi_dR / r, dPhi_dz / r
    num = 3 * z_ * gR - 3 * R_ * gz + R_ * z_ * (d2Phi_dR2 - d2Phi_dz2)
    degenerate = (R == 0) | (d2Phi_dRdz == 0)  # 0/0 or x/0
    a2_c2 = z_**2 - R_**2 + num / d2Phi_dRdz

    # Zero |Delta^2| < 1e3 eps S (gala's |Delta^2| < 1e-12, made unit-free),
    # with S the size of the terms of eq. 9 that cancel, so the relative error
    # is ~ eps S / (2 Delta^2). S >= r^2 for a spherical Phi with Phi' > 0, so
    # R^2 + z^2 is omitted (all of it in units of r^2, like `a2_c2`). Spherical
    # round-off is <~ 1.5 eps S for every spherical built-in from 1e-8 to 1e8
    # scale lengths, so 1e3 keeps a wide margin; the tests pin the factor to
    # (316, ~2400], the upper side set by the float32 Kuzmin-Kutuzov far
    # points.
    scale = (
        jnp.abs(3 * z_ * gR)
        + jnp.abs(3 * R_ * gz)
        + jnp.abs(R_ * z_) * (jnp.abs(d2Phi_dR2) + jnp.abs(d2Phi_dz2))
    ) / jnp.abs(d2Phi_dRdz)
    tol = 1e3 * jnp.finfo(a2_c2.dtype).eps  # type: ignore[no-untyped-call]
    a2_c2 = jnp.where(jnp.abs(a2_c2) < tol * scale, 0.0, a2_c2)

    # NaN rather than a wrong value where the potential's derivatives are not
    # to be trusted. A potential evaluated near underflow can have subnormal
    # autodiff intermediates behind outputs that look normal (one form of
    # Kuzmin-Kutuzov at m = 1e-290 Msun gave a gradient 1% off, and Delta 16
    # for 2.8). What gives it away is that d2Phi/dRdz, taken along e_R then z
    # and along e_z then R, then disagree, by ~1e14 eps of the Hessian there,
    # against <~ 5 eps for every built-in (float32 and float64): cut at
    # sqrt(eps). A subnormal d2Phi/dRdz has lost its relative precision outright.
    finfo = jnp.finfo(a2_c2.dtype)  # type: ignore[no-untyped-call]
    d2Phi_dzdR = H_ez[0] * e_R[0] + H_ez[1] * e_R[1]
    h = jnp.max(jnp.abs(jnp.stack([gR, gz, d2Phi_dR2, d2Phi_dz2, d2Phi_dRdz])))
    asymmetric = jnp.abs(d2Phi_dRdz - d2Phi_dzdR) > jnp.sqrt(finfo.eps) * h
    unreliable = asymmetric | (jnp.abs(d2Phi_dRdz) < finfo.tiny)
    bad = degenerate | (a2_c2 < 0) | unreliable
    return jax.lax.stop_gradient(jnp.where(bad, jnp.nan, r * jnp.sqrt(a2_c2)))


@ft.partial(jax.jit, inline=True)
def _delta(pot: Any, q: Array, t: Array) -> Array:
    """Batched ``delta`` on bare arrays in ``pot.units``; ``q`` (..., 3), ``t`` (...).

    The result is a length in ``pot.units["length"]``. A potential declaring
    `Symmetry.SPHERICAL` is already separable in spherical coordinates, the
    Delta = 0 limit of the spheroidal ones, so it gets exactly 0 everywhere --
    plane, axis and origin included -- without the Hessian, whose round-off
    eq. 9 would otherwise amplify.
    """
    if Symmetry(pot.symmetry) == Symmetry.SPHERICAL:  # static: no retrace
        shape = jnp.broadcast_shapes(q.shape[:-1], t.shape)
        return jnp.zeros(shape, dtype=jnp.result_type(q, t))
    f = ft.partial(_delta_single, pot)
    for _ in t.shape:
        f = jax.vmap(f)
    return f(q, t)


@dispatch
def staeckel_fudge_delta(pot: AbstractPotential, q: Any, t: Any, /) -> Any:
    """Focal distance from positions and time."""
    q, _, t, as_quantity = parse_qpt(pot, q, None, t)
    delta = _delta(pot, q, t)
    return u.Q(delta, pot.units["length"]) if as_quantity else delta


@dispatch
def staeckel_fudge_delta(
    pot: AbstractPotential, w: gc.AbstractPhaseSpaceCoordinate, /
) -> Any:
    """Focal distance of a phase-space coordinate or orbit."""
    q, _, t = parse_w(pot, w)
    return u.Q(_delta(pot, q, t), pot.units["length"])
