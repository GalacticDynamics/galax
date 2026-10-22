"""Result type for action-angle-frequency estimation."""

__all__ = ["ActionsAnglesFrequencies"]

from jaxtyping import Array

import equinox as eqx

import unxt as u


class ActionsAnglesFrequencies(eqx.Module):
    """Actions, angles and frequencies, each of shape ``(..., 3)``.

    The last axis is ordered (R, phi, z): actions ``(J_R, J_phi, J_z)``,
    angles ``(theta_R, theta_phi, theta_z)`` and frequencies
    ``(Omega_R, Omega_phi, Omega_z)``.
    """

    actions: Array | u.AbstractQuantity
    angles: Array | u.AbstractQuantity
    freqs: Array | u.AbstractQuantity
