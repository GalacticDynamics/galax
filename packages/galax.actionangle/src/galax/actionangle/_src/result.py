"""Result type for action-angle-frequency estimation."""

__all__ = ["ActionsAnglesFrequencies"]

from jaxtyping import Array

import equinox as eqx

import unxt as u


class ActionsAnglesFrequencies(eqx.Module):
    """Actions, angles and frequencies, each of shape ``(..., 3)``.

    The last axis is in the estimator's order: (R, φ, z) for axisymmetric
    estimators and O2GF loop orbits, (x, y, z) for O2GF box orbits. In the
    (R, φ, z) order the actions are ``(J_R, J_phi, J_z)``, the angles
    ``(theta_R, theta_phi, theta_z)`` and the frequencies
    ``(Omega_R, Omega_phi, Omega_z)``.
    """

    actions: Array | u.AbstractQuantity
    angles: Array | u.AbstractQuantity
    freqs: Array | u.AbstractQuantity
