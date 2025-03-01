"""Tests for `galax.potential._src.frame` package."""

from dataclasses import replace

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.potential as gp


def test_bar_means_of_rotation() -> None:
    """Test the equivalence of hard-coded vs operator means of rotation."""
    basepot = gp.LongMuraliBarPotential(
        m_tot=1e9, a=5.0, b=0.1, c=0.1, alpha=0, units="galactic"
    )

    Omega_z_angv = u.Quantity(220.0, "deg/Myr")

    def alpha_func(t: u.Quantity) -> u.Quantity["deg"]:
        return Omega_z_angv * t

    # Hard-coded means of rotation
    hardpot = replace(basepot, alpha=alpha_func)

    # Operator means of rotation
    op = gc.ops.ConstantRotationZOperator(Omega_z=Omega_z_angv)
    xpot = gp.TransformedPotential(basepot, op)

    # quick test of the op
    q = u.Quantity([5.0, 0.0, 0.0], "kpc")
    t = u.Quantity(0.0, "Myr")

    newq, newt = op.inverse(q, t)
    assert isinstance(newq, u.Quantity)
    assert isinstance(newt, u.Quantity)

    # They should be equivalent at t=0
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=110 Myr (1/2 period)
    t = u.Quantity(110, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=220 Myr (1 period)
    t = u.Quantity(220, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=55 Myr (1/4 period)
    t = u.Quantity(55, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # TODO: move this test to a more appropriate location
    # Test that the frame's constants are the same as the base potential's
    assert xpot.constants is basepot.constants
