"""Tests for `galax.potential._src.frame` package."""

from dataclasses import replace

from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates.operators as gco
import galax.potential as gp


def test_bar_means_of_rotation() -> None:
    """Test the equivalence of hard-coded vs operator means of rotation."""
    base_pot = gp.BarPotential(
        m_tot=u.Quantity(1e9, "Msun"),
        a=u.Quantity(5.0, "kpc"),
        b=u.Quantity(0.1, "kpc"),
        c=u.Quantity(0.1, "kpc"),
        Omega=u.Quantity(0.0, "Hz"),
        units="galactic",
    )

    Omega_z_freq = u.Quantity(220.0, "1/Myr")
    Omega_z_angv = jnp.multiply(Omega_z_freq, u.Quantity(1.0, "rad"))

    # Hard-coded means of rotation
    hardpot = replace(base_pot, Omega=Omega_z_freq)

    # Operator means of rotation
    op = gco.ConstantRotationZOperator(Omega_z=Omega_z_angv)
    framedpot = gp.PotentialFrame(base_pot, op)

    # quick test of the op
    q = u.Quantity([5.0, 0.0, 0.0], "kpc")
    t = u.Quantity(0.0, "Myr")

    newq, newt = op.inverse(q, t)
    assert isinstance(newq, u.Quantity)
    assert isinstance(newt, u.Quantity)

    # They should be equivalent at t=0
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), u.Quantity),
        convert(hardpot.acceleration(q, t), u.Quantity),
    )

    # They should be equivalent at t=110 Myr (1/2 period)
    t = u.Quantity(110, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), u.Quantity),
        convert(hardpot.acceleration(q, t), u.Quantity),
    )

    # They should be equivalent at t=220 Myr (1 period)
    t = u.Quantity(220, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), u.Quantity),
        convert(hardpot.acceleration(q, t), u.Quantity),
    )

    # They should be equivalent at t=55 Myr (1/4 period)
    t = u.Quantity(55, "Myr")
    assert framedpot.potential(q, t) == hardpot.potential(q, t)
    assert jnp.array_equal(
        convert(framedpot.acceleration(q, t), u.Quantity),
        convert(hardpot.acceleration(q, t), u.Quantity),
    )

    # TODO: move this test to a more appropriate location
    # Test that the frame's constants are the same as the base potential's
    assert framedpot.constants is base_pot.constants
