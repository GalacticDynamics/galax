"""Tests for `galax.potential._src.frame` package."""

from dataclasses import replace

import numpy as np
from unxts.parametric import ParametricQuantity

import unxt as u

import galax.coordinates as gc
import galax.potential as gp


def assert_same(a: u.Quantity, b: u.Quantity) -> None:
    """Assert equal to rounding: the two routes round in different orders.

    Rounding relative to the vectors' size (the larger of the two, so the
    comparison is symmetric), so a component that is 0 in one route and a
    rounding residue in the other still compares equal.
    """
    a, b = np.asarray(a.ustrip(b.unit)), np.asarray(b.ustrip(b.unit))
    scale = max(np.abs(a).max(), np.abs(b).max())
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-15 * scale)


def test_bar_means_of_rotation() -> None:
    """Test the equivalence of hard-coded vs operator means of rotation."""
    basepot = gp.LongMuraliBarPotential(
        m_tot=1e9, a=5.0, b=0.1, c=0.1, alpha=0, units="galactic"
    )

    Omega_z_angv = u.Q(220.0, "deg/Myr")

    def alpha_func(t: u.Quantity) -> ParametricQuantity["deg"]:
        return Omega_z_angv * t

    # Hard-coded means of rotation
    hardpot = replace(basepot, alpha=alpha_func)

    # Operator means of rotation
    op = gc.ops.ConstantRotationZOperator(Omega_z=Omega_z_angv)
    xpot = gp.TransformedPotential(basepot, op)

    # quick test of the op
    q = u.Q([5.0, 0.0, 0.0], "kpc")
    t = u.Q(0.0, "Myr")

    newq, newt = op.inverse(t, q)
    assert isinstance(newq, u.Quantity)
    assert isinstance(newt, u.Quantity)

    # They should be equivalent at t=0
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert_same(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=110 Myr (1/2 period)
    t = u.Q(110, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert_same(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=220 Myr (1 period)
    t = u.Q(220, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert_same(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # They should be equivalent at t=55 Myr (1/4 period)
    t = u.Q(55, "Myr")
    assert xpot.potential(q, t) == hardpot.potential(q, t)
    assert_same(xpot.acceleration(q, t), hardpot.acceleration(q, t))

    # TODO: move this test to a more appropriate location
    # Test that the frame's constants are the same as the base potential's
    assert xpot.constants is basepot.constants
