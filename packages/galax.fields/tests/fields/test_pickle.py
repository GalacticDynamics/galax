"""Pickling of `galax.fields` objects."""

import pickle

import numpy as np

import unxt as u

import galax.fields as gf
import galax.potential as gp


def test_hamiltonian_field_roundtrip() -> None:
    """A `HamiltonianField` survives a pickle round trip."""
    pot = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    field = gf.HamiltonianField(pot)

    out = pickle.loads(pickle.dumps(field))

    assert type(out) is gf.HamiltonianField

    # The potential survives too: both fields give the same output.
    t = u.Q(0, "Gyr")
    x = u.Q([8.0, 0, 0], "kpc")
    v = u.Q([0, 220, 0], "km/s")
    for expected, got in zip(field(t, x, v), out(t, x, v), strict=True):
        np.testing.assert_allclose(got, expected)
