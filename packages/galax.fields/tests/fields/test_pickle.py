"""Pickling of `galax.fields` objects."""

import pickle

import unxt as u

import galax.fields as gf
import galax.potential as gp


def test_hamiltonian_field_roundtrip() -> None:
    """A `HamiltonianField` survives a pickle round trip."""
    pot = gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")
    field = gf.HamiltonianField(pot)

    out = pickle.loads(pickle.dumps(field))

    assert type(out) is gf.HamiltonianField
