"""galax.dynamics re-exports the objects that live in galax.nbody."""

import galax.dynamics as gd
import galax.nbody as gn


def test_nbody_field_identity():
    assert gd.NBodyField is gn.NBodyField
    assert gd.fields.NBodyField is gn.NBodyField
