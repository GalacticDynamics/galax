"""galax.dynamics re-exports the objects that live in galax.nbody."""

import galax.dynamics as gd
import galax.nbody as gn


def test_nbody_field_identity():
    assert gd.NBodyField is gn.NBodyField
    assert gd.fields.NBodyField is gn.NBodyField


def test_dynamics_nbody_module_reexports_same_objects():
    for name in gn.__all__:
        assert getattr(gd.nbody, name) is getattr(gn, name)


def test_nbody_all_lists_match():
    assert set(gd.nbody.__all__) == set(gn.__all__)
