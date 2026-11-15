"""galax.dynamics re-exports the objects that live in galax.orbits."""

import galax.dynamics as gd
import galax.dynamics._src.utils as gdu
import galax.orbits as go
import galax.orbits.utils as gou


def test_fields_identity():
    assert gd.AbstractOrbitField is go.AbstractOrbitField
    assert gd.HamiltonianField is go.HamiltonianField
    assert gd.fields.HamiltonianField is go.HamiltonianField
    assert gdu.parse_to_t_y is gou.parse_to_t_y
