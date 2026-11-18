"""`galax.dynamics` re-exports the objects `galax.fields` owns."""

import pytest

import galax.dynamics as gd
import galax.dynamics._src.orbit as gdo
import galax.fields as gf

NAMES = (
    "AbstractField",
    "AbstractSolver",
    "SolveState",
    "integrate_field",
    "parse_time_specification",
)

ORBIT_FIELDS = ("AbstractOrbitField", "HamiltonianField", "NBodyField")


@pytest.mark.parametrize("name", NAMES)
def test_same_objects(name):
    assert getattr(gd, name) is getattr(gf, name)


def test_fields_module_abstract_field():
    assert gd.fields.AbstractField is gf.AbstractField


@pytest.mark.parametrize("name", ORBIT_FIELDS)
def test_orbit_fields_reexported_by_dynamics(name):
    obj = getattr(gdo, name)
    assert getattr(gd, name) is obj
    assert getattr(gd.fields, name) is obj
    assert not hasattr(gf, name)


def test_submodule_reexports():
    assert gd.orbit.AbstractSolver is gf.AbstractSolver
    assert gd.orbit.parse_time_specification is gf.parse_time_specification
    assert gd.integrate.parse_time_specification is gf.parse_time_specification
