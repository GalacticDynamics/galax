"""`galax.dynamics` re-exports the objects `galax.fields` owns."""

import pytest

import galax.dynamics as gd
import galax.fields as gf

NAMES = (
    "AbstractField",
    "AbstractOrbitField",
    "AbstractSolver",
    "HamiltonianField",
    "NBodyField",
    "SolveState",
    "integrate_field",
    "parse_time_specification",
)


@pytest.mark.parametrize("name", NAMES)
def test_same_objects(name):
    assert getattr(gd, name) is getattr(gf, name)


@pytest.mark.parametrize(
    "name", ["AbstractField", "AbstractOrbitField", "HamiltonianField", "NBodyField"]
)
def test_fields_module_same_objects(name):
    assert getattr(gd.fields, name) is getattr(gf, name)


def test_submodule_reexports():
    assert gd.fields.AbstractField is gf.AbstractField
    assert gd.orbit.AbstractSolver is gf.AbstractSolver
    assert gd.orbit.parse_time_specification is gf.parse_time_specification
    assert gd.integrate.parse_time_specification is gf.parse_time_specification
