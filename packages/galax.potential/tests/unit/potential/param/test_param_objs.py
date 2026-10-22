"""Test the ``param_objs`` attribute of potentials."""

import jax.numpy as jnp

import unxt as u

import galax.potential as gp
import galax.potential.params as gpp


def _kepler() -> gp.KeplerPotential:
    return gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic")


def test_class_param_objs_maps_to_fields() -> None:
    """On the class, ``param_objs`` maps names to `ParameterField`."""
    assert set(gp.KeplerPotential.param_objs) == {"m_tot"}
    assert isinstance(gp.KeplerPotential.param_objs["m_tot"], gpp.ParameterField)


def test_instance_param_objs_maps_to_parameters() -> None:
    """On an instance, ``param_objs`` maps names to the parameter objects."""
    pot = _kepler()
    assert pot.param_objs["m_tot"] is pot.m_tot


def test_parameters_attribute_removed() -> None:
    """The old name is gone, with no alias."""
    assert not hasattr(_kepler(), "parameters")
    assert not hasattr(gp.KeplerPotential, "parameters")


def test_subclass_gets_param_objs() -> None:
    """A subclass declared after import gets its ``param_objs`` populated."""

    class MassPotential(gp.AbstractSinglePotential):
        mass: gpp.AbstractParameter = gpp.ParameterField(dimensions="mass")  # type: ignore[assignment]

        def _potential(self, q, t):
            return jnp.zeros(())

    assert set(MassPotential.param_objs) == {"mass"}


def test_composite_param_objs_nested() -> None:
    """A composite's ``param_objs`` is ``{name: sub.param_objs}``."""
    pot = _kepler()
    comp = gp.CompositePotential(kepler=pot)
    assert list(comp.param_objs) == ["kepler"]
    assert comp.param_objs["kepler"] == pot.param_objs


def test_descriptor_class_names() -> None:
    """The descriptor classes carry the ``ParamObjs`` names only."""
    for name in (
        "AbstractParamObjsAttribute",
        "ParamObjsAttribute",
        "CompositeParamObjsAttribute",
    ):
        assert hasattr(gpp, name)
    for old in (
        "AbstractParametersAttribute",
        "ParametersAttribute",
        "CompositeParametersAttribute",
    ):
        assert not hasattr(gpp, old)
