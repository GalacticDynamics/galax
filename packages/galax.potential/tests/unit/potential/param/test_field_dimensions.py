"""A parameter function's return annotation must record its dimension."""

import pytest
from unxts.parametric import ParametricQuantity

import unxt as u

import galax.potential as gp


def test_parametric_annotation_is_accepted() -> None:
    """`ParametricQuantity[...]` carries the dimension, so it can be checked."""

    def m_of_t(t: ParametricQuantity["time"]) -> ParametricQuantity["mass"]:
        return u.Q(1e12, "Msun")

    pot = gp.KeplerPotential(m_tot=m_of_t, units="galactic")
    assert pot.m_tot(u.Q(0, "Gyr")).unit == u.unit("solMass")


def test_bare_quantity_annotation_is_rejected() -> None:
    """Unxt v2's `Quantity` is not parametric, so it records no dimension.

    Accepting it would mean silently skipping the check that a parameter
    function returns what the field declares, so it is an error that names
    the fix.
    """

    def m_of_t(t: u.Quantity["time"]) -> u.Quantity["mass"]:
        return u.Q(1e12, "Msun")

    with pytest.raises(TypeError, match="must record its dimension"):
        gp.KeplerPotential(m_tot=m_of_t, units="galactic")


def test_wrong_dimension_is_still_caught() -> None:
    """The check itself: a length where a mass belongs."""

    def wrong(t: ParametricQuantity["time"]) -> ParametricQuantity["length"]:
        return u.Q(1.0, "kpc")

    with pytest.raises(ValueError, match="dimensions consistent"):
        gp.KeplerPotential(m_tot=wrong, units="galactic")


def test_non_quantity_annotation_is_rejected() -> None:
    """The pre-existing branch, for an annotation that is no quantity at all."""

    def m_of_t(t: ParametricQuantity["time"]) -> float:
        return 1e12

    with pytest.raises(TypeError, match="must be a Quantity"):
        gp.KeplerPotential(m_tot=m_of_t, units="galactic")
