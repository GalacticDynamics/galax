"""Tests for `galax.potential.Symmetry` and symmetry-aware input parsing."""

import pytest

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

SPHERICAL_POTS = [
    gp.KeplerPotential(m_tot=u.Q(1e12, "Msun"), units="galactic"),
    gp.HernquistPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
    gp.PlummerPotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
    gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
    gp.IsochronePotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
    gp.JaffePotential(m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
    gp.BurkertPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(5, "kpc"), units="galactic"),
]

MIYAMOTO_NAGAI = gp.MiyamotoNagaiPotential(
    m_tot=u.Q(1e12, "Msun"), a=u.Q(5, "kpc"), b=u.Q(1, "kpc"), units="galactic"
)

T = u.Q(0, "Gyr")
R = cx.vecs.RadialPos(r=u.Q(8.0, "kpc"))
XYZ = cx.CartesianPos3D.from_([8.0, 0.0, 0.0], "kpc")


# ============================================================================
# The enum


@pytest.mark.parametrize("member", list(gp.Symmetry))
def test_roundtrip_through_value(member: gp.Symmetry) -> None:
    """Every member round-trips through its string value."""
    assert gp.Symmetry(member.value) is member
    assert member == member.value


def test_none_is_an_alias() -> None:
    assert gp.Symmetry(None) is gp.Symmetry.NONE


@pytest.mark.parametrize("value", ["axisymmetric", "triaxial", "NONE", ""])
def test_unknown_value_raises(value: str) -> None:
    # The message names the valid values: "axisymmetric" and "triaxial" are
    # the natural guesses, and they are Agama's vocabulary, not this one.
    with pytest.raises(ValueError, match="Unknown symmetry") as excinfo:
        gp.Symmetry(value)
    assert "spherical" in str(excinfo.value)


# ============================================================================
# Declarations


@pytest.mark.parametrize("pot", SPHERICAL_POTS, ids=lambda p: type(p).__name__)
def test_declared_spherical(pot: gp.AbstractPotential) -> None:
    assert pot.symmetry is gp.Symmetry.SPHERICAL


def test_undeclared_is_none() -> None:
    assert MIYAMOTO_NAGAI.symmetry is gp.Symmetry.NONE


# ============================================================================
# Symmetry-aware parsing


@pytest.mark.parametrize("pot", SPHERICAL_POTS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("method", ["potential", "density", "gradient", "hessian"])
def test_radial_matches_cartesian(pot: gp.AbstractPotential, method: str) -> None:
    """A `RadialPos` gives the same answer as the equivalent Cartesian input."""
    assert jnp.all(getattr(pot, method)(R, T) == getattr(pot, method)(XYZ, T))


@pytest.mark.parametrize("method", ["potential", "density", "gradient", "hessian"])
def test_radial_on_nonspherical_raises(method: str) -> None:
    """A radius is ambiguous without spherical symmetry."""
    with pytest.raises(TypeError, match="RadialPos is ambiguous"):
        getattr(MIYAMOTO_NAGAI, method)(R, T)


def test_length_3_quantity_is_still_a_position() -> None:
    """A bare 1-D length-3 Quantity keeps meaning a single ``xyz``."""
    pot = SPHERICAL_POTS[1]
    xyz = u.Q([8.0, 0.0, 0.0], "kpc")
    assert jnp.all(pot.potential(xyz, T) == pot.potential(XYZ, T))


def test_symmetry_may_be_declared_as_a_plain_string() -> None:
    """A `StrEnum` exists so the plain string works; the gate must honour that.

    Declaring `symmetry = "spherical"` is the natural thing to write, and an
    identity check would reject it while reporting `declares symmetry
    'spherical'` -- refusing a spherical potential for not being spherical.
    """

    class StrDeclared(gp.KeplerPotential):
        symmetry = "spherical"

    pot = StrDeclared(m_tot=u.Q(1e12, "Msun"), units="galactic")
    radial = cx.vecs.RadialPos(r=u.Q(8.0, "kpc"))

    assert pot.symmetry == gp.Symmetry.SPHERICAL
    assert jnp.isclose(
        pot.potential(radial, t=0),
        pot.potential(u.Q([8.0, 0.0, 0.0], "kpc"), t=0),
        atol=u.Q(0.0, "kpc2 / Myr2"),
    )


def test_misspelled_symmetry_is_reported_as_invalid_not_ambiguous() -> None:
    """A bad declaration must not masquerade as a non-spherical potential.

    `symmetry` is a plain class attribute, so unlike a `ParameterField` there
    is no converter to catch a typo. Comparing the raw value means a
    misspelling merely tests unequal, and the caller is told a `RadialPos` is
    *ambiguous* for their potential -- sending them to find a direction to
    pass, when the real fault is the spelling. Validate instead.
    """

    class Misspelled(gp.KeplerPotential):
        symmetry = "sphericl"

    pot = Misspelled(m_tot=u.Q(1e12, "Msun"), units="galactic")

    with pytest.raises(ValueError, match="Unknown symmetry 'sphericl'") as e:
        pot.potential(cx.vecs.RadialPos(r=u.Q(8.0, "kpc")), t=0)
    assert "spherical" in str(e.value)  # the message names the valid values


def test_none_declaration_is_normalized_in_the_ambiguity_message() -> None:
    """`None` is a documented alias, so it must report as `none`, not `None`."""

    class NoneDeclared(gp.KeplerPotential):
        symmetry = None

    pot = NoneDeclared(m_tot=u.Q(1e12, "Msun"), units="galactic")

    with pytest.raises(TypeError, match="declares symmetry 'none'"):
        pot.potential(cx.vecs.RadialPos(r=u.Q(8.0, "kpc")), t=0)
