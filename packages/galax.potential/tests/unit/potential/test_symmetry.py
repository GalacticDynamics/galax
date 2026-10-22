"""Tests for `galax.potential.Symmetry` and symmetry-aware input parsing."""

import inspect

import jax
import pytest

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.potential._src.symmetry import invariances

G = "galactic"
M = u.Q(1e12, "Msun")
RS = u.Q(5, "kpc")
_HERNQUIST = gp.HernquistPotential(m_tot=M, r_s=RS, units=G)
_MN = gp.MiyamotoNagaiPotential(m_tot=M, a=RS, b=u.Q(1, "kpc"), units=G)
_LM = jnp.zeros((3, 3)).at[0, 0].set(1.0).at[2, 1].set(0.3)
_NLM = jnp.zeros((2, 3, 3)).at[0, 0, 0].set(1.0).at[1, 2, 1].set(0.3)
# A bar rotated in the x-y plane. It is still even in z (and in (x, y) -> (-x, -y)),
# but no `Symmetry` member captures that alone, so it declares `NONE`.
_BAR = gp.LongMuraliBarPotential(
    m_tot=M, a=RS, b=u.Q(1, "kpc"), c=u.Q(1, "kpc"), alpha=u.Q(0.3, "rad"), units=G
)
_PROFILE = {"r_min": u.Q(1e-2, "kpc"), "r_max": u.Q(1e3, "kpc"), "n_r": 128}

# One instance of every concrete public potential (see
# `test_every_builtin_is_covered`). Parameters are chosen so that a class that
# is not spherical in general is not spherical here either: no unit axis
# ratios, no isotropic coefficients.
BUILTINS = [
    gp.AxisymmetricGaussianPotential(m_tot=M, r_s=RS, q2=0.7, units=G),
    gp.BovyMWPotential2014(),
    gp.BurkertPotential(m=M, r_s=RS, units=G),
    gp.CompositePotential(halo=_HERNQUIST, disk=_MN),
    gp.FlattenedInThePotential(base_potential=_HERNQUIST, q_z=0.7),
    gp.GaussianPotential(m_tot=M, r_s=RS, units=G),
    gp.HardCutoffNFWPotential(m=M, r_s=RS, r_t=u.Q(20, "kpc"), units=G),
    # `omega` may be a 3-vector, so the class is not spherical in general.
    gp.HarmonicOscillatorPotential(omega=u.Q([1.0, 2.0, 3.0], "1/Myr"), units=G),
    gp.HenonHeilesPotential(
        coeff=u.Q(1.0, "1/kpc"), timescale=u.Q(1.0, "Myr"), units=G
    ),
    _HERNQUIST,
    gp.IsochronePotential(m_tot=M, r_s=RS, units=G),
    gp.JaffePotential(m_tot=M, r_s=RS, units=G),
    gp.KeplerPotential(m_tot=M, units=G),
    gp.KuzminPotential(m_tot=M, r_s=RS, units=G),
    gp.KuzminKutuzovPotential(m_tot=M, a=RS, c=u.Q(1, "kpc"), units=G),
    gp.LM10Potential(),
    gp.LMJ09LogarithmicPotential(
        v_c=u.Q(220, "km/s"),
        r_s=RS,
        q1=1.0,
        q2=0.8,
        q3=0.6,
        phi=u.Q(0.3, "rad"),
        units=G,
    ),
    gp.LeeSutoTriaxialNFWPotential(m=M, r_s=RS, a1=1.0, a2=0.8, a3=0.6, units=G),
    gp.LogarithmicPotential(v_c=u.Q(220, "km/s"), r_s=RS, units=G),
    _BAR,
    gp.MN3ExponentialPotential(m_tot=M, h_R=RS, h_z=u.Q(0.3, "kpc"), units=G),
    gp.MN3Sech2Potential(m_tot=M, h_R=RS, h_z=u.Q(0.3, "kpc"), units=G),
    gp.MilkyWayPotential(),
    gp.MilkyWayPotential2022(),
    _MN,
    gp.MonariEtAl2016BarPotential(
        alpha=u.Q(0.01, ""),
        R0=u.Q(8.0, "kpc"),
        v0=u.Q(220.0, "km/s"),
        Rb=u.Q(3.5, "kpc"),
        phi_b=u.Q(25, "deg"),
        Omega=u.Q(52.2, "km/(s kpc)"),
        units=G,
    ),
    gp.MultipoleInnerPotential(
        m_tot=M, r_s=RS, l_max=2, Slm=_LM, Tlm=jnp.zeros_like(_LM), units=G
    ),
    gp.MultipoleOuterPotential(
        m_tot=M, r_s=RS, l_max=2, Slm=_LM, Tlm=jnp.zeros_like(_LM), units=G
    ),
    gp.MultipolePotential(
        m_tot=M,
        r_s=RS,
        l_max=2,
        ISlm=_LM,
        ITlm=jnp.zeros_like(_LM),
        OSlm=_LM,
        OTlm=jnp.zeros_like(_LM),
        units=G,
    ),
    # `symmetry` is per instance here: one of each.
    gp.MultipoleProfilePotential.from_potential(
        _HERNQUIST, l_max=0, symmetry="spherical", **_PROFILE
    ),
    gp.MultipoleProfilePotential.from_potential(
        _BAR,
        l_max=2,
        **_PROFILE,
    ),
    gp.NFWPotential(m=M, r_s=RS, units=G),
    gp.NullPotential(),
    gp.PlummerPotential(m_tot=M, r_s=RS, units=G),
    gp.PowerLawCutoffPotential(m_tot=M, alpha=1.5, r_c=RS, units=G),
    gp.SCFPotential(m_tot=M, r_s=RS, Snlm=_NLM, Tnlm=jnp.zeros_like(_NLM), units=G),
    gp.SatohPotential(m_tot=M, a=RS, b=u.Q(1, "kpc"), units=G),
    gp.StoneOstriker15Potential(m_tot=M, r_c=u.Q(1, "kpc"), r_h=RS, units=G),
    gp.TransformedPotential(
        _HERNQUIST, cx.ops.GalileanSpatialTranslation(u.Q([1.0, 0, 0], "kpc"))
    ),
    gp.TranslatedPotential(
        base_potential=_HERNQUIST, translation=u.Q([1.0, 0, 0], "kpc")
    ),
    gp.TriaxialGaussianPotential(m_tot=M, r_s=RS, q1=0.8, q2=0.6, units=G),
    gp.TriaxialHernquistPotential(m_tot=M, r_s=RS, q1=0.8, q2=0.6, units=G),
    gp.TriaxialInThePotential(base_potential=_HERNQUIST, y_over_x=0.8, z_over_x=0.6),
    gp.TriaxialNFWPotential(m=M, r_s=RS, q1=0.8, q2=0.6, units=G),
    gp.Vogelsberger08TriaxialNFWPotential(m=M, r_s=RS, q1=0.8, a_r=1.0, units=G),
    gp.ZhaoPotential(m=M, r_s=RS, alpha=0.5, beta=4.0, gamma=1.0, units=G),
    gp.gNFWPotential(m=M, r_s=RS, gamma=0.5, units=G),
]

SPHERICAL_POTS = [p for p in BUILTINS if p.symmetry == gp.Symmetry.SPHERICAL]
OTHER_POTS = [p for p in BUILTINS if p.symmetry != gp.Symmetry.SPHERICAL]

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
    # the natural guesses, and neither is this vocabulary's word for anything.
    with pytest.raises(ValueError, match="Unknown symmetry") as excinfo:
        gp.Symmetry(value)
    assert "spherical" in str(excinfo.value)


# ============================================================================
# Declarations


def _ids(pot: gp.AbstractPotential) -> str:
    return type(pot).__name__


def test_every_builtin_is_covered() -> None:
    """A new public potential must join `BUILTINS`, so its symmetry is checked."""
    concrete = {
        obj
        for name in gp.__all__
        if inspect.isclass(obj := getattr(gp, name))
        and issubclass(obj, gp.AbstractPotential)
        and not inspect.isabstract(obj)
        and not name.startswith("Abstract")
    }
    assert concrete - {type(p) for p in BUILTINS} == set()


# Points on the sphere r = 3 kpc, in directions with no special alignment.
_DIRS = jax.random.normal(jax.random.key(0), (16, 3))
_XYZ = u.Q(3.0 * _DIRS / jnp.linalg.vector_norm(_DIRS, axis=-1, keepdims=True), "kpc")


def _on_sphere(pot: gp.AbstractPotential) -> tuple[jax.Array, jax.Array]:
    phi = pot.potential(_XYZ, T).ustrip(pot.units["specific energy"])
    grad = pot.gradient(_XYZ, T).ustrip(pot.units["acceleration"])
    return phi, grad


@pytest.mark.parametrize("pot", SPHERICAL_POTS, ids=_ids)
def test_declared_spherical_is_rotation_invariant(pot: gp.AbstractPotential) -> None:
    """On a sphere, Φ is constant and ∇Φ is radial with constant magnitude."""
    phi, grad = _on_sphere(pot)
    rhat = _XYZ.value / 3.0
    g_r = jnp.sum(grad * rhat, axis=-1)
    scale = jnp.max(jnp.abs(g_r)) + 1e-300  # `NullPotential` is all zeros

    assert jnp.allclose(phi, phi[0], rtol=1e-10, atol=0)
    assert jnp.allclose(g_r, g_r[0], rtol=1e-10, atol=0)
    assert jnp.max(jnp.abs(grad - g_r[:, None] * rhat)) <= 1e-10 * scale


@pytest.mark.parametrize("pot", OTHER_POTS, ids=_ids)
def test_undeclared_is_not_spherical(pot: gp.AbstractPotential) -> None:
    """A potential that is spherical must say so: Φ varies on the sphere here."""
    phi, _ = _on_sphere(pot)
    assert jnp.max(jnp.abs(phi - phi[0])) > 1e-6 * jnp.max(jnp.abs(phi))


# A generator of each invariance: rotations by angles with no special
# alignment, and the plane reflections.
_C, _S = jnp.cos(0.7), jnp.sin(0.7)
_ROT_Z = jnp.asarray([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
_ROT_X = jnp.asarray([[1.0, 0.0, 0.0], [0.0, _C, -_S], [0.0, _S, _C]])
_OPS = {
    "rotation": _ROT_X @ _ROT_Z,
    "z_rotation": _ROT_Z,
    "x_reflection": jnp.diag(jnp.asarray([-1.0, 1.0, 1.0])),
    "y_reflection": jnp.diag(jnp.asarray([1.0, -1.0, 1.0])),
    "z_reflection": jnp.diag(jnp.asarray([1.0, 1.0, -1.0])),
}


def _is_invariant(pot: gp.AbstractPotential, op: jax.Array) -> bool:
    """Whether Φ(Mx) = Φ(x) and ∇Φ(Mx) = M ∇Φ(x) on the test sphere."""
    phi, grad = _on_sphere(pot)
    xyz = u.Q(_XYZ.value @ op.T, "kpc")
    phi_op = pot.potential(xyz, T).ustrip(pot.units["specific energy"])
    grad_op = pot.gradient(xyz, T).ustrip(pot.units["acceleration"])
    return bool(
        jnp.allclose(phi_op, phi, rtol=1e-10, atol=0)
        and jnp.allclose(grad_op, grad @ op.T, rtol=1e-8, atol=1e-12)
    )


@pytest.mark.parametrize(
    ("pot", "inv"),
    [(p, i) for p in BUILTINS for i in sorted(invariances(p.symmetry))],
    ids=lambda x: x if isinstance(x, str) else _ids(x),
)
def test_declared_symmetry_holds(pot: gp.AbstractPotential, inv: str) -> None:
    """Each invariance a potential declares, directly or from its parts, holds."""
    assert _is_invariant(pot, _OPS[inv])


@pytest.mark.parametrize(
    ("pot", "stronger"),
    [
        (p, s)
        for p in BUILTINS
        for s in gp.Symmetry
        if not invariances(s) <= invariances(p.symmetry)
    ],
    ids=lambda x: x.value if isinstance(x, gp.Symmetry) else _ids(x),
)
def test_no_stronger_symmetry_than_declared(
    pot: gp.AbstractPotential, stronger: gp.Symmetry
) -> None:
    """A potential must declare the strongest symmetry it has.

    For each member the declaration does not imply, at least one of that
    member's invariances fails here -- so a missing or too-weak flag is caught.
    """
    assert not all(_is_invariant(pot, _OPS[i]) for i in sorted(invariances(stronger)))


@pytest.mark.parametrize(
    ("pot", "expected"),
    [
        (gp.CompositePotential(halo=_HERNQUIST, disk=_MN), "zrotation_zreflection"),
        (gp.CompositePotential(a=_HERNQUIST, b=_HERNQUIST), "spherical"),
        (
            gp.CompositePotential(
                halo=_HERNQUIST,
                bar=gp.TriaxialHernquistPotential(
                    m_tot=M, r_s=RS, q1=0.8, q2=0.6, units=G
                ),
            ),
            "plane_reflection",
        ),
        (gp.CompositePotential(halo=_HERNQUIST, bar=_BAR), "none"),
        (
            gp.FlattenedInThePotential(base_potential=_HERNQUIST, q_z=0.7),
            "zrotation_zreflection",
        ),
        (
            gp.FlattenedInThePotential(base_potential=_MN, q_z=0.7),
            "zrotation_zreflection",
        ),
        (
            gp.TriaxialInThePotential(base_potential=_MN, y_over_x=0.8, z_over_x=0.6),
            "plane_reflection",
        ),
        (
            gp.TranslatedPotential(
                base_potential=_HERNQUIST, translation=u.Q([1.0, 0, 0], "kpc")
            ),
            "none",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else _ids(x),
)
def test_derived_symmetry(pot: gp.AbstractPotential, expected: str) -> None:
    """Composites keep what their parts share; wrappers keep what they preserve."""
    assert pot.symmetry is gp.Symmetry(expected)


def test_undeclared_is_none() -> None:
    assert _BAR.symmetry is gp.Symmetry.NONE


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
    with pytest.raises(TypeError, match="RadialPos needs 'spherical'"):
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
