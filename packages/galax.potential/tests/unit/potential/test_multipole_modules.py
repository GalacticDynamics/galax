"""Tests for `galax.potential.multipoleinner`, `multipoleouter` and `multipole`.

The generic conformance tests are in `test_potential_modules`. These pin the
multipole-specific properties: single terms against ``mpmath.spherharm``, the
degree read from the coefficient shape, the z axis, and batches.
"""

import jax
import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp

M = u.Q(1e12, "Msun")
RS = u.Q(5, "kpc")
KW = {"m_tot": M, "r_s": RS, "units": "galactic"}

# A point off the axes, in kpc.
XYZ = np.array([3.0, -2.0, 4.5])
TERMS = [(0, 0), (1, 0), (1, 1), (2, 1)]
L_MAX = 3


def table(l: int, m: int, value: float = 1.0) -> np.ndarray:
    """Return an ``(L_MAX + 1, L_MAX + 1)`` table, zero except at ``[l, m]``."""
    out = np.zeros((L_MAX + 1, L_MAX + 1))
    out[l, m] = value
    return out


def spherharm(l: int, m: int, xyz: np.ndarray) -> tuple[float, float, float]:
    """``(r, Re Y_l^m, Im Y_l^m)`` from mpmath at the position ``xyz``."""
    r = float(np.linalg.norm(xyz))
    theta, phi = float(np.arccos(xyz[2] / r)), float(np.arctan2(xyz[1], xyz[0]))
    y = mp.spherharm(l, m, theta, phi)
    return r, float(y.real), float(y.imag)


@pytest.mark.parametrize(("l", "m"), TERMS)
@pytest.mark.parametrize("which", ["S", "T"])
@pytest.mark.parametrize("kind", ["inner", "outer"])
def test_single_term_matches_mpmath(l, m, which, kind):
    zero = np.zeros((L_MAX + 1, L_MAX + 1))
    one = table(l, m)
    slm, tlm = (one, zero) if which == "S" else (zero, one)
    cls = gp.MultipoleInnerPotential if kind == "inner" else gp.MultipoleOuterPotential
    pot = cls(l_max=L_MAX, Slm=slm, Tlm=tlm, **KW)
    module = getattr(gp, f"multipole{kind}")
    p = pot.params(0.0)

    r, re, im = spherharm(l, m, XYZ)
    s = r / p["r_s"]
    radial = s**l if kind == "inner" else s ** (-(l + 1))
    want = p["G"] * p["m_tot"] / p["r_s"] * radial * (re if which == "S" else im)

    np.testing.assert_allclose(module.potential(p, jnp.asarray(XYZ)), want, rtol=1e-12)
    np.testing.assert_allclose(
        pot.potential(u.Q(XYZ, "kpc"), 0.0).value, want, rtol=1e-12
    )


def test_combined_term_is_the_sum_of_inner_and_outer():
    ins = (table(2, 1, 0.7), table(1, 1, -0.4))
    out = (table(2, 0, 1.3), table(3, 2, 0.2))
    both = gp.MultipolePotential(
        l_max=L_MAX, ISlm=ins[0], ITlm=ins[1], OSlm=out[0], OTlm=out[1], **KW
    )
    inner = gp.MultipoleInnerPotential(l_max=L_MAX, Slm=ins[0], Tlm=ins[1], **KW)
    outer = gp.MultipoleOuterPotential(l_max=L_MAX, Slm=out[0], Tlm=out[1], **KW)
    x = jnp.asarray(XYZ)
    got = gp.multipole.potential(both.params(0.0), x)
    want = gp.multipoleinner.potential(inner.params(0.0), x) + (
        gp.multipoleouter.potential(outer.params(0.0), x)
    )
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_l_max_comes_from_the_shape():
    """Tables of shape ``(5, 5)`` act like a class built with ``l_max = 4``."""
    rng = np.random.default_rng(3)
    slm, tlm = (np.tril(rng.normal(size=(5, 5))) for _ in range(2))
    pot = gp.MultipoleInnerPotential(l_max=4, Slm=slm, Tlm=tlm, **KW)
    p = pot.params(0.0)
    assert p["Slm"].shape == (5, 5)
    x = jnp.asarray(XYZ)
    np.testing.assert_allclose(
        gp.multipoleinner.potential(p, x), pot.potential(u.Q(XYZ, "kpc"), 0.0).value
    )

    # Zero rows and columns beyond l = 2 change nothing, so a table padded to
    # (5, 5) agrees with the class of l_max = 2 built from the (3, 3) block.
    small = gp.MultipoleInnerPotential(l_max=2, Slm=slm[:3, :3], Tlm=tlm[:3, :3], **KW)
    padded = {**small.params(0.0)}
    for k in ("Slm", "Tlm"):
        padded[k] = jnp.zeros((5, 5)).at[:3, :3].set(padded[k])
    np.testing.assert_allclose(
        gp.multipoleinner.potential(padded, x),
        small.potential(u.Q(XYZ, "kpc"), 0.0).value,
        rtol=1e-13,
    )


@pytest.mark.parametrize("l_max", [0, 1, 4])
def test_class_equals_kernel_for_any_l_max(l_max):
    """A kernel given ``(l_max + 1, l_max + 1)`` tables equals the class."""
    rng = np.random.default_rng(l_max)
    slm, tlm = (np.tril(rng.normal(size=(l_max + 1, l_max + 1))) for _ in range(2))
    pot = gp.MultipoleInnerPotential(l_max=l_max, Slm=slm, Tlm=tlm, **KW)
    p = pot.params(0.0)
    np.testing.assert_allclose(
        gp.multipoleinner.potential(p, jnp.asarray(XYZ)),
        pot.potential(u.Q(XYZ, "kpc"), 0.0).value,
        rtol=1e-14,
    )
    if l_max > 0:
        # Dropping the highest degree from the tables changes the value.
        short = {**p, "Slm": p["Slm"][:l_max, :l_max], "Tlm": p["Tlm"][:l_max, :l_max]}
        assert gp.multipoleinner.potential(short, jnp.asarray(XYZ)) != (
            gp.multipoleinner.potential(p, jnp.asarray(XYZ))
        )


def test_wrong_shape_is_rejected_by_the_class():
    with pytest.raises(Exception, match="shape"):
        gp.MultipoleInnerPotential(
            l_max=3, Slm=np.zeros((5, 5)), Tlm=np.zeros((5, 5)), **KW
        )


def test_density_is_one_shared_function():
    assert gp.multipoleinner.density is gp.multipole.density
    assert gp.multipoleouter.density is gp.multipole.density
    out = gp.multipole.density({}, jnp.asarray(XYZ))
    assert out.shape == ()
    assert out == 0.0


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_laplacian_is_exactly_zero(kind):
    ones = table(1, 1)
    if kind == "":
        pot = gp.MultipolePotential(
            l_max=L_MAX, ISlm=ones, ITlm=ones, OSlm=ones, OTlm=ones, **KW
        )
    else:
        cls = (
            gp.MultipoleInnerPotential
            if kind == "inner"
            else gp.MultipoleOuterPotential
        )
        pot = cls(l_max=L_MAX, Slm=ones, Tlm=ones, **KW)
    module = getattr(gp, f"multipole{kind}")
    out = module.laplacian(pot.params(0.0), jnp.asarray(XYZ))
    assert out.shape == ()
    assert out == 0.0
    assert pot.laplacian(u.Q(XYZ, "kpc"), 0.0) == 0.0


@pytest.mark.parametrize("kind", ["inner", "outer"])
def test_gradient_on_the_z_axis(kind):
    """The (1, 1) term is ``∝ x``-like; its z-axis gradient is finite and exact."""
    cls = gp.MultipoleInnerPotential if kind == "inner" else gp.MultipoleOuterPotential
    pot = cls(l_max=L_MAX, Slm=table(1, 1), Tlm=table(1, 1, 0.5), **KW)
    module = getattr(gp, f"multipole{kind}")
    p = pot.params(0.0)
    for z in (0.5, 3.0, -4.0):
        x = jnp.array([0.0, 0.0, z])
        got = module.gradient(p, x)
        assert jnp.all(jnp.isfinite(got))
        np.testing.assert_allclose(got, jax.grad(module.potential, 1)(p, x))
        np.testing.assert_allclose(got, pot.gradient(x, 0.0), rtol=1e-12)
        assert jnp.all(jnp.isfinite(module.hessian(p, x)))


def test_inner_dipole_gradient_is_the_closed_form():
    """``Re Y_1^0 = sqrt(3/4 pi) cos(theta)`` makes the (1, 0) term ``∝ z``."""
    pot = gp.MultipoleInnerPotential(
        l_max=L_MAX, Slm=table(1, 0), Tlm=table(0, 0, 0), **KW
    )
    p = pot.params(0.0)
    want = p["G"] * p["m_tot"] / p["r_s"] ** 2 * np.sqrt(3 / (4 * np.pi))
    for xyz in ([0.0, 0.0, 2.0], [1.0, -2.0, 3.0]):
        got = gp.multipoleinner.gradient(p, jnp.array(xyz))
        np.testing.assert_allclose(got, [0.0, 0.0, want], rtol=1e-12, atol=1e-14 * want)


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_batch_equals_per_point(kind):
    rng = np.random.default_rng(7)
    c = [np.tril(rng.normal(size=(L_MAX + 1, L_MAX + 1))) for _ in range(4)]
    if kind == "":
        pot = gp.MultipolePotential(
            l_max=L_MAX, ISlm=c[0], ITlm=c[1], OSlm=c[2], OTlm=c[3], **KW
        )
    else:
        cls = {
            "inner": gp.MultipoleInnerPotential,
            "outer": gp.MultipoleOuterPotential,
        }[kind]
        pot = cls(l_max=L_MAX, Slm=c[0], Tlm=c[1], **KW)
    module = getattr(gp, f"multipole{kind}")
    p = pot.params(0.0)

    xyz = jnp.asarray(rng.normal(size=(7, 3)) * 4.0)
    batch = module.potential(p, xyz)
    assert batch.shape == (7,)
    each = jnp.stack([module.potential(p, x) for x in xyz])
    np.testing.assert_allclose(batch, each, rtol=1e-13)
    np.testing.assert_allclose(
        jax.vmap(module.potential, (None, 0))(p, xyz), each, rtol=1e-13
    )

    deep = xyz.reshape(7, 1, 3)[:, [0, 0]]
    assert module.potential(p, deep).shape == (7, 2)
    assert module.potential(p, jnp.zeros((0, 3))).shape == (0,)
