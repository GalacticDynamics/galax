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
    ins = (table(2, 1, 0.7), table(2, 1, -0.4))  # off-diagonal: [l, m] != [m, l]
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
    """Tables padded to ``(5, 5)`` act like a class built with ``l_max = 2``."""
    rng = np.random.default_rng(3)
    slm, tlm = (np.tril(rng.normal(size=(3, 3))) for _ in range(2))
    x = jnp.asarray(XYZ)
    # Zero rows and columns beyond l = 2 change nothing.
    small = gp.MultipoleInnerPotential(l_max=2, Slm=slm, Tlm=tlm, **KW)
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


@pytest.mark.parametrize(
    ("cls", "names"),
    [
        (gp.MultipoleInnerPotential, ("Slm", "Tlm")),
        (gp.MultipoleOuterPotential, ("Slm", "Tlm")),
        (gp.MultipolePotential, ("ISlm", "ITlm", "OSlm", "OTlm")),
    ],
    ids=["inner", "outer", "multipole"],
)
def test_wrong_shape_is_rejected_by_the_class(cls, names):
    """Each table is checked: one table of the wrong shape is enough."""
    for bad in names:
        tables = {n: np.zeros((5, 5) if n == bad else (4, 4)) for n in names}
        with pytest.raises(Exception, match="shape"):
            cls(l_max=3, **tables, **KW)


def test_density_is_one_shared_function():
    assert gp.multipoleinner.density is gp.multipole.density
    assert gp.multipoleouter.density is gp.multipole.density
    out = gp.multipole.density({}, jnp.asarray(XYZ))
    assert out.shape == ()
    assert out == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3,), (4, 3), (0, 3)])
def test_density_keeps_shape_and_dtype(dtype, shape):
    xyz = jnp.zeros(shape, dtype=dtype)
    for module in (gp.multipole, gp.multipoleinner, gp.multipoleouter):
        out = module.density({}, xyz)
        assert out.shape == shape[:-1]
        assert out.dtype == dtype


def _pot(kind):
    ones = table(1, 1)
    if kind == "":
        return gp.MultipolePotential(
            l_max=L_MAX, ISlm=ones, ITlm=ones, OSlm=ones, OTlm=ones, **KW
        )
    cls = gp.MultipoleInnerPotential if kind == "inner" else gp.MultipoleOuterPotential
    return cls(l_max=L_MAX, Slm=ones, Tlm=ones, **KW)


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_class_converts_the_length_unit(kind):
    """``_potential`` strips positions in the unit system of the class.

    The public ``potential`` converts first, so call the private method, which
    is what gets a position in another length unit.
    """
    pot = _pot(kind)
    kpc = pot._potential(u.Q(XYZ, "kpc"), 0.0)
    for x in (
        u.Q(np.asarray(XYZ) * 1000, "pc"),
        u.Q(np.asarray(XYZ) * 3.0856775814913673e19, "m"),
    ):
        np.testing.assert_allclose(pot._potential(x, 0.0), kpc, rtol=1e-12)


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_laplacian_is_exactly_zero(kind):
    pot = _pot(kind)
    module = getattr(gp, f"multipole{kind}")
    out = module.laplacian(pot.params(0.0), jnp.asarray(XYZ))
    assert out.shape == ()
    assert out == 0.0
    assert pot.laplacian(u.Q(XYZ, "kpc"), 0.0) == 0.0


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_gradient_on_the_z_axis(kind):
    """On the z axis the (1, 1) terms have a closed-form gradient.

    ``Re Y_1^1 = -k sin(theta) cos(phi)`` and ``Im Y_1^1 = -k sin(theta)
    sin(phi)`` with ``k = sqrt(3/8 pi)``, so the inner term is ``-k x`` and
    ``-k y`` times ``G m / r_s^2``, uniform, and the outer term is ``-k x / r^3``
    and ``-k y / r^3`` times ``G m r_s``. Slm = 1 and Tlm = 0.5 (at ``[1, 1]``).
    """
    ones = table(1, 1)
    if kind == "":
        pot = gp.MultipolePotential(
            l_max=L_MAX, ISlm=ones, ITlm=0.5 * ones, OSlm=ones, OTlm=0.5 * ones, **KW
        )
    else:
        cls = (
            gp.MultipoleInnerPotential
            if kind == "inner"
            else gp.MultipoleOuterPotential
        )
        pot = cls(l_max=L_MAX, Slm=ones, Tlm=0.5 * ones, **KW)
    module = getattr(gp, f"multipole{kind}")
    p = pot.params(0.0)
    k = np.sqrt(3 / (8 * np.pi))
    gm = float(p["G"] * p["m_tot"])
    r_s = float(p["r_s"])
    for z in (0.5, 3.0, -4.0):
        scale = 0.0
        if kind in ("inner", ""):
            scale += gm / r_s**2
        if kind in ("outer", ""):
            scale += gm * r_s / abs(z) ** 3
        want = -k * scale * np.array([1.0, 0.5, 0.0])
        x = jnp.array([0.0, 0.0, z])
        got = module.gradient(p, x)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14 * abs(want).max())
        np.testing.assert_allclose(
            pot.gradient(x, 0.0), want, rtol=1e-12, atol=1e-14 * abs(want).max()
        )
        # The inner (1, 1) terms are linear, so their Hessian is zero. The outer
        # terms are ``-k C x / r^3`` and ``-k C y / r^3`` with ``C = G m r_s``,
        # with the mixed derivative ``3 k C z / |z|^5`` at x = y = 0.
        h = 0.0
        if kind in ("outer", ""):
            h = 3 * k * gm * r_s * z / abs(z) ** 5
        hess = np.zeros((3, 3))
        hess[0, 2] = hess[2, 0] = h
        hess[1, 2] = hess[2, 1] = 0.5 * h
        np.testing.assert_allclose(
            module.hessian(p, x), hess, rtol=1e-10, atol=1e-10 * gm / r_s**3
        )


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


@pytest.mark.parametrize("kind", ["inner", "outer", ""])
def test_class_follows_a_time_dependent_table(kind):
    """The class passes the parameters at ``t``, not at 0, to the kernel."""
    slope = u.Q(table(2, 1, 0.1), "1 / Myr")
    start = u.Q(table(2, 1), "")
    lin = gp.params.LinearParameter(
        slope=slope, point_time=u.Q(0, "Myr"), point_value=start
    )
    ones = table(1, 1)
    if kind == "":
        pot = gp.MultipolePotential(
            l_max=L_MAX, ISlm=lin, ITlm=ones, OSlm=lin, OTlm=ones, **KW
        )
    else:
        cls = (
            gp.MultipoleInnerPotential
            if kind == "inner"
            else gp.MultipoleOuterPotential
        )
        pot = cls(l_max=L_MAX, Slm=lin, Tlm=ones, **KW)
    module = getattr(gp, f"multipole{kind}")
    x, t = u.Q(XYZ, "kpc"), u.Q(5.0, "Myr")

    p0, p5 = pot.params(0.0), pot.params(t)
    assert not np.allclose(
        module.potential(p0, jnp.asarray(XYZ)), module.potential(p5, jnp.asarray(XYZ))
    )
    np.testing.assert_allclose(
        pot.potential(x, t).value, module.potential(p5, jnp.asarray(XYZ)), rtol=1e-13
    )


def test_inner_value_is_finite_at_the_origin():
    """At r = 0 the value is the l = 0 term and the derivatives are finite."""
    for l_max in (0, 2):
        slm = np.zeros((l_max + 1, l_max + 1))
        slm[0, 0] = 1.0
        if l_max:
            slm[2, 1] = 0.7
        pot = gp.MultipoleInnerPotential(
            l_max=l_max, Slm=slm, Tlm=np.zeros_like(slm), **KW
        )
        p, x0 = pot.params(0.0), jnp.zeros(3)
        want = float(p["G"] * p["m_tot"] / p["r_s"]) / np.sqrt(4 * np.pi)
        np.testing.assert_allclose(gp.multipoleinner.potential(p, x0), want, rtol=1e-12)
        grad = gp.multipoleinner.gradient(p, x0)
        assert jnp.all(jnp.isfinite(grad))
        np.testing.assert_allclose(grad, 0.0, atol=1e-12 * want / float(p["r_s"]))
        assert jnp.all(jnp.isfinite(gp.multipoleinner.hessian(p, x0)))
