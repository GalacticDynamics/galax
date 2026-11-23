"""Tests for `galax.orbits.circulation` and `galax.orbits.align_circulation`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import coordinax as cx
import quaxed.numpy as qnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go
import galax.potential as gp

T = jnp.linspace(0.0, 40.0, 400)
C, S, Z = jnp.cos(T), jnp.sin(T), jnp.zeros_like(T)


def _orbit(x, y, z):
    q = jnp.stack([x, y, z], axis=1)
    return q, jnp.gradient(q, T, axis=0)


ORBITS = {
    "xy": (_orbit(C, S, Z), (False, False, True), 2),
    "yz": (_orbit(Z, C, S), (True, False, False), 0),
    "xz": (_orbit(C, Z, S), (False, True, False), 1),
    "box": (
        _orbit(C, 0.7 * jnp.cos(1.3 * T), 0.5 * jnp.cos(1.7 * T)),
        (False, False, False),
        2,
    ),
}


def _check_aligned(q, p, expected, axis, axis_exp):
    assert int(axis) == axis_exp
    if any(expected):
        L = jnp.cross(q, p)
        assert jnp.all(L[:, 2] * L[0, 2] > 0)
        assert jnp.allclose(L[:, :2], 0, atol=1e-6)


# ===================================================================
# Arrays


@pytest.mark.parametrize("name", list(ORBITS))
@pytest.mark.parametrize("jit", [False, True])
def test_arrays(name, jit):
    (q, p), expected, axis_exp = ORBITS[name]
    circ_f = jax.jit(go.circulation) if jit else go.circulation
    align_f = jax.jit(go.align_circulation) if jit else go.align_circulation

    circ = circ_f(q, p)
    assert circ.dtype == bool
    assert tuple(np.asarray(circ)) == expected

    q2, p2, axis = align_f(q, p, circ)
    _check_aligned(q2, p2, expected, axis, axis_exp)

    # circ is optional
    q3, p3, axis3 = align_f(q, p)
    assert jnp.array_equal(q3, q2)
    assert jnp.array_equal(p3, p2)
    assert int(axis3) == axis_exp


def test_batch_and_vmap():
    qs = jnp.stack([v[0][0] for v in ORBITS.values()])
    ps = jnp.stack([v[0][1] for v in ORBITS.values()])
    expected = np.array([v[1] for v in ORBITS.values()])
    axes = [v[2] for v in ORBITS.values()]

    # Batch axes are handled by the function itself.
    circ = go.circulation(qs, ps)
    assert circ.shape == (4, 3)
    np.testing.assert_array_equal(circ, expected)
    # Two batch axes.
    circ2 = go.circulation(jnp.stack([qs, qs]), jnp.stack([ps, ps]))
    np.testing.assert_array_equal(circ2, np.stack([expected, expected]))

    # And by an outer vmap.
    circ_v = jax.jit(jax.vmap(go.circulation))(qs, ps)
    np.testing.assert_array_equal(circ_v, expected)

    q2, p2, axis = go.align_circulation(qs, ps, circ)
    q2v, _, axisv = jax.jit(jax.vmap(go.align_circulation))(qs, ps, circ)
    assert jnp.array_equal(q2, q2v)
    assert jnp.array_equal(axis, axisv)
    for i, name in enumerate(ORBITS):
        _check_aligned(q2[i], p2[i], ORBITS[name][1], axis[i], axes[i])


def test_skips_nonfinite_samples():
    (q, p), expected, _ = ORBITS["xy"]
    q = q.at[0].set(jnp.nan).at[200].set(jnp.nan)
    assert tuple(np.asarray(go.circulation(q, p))) == expected


def test_quantities():
    (q, p), expected, axis_exp = ORBITS["yz"]
    qq, pq = u.Q(q, "kpc"), u.Q(p, "km/s")
    assert tuple(np.asarray(go.circulation(qq, pq))) == expected
    # The floor is relative, so tiny units do not change the answer.
    assert tuple(np.asarray(go.circulation(1e-20 * q, 1e-20 * p))) == expected

    q2, p2, axis = go.align_circulation(qq, pq)
    assert isinstance(q2, u.AbstractQuantity)
    assert q2.unit == qq.unit
    assert p2.unit == pq.unit
    _check_aligned(q2.value, p2.value, expected, axis, axis_exp)


def test_align_gradient_is_finite():
    (q, p), _, _ = ORBITS["yz"]

    def loss(q, p):
        q2, p2, _ = go.align_circulation(q, p)
        return jnp.sum(q2[:, 2] ** 2 + p2[:, 2] ** 2)

    gq, gp_ = jax.grad(loss, argnums=(0, 1))(q, p)
    assert jnp.all(jnp.isfinite(gq))
    assert jnp.all(jnp.isfinite(gp_))
    # The x column was moved to z, so only it gets a gradient.
    assert jnp.allclose(gq[:, 0], 2 * q[:, 0])
    assert jnp.allclose(gq[:, 1:], 0)


def test_float32_large_magnitudes():
    """|L|^2 would overflow float32 here; the result must not change."""
    (q, p), expected, _ = ORBITS["yz"]
    q, p = q.astype(jnp.float32), p.astype(jnp.float32)
    # A Milky Way orbit in m and m/s has |L| of about 5e25.
    big = go.circulation(q * 3e20, p * 2e5)
    np.testing.assert_array_equal(big, expected)
    np.testing.assert_array_equal(go.circulation(q * 1e-20, p * 1e-20), expected)


@pytest.mark.parametrize(
    ("q", "p"),
    [([[1.1, 1.1, 1.1]], [[0.3, 0.3, 0.3]]), ([[1.1, 2.2, 3.3]], [[0.77, 1.54, 2.31]])],
)
def test_radial_samples_do_not_circulate(q, p):
    """The rounding residual of q x p for q parallel to p is not circulation."""
    q, p = jnp.asarray(q), jnp.asarray(p)
    np.testing.assert_array_equal(go.circulation(q, p), [False, False, False])


# ===================================================================
# Orbits and coordinates


@pytest.fixture(scope="module")
def pot():
    return gp.KeplerPotential(m_tot=u.Q(1e11, "Msun"), units="galactic")


@pytest.fixture(scope="module")
def w0():
    """Circular orbits in the xy, yz and xz planes."""
    return gc.PhaseSpaceCoordinate(
        q=u.Q([[8.0, 0, 0], [0, 8.0, 0], [8.0, 0, 0]], "kpc"),
        p=u.Q([[0, 230.0, 0], [0, 0, 230.0], [0, 0, 230.0]], "km/s"),
        t=u.Q([0.0, 0.0, 0.0], "Myr"),
    )


EXPECTED_ORBITS = np.array(
    [[False, False, True], [True, False, False], [False, True, False]]
)


@pytest.fixture(scope="module")
def orbit(pot, w0):
    ts = u.Q(jnp.linspace(0, 1, 100), "Gyr")
    return go.compute_orbit(pot, w0, ts, dense=True)


def test_orbit(orbit):
    circ = go.circulation(orbit)
    assert circ.shape == (3, 3)
    np.testing.assert_array_equal(circ, EXPECTED_ORBITS)
    np.testing.assert_array_equal(orbit.circulation(), EXPECTED_ORBITS)
    np.testing.assert_array_equal(jax.jit(go.circulation)(orbit), EXPECTED_ORBITS)


def _cart(obj):
    q = jnp.stack([obj.q.x.value, obj.q.y.value, obj.q.z.value], -1)
    p = jnp.stack([obj.p.x.value, obj.p.y.value, obj.p.z.value], -1)
    return q, p


def test_orbit_time_refinement():
    """A sign flip between stored samples is found on a finer time grid."""
    # An inclined orbit in a flattened potential is a z tube whose L_x and L_y
    # rotate about z and change sign.
    pot = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3, "kpc"), b=u.Q(0.3, "kpc"), units="galactic"
    )
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([8.0, 0, 0], "kpc"), p=u.Q([0, 180.0, 60.0], "km/s"), t=u.Q(0.0, "Myr")
    )
    ts = u.Q(jnp.linspace(0, 2, 4000), "Gyr")
    fine = go.compute_orbit(pot, w0, ts, dense=True)
    np.testing.assert_array_equal(go.circulation(fine), [False, False, True])

    # Store only samples at which L_y has the sign it starts with, so the
    # stored samples miss the sign flips of L_y.
    L = jnp.cross(*_cart(fine))
    keep = np.flatnonzero(np.asarray(L[:, 1] * L[0, 1] > 0))
    assert len(keep) < len(ts)
    coarse = go.compute_orbit(pot, w0, ts[keep[::50]], dense=True)
    np.testing.assert_array_equal(go.circulation(coarse), [False, True, True])

    # Evaluating the interpolant on a fine grid finds them.
    t_fine = u.Q(jnp.linspace(0, 1, 2000), "Gyr") * (coarse.t[-1] / u.Q(1, "Gyr"))
    circ = go.circulation(coarse, t_fine)
    np.testing.assert_array_equal(circ, [False, False, True])
    # A plain array in the orbit's time unit works too, and so does the method.
    t_arr = u.ustrip(coarse.t.unit, t_fine)
    np.testing.assert_array_equal(go.circulation(coarse, t_arr), circ)
    np.testing.assert_array_equal(coarse.circulation(t_fine), circ)


def test_coordinate(orbit):
    w = gc.PhaseSpaceCoordinate(q=orbit.q, p=orbit.p, t=orbit.t)
    np.testing.assert_array_equal(go.circulation(w), EXPECTED_ORBITS)


def test_coordinate_rejects_times(orbit):
    w = gc.PhaseSpaceCoordinate(q=orbit.q, p=orbit.p, t=orbit.t)
    with pytest.raises(TypeError, match="no interpolant"):
        go.circulation(w, orbit.t)


@pytest.mark.parametrize("as_coordinate", [False, True])
def test_align_orbit_and_coordinate(orbit, as_coordinate):
    obj = (
        gc.PhaseSpaceCoordinate(q=orbit.q, p=orbit.p, t=orbit.t)
        if as_coordinate
        else orbit
    )
    aligned, axis = go.align_circulation(obj)
    assert type(aligned) is type(obj)
    np.testing.assert_array_equal(axis, [2, 0, 1])
    np.testing.assert_array_equal(go.circulation(aligned), [[False, False, True]] * 3)
    if not as_coordinate:
        assert aligned.interpolant is None
        _, axis_m = orbit.align_circulation()
        np.testing.assert_array_equal(axis_m, axis)

    # L is parallel to z after the alignment.
    L = jnp.cross(*_cart(aligned))
    assert jnp.allclose(L[..., :2], 0, atol=1e-10 * float(jnp.abs(L).max()))

    # An explicit circ gives the same result.
    aligned2, axis2 = go.align_circulation(obj, go.circulation(obj))
    np.testing.assert_array_equal(axis2, axis)
    assert jnp.array_equal(aligned2.q.x.value, aligned.q.x.value)


# ===================================================================
# Edge cases


def test_all_masked_orbit_circulates_about_no_axis():
    (q, p), _, _ = ORBITS["xy"]
    q = jnp.full_like(q, jnp.nan)
    np.testing.assert_array_equal(go.circulation(q, p), [False, False, False])
    # A batch with one masked orbit leaves the others alone.
    (q2, p2), expected, _ = ORBITS["yz"]
    circ = go.circulation(jnp.stack([q, q2]), jnp.stack([p, p2]))
    np.testing.assert_array_equal(circ, [[False, False, False], expected])


def test_zero_samples_raise():
    (q, p), _, _ = ORBITS["xy"]
    with pytest.raises(ValueError, match="T > 0"):
        go.circulation(q[:0], p[:0])


def test_circ_shape_is_checked(orbit):
    (q, p), _, _ = ORBITS["yz"]
    bad = jnp.array([[True, False, False], [False, False, True]])
    with pytest.raises(ValueError, match="circ has shape"):
        go.align_circulation(q, p, bad)
    with pytest.raises(ValueError, match="circ has shape"):
        jax.jit(go.align_circulation)(q, p, bad)
    with pytest.raises(ValueError, match="circ has shape"):
        go.align_circulation(orbit, jnp.array([True, False, False]))


def test_aligned_orbit_has_no_interpolant(orbit):
    aligned, _ = go.align_circulation(orbit)
    with pytest.raises(Exception, match="rebuilt by align_circulation"):
        aligned(orbit.t)


# ===================================================================
# Composite coordinates


@pytest.fixture(scope="module")
def composite(orbit):
    """Two halves of the orbits, joined along time."""
    a = gc.PhaseSpaceCoordinate(q=orbit.q[:, :50], p=orbit.p[:, :50], t=orbit.t[:50])
    b = gc.PhaseSpaceCoordinate(q=orbit.q[:, 50:], p=orbit.p[:, 50:], t=orbit.t[50:])
    return gc.CompositePhaseSpaceCoordinate(a=a, b=b)


def test_composite(composite):
    np.testing.assert_array_equal(go.circulation(composite), EXPECTED_ORBITS)

    aligned, axis = go.align_circulation(composite)
    assert type(aligned) is type(composite)
    assert set(aligned.keys()) == {"a", "b"}
    np.testing.assert_array_equal(axis, [2, 0, 1])
    np.testing.assert_array_equal(go.circulation(aligned), [[False, False, True]] * 3)
    L = jnp.cross(*_cart(aligned))
    assert jnp.allclose(L[..., :2], 0, atol=1e-10 * float(jnp.abs(L).max()))

    # An explicit circ reaches the coordinate overload, not the array one.
    aligned2, axis2 = go.align_circulation(composite, go.circulation(composite))
    np.testing.assert_array_equal(axis2, axis)
    assert jnp.array_equal(aligned2.q.x.value, aligned.q.x.value)


def test_composite_mixed_representations(orbit):
    """Components in different representations are joined as Cartesian."""
    q, p, t = orbit.q[0], orbit.p[0], orbit.t
    a = gc.PhaseSpaceCoordinate(q=q[:50], p=p[:50], t=t[:50])
    b = gc.PhaseSpaceCoordinate(
        q=cx.vconvert(cx.vecs.SphericalPos, q[50:]),
        p=cx.vconvert(cx.vecs.SphericalVel, p[50:], q[50:]),
        t=t[50:],
    )
    composite = gc.CompositePhaseSpaceCoordinate(a=a, b=b)
    np.testing.assert_array_equal(go.circulation(composite), EXPECTED_ORBITS[0])

    aligned, axis = go.align_circulation(composite)
    assert int(axis) == 2
    np.testing.assert_array_equal(go.circulation(aligned), EXPECTED_ORBITS[0])


@pytest.mark.parametrize(
    "t",
    [u.Q(0.5, "Gyr"), u.Q(jnp.linspace(0, 1, 40).reshape(2, 20), "Gyr")],
    ids=["scalar", "2d"],
)
def test_orbit_t_must_be_1d(orbit, t):
    with pytest.raises(ValueError, match=r"needs t of shape \(T,\)"):
        go.circulation(orbit, t)
    with pytest.raises(ValueError, match=r"needs t of shape \(T,\)"):
        jax.jit(go.circulation)(orbit, t)


def test_orbit_t_none_uses_stored_samples(orbit):
    np.testing.assert_array_equal(go.circulation(orbit, None), EXPECTED_ORBITS)
    np.testing.assert_array_equal(orbit.circulation(None), EXPECTED_ORBITS)


# ===================================================================
# float32 and N-D coordinates


def _check_float32():
    rad_q = np.array([[1.1, 1.1, 1.1], [1.1, 2.2, 3.3]], dtype=np.float32)
    rad_p = np.array([[0.3, 0.3, 0.3], [0.77, 1.54, 2.31]], dtype=np.float32)
    for name, ((qq, pp), expected, _) in ORBITS.items():
        out = go.circulation(np.asarray(qq, np.float32), np.asarray(pp, np.float32))
        assert out.tolist() == list(expected), name
    # Radial samples, one at a time and as orbits, do not circulate.
    for a, b in zip(rad_q, rad_p, strict=True):
        assert not go.circulation(a[None], b[None]).any()


def test_float32_inputs():
    _check_float32()


def test_float32_without_x64():
    with jax.enable_x64(False):  # noqa: FBT003
        assert jnp.asarray(1.0).dtype == jnp.float32
        _check_float32()


def test_nd_non_cartesian_coordinate():
    """Two batch axes survive the conversion to Cartesian."""
    (q, p), expected, _ = ORBITS["xy"]
    qc = cx.CartesianPos3D.from_(u.Q(q, "kpc"))
    pc = cx.CartesianVel3D.from_(u.Q(p, "kpc/Myr"))
    qs = cx.vconvert(cx.vecs.SphericalPos, qc)
    ps = cx.vconvert(cx.vecs.SphericalVel, pc, qc)
    st = lambda a: qnp.stack([a, a])
    w = gc.PhaseSpaceCoordinate(
        q=cx.vecs.SphericalPos(r=st(qs.r), theta=st(qs.theta), phi=st(qs.phi)),
        p=cx.vecs.SphericalVel(r=st(ps.r), theta=st(ps.theta), phi=st(ps.phi)),
        t=u.Q(T, "Myr"),
    )
    np.testing.assert_array_equal(go.circulation(w), [expected, expected])
    _, axis = go.align_circulation(w)
    np.testing.assert_array_equal(axis, [2, 2])


@pytest.mark.parametrize("rep", [cx.CartesianPos3D, cx.vecs.SphericalPos])
@pytest.mark.parametrize("q_batched", [False, True])
def test_coordinate_q_and_p_with_different_batch_shapes(rep, q_batched):
    """Q of shape (T,) with p of shape (B, T), or the reverse, broadcast."""
    (q, p), expected, _ = ORBITS["xy"]
    q, p = jnp.stack([q, q]) if q_batched else q, p if q_batched else jnp.stack([p, p])
    qv = cx.CartesianPos3D.from_(u.Q(q, "kpc"))
    pv = cx.CartesianVel3D.from_(u.Q(p, "kpc/Myr"))
    qr = cx.vconvert(rep, qv)
    w = gc.PhaseSpaceCoordinate(q=qr, p=pv, t=u.Q(T, "Myr"))
    assert w.q.shape != w.p.shape
    np.testing.assert_array_equal(go.circulation(w), [expected, expected])


def test_float32_p_with_float64_q():
    """The floor follows the coarser of the two dtypes."""
    with jax.enable_x64(True):  # noqa: FBT003
        q = np.array([[1.1, 2.2, 3.3]] * 2, dtype=np.float64)
        p = np.array([[0.77, 1.54, 2.31]] * 2, dtype=np.float32)
        assert not go.circulation(q, p).any()
        assert not go.circulation(q.astype(np.float32), p.astype(np.float64)).any()


@pytest.mark.parametrize("rep", [cx.CartesianPos3D, cx.vecs.SphericalPos])
@pytest.mark.parametrize("q_short", [True, False])
@pytest.mark.parametrize("short", [(1,), ()])
def test_coordinate_q_and_p_with_short_batch_shapes(rep, q_short, short):
    """A (1,) or () vector broadcasts against a (T,) one, both ways."""
    (q, p), _, _ = ORBITS["xy"]
    i = slice(7, 8) if short == (1,) else 7
    qa, pa = (q[i], p) if q_short else (q, p[i])
    qv = cx.CartesianPos3D.from_(u.Q(qa, "kpc"))
    pv = cx.CartesianVel3D.from_(u.Q(pa, "kpc/Myr"))
    w = gc.PhaseSpaceCoordinate(q=cx.vconvert(rep, qv), p=pv, t=u.Q(T, "Myr"))
    assert w.q.shape != w.p.shape
    expected = go.circulation(
        jnp.broadcast_to(qa, (T.size, 3)), jnp.broadcast_to(pa, (T.size, 3))
    )
    np.testing.assert_array_equal(go.circulation(w), expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("name", ["xy", "yz", "box"])
def test_half_precision_matches_float32(name, dtype):
    (q, p), expected, _ = ORBITS[name]
    out = go.circulation(q.astype(dtype), p.astype(dtype))
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(
        out, go.circulation(q.astype(jnp.float32), p.astype(jnp.float32))
    )


def test_half_precision_kepler_matches_float32():
    """A tilted Kepler orbit in AU and km/s, which float16 rounds coarsely."""
    pot = gp.KeplerPotential(m_tot=u.Q(1.0, "Msun"), units="solarsystem")
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([1.0, 0, 0.1], "AU"), p=u.Q([0, 20.0, 1.0], "km/s"), t=u.Q(0.0, "yr")
    )
    o = go.compute_orbit(pot, w0, u.Q(jnp.linspace(0, 5, 2001), "yr"))
    q, p = (np.stack([getattr(v, c).value for c in "xyz"], -1) for v in (o.q, o.p))
    ref = go.circulation(q.astype(np.float32), p.astype(np.float32))
    for dtype in (jnp.float16, jnp.bfloat16):
        np.testing.assert_array_equal(
            go.circulation(q.astype(dtype), p.astype(dtype)), ref
        )


# ===================================================================
# Units


@pytest.mark.parametrize("name", list(ORBITS))
def test_units_do_not_change_the_answer(name):
    (q, p), expected, _ = ORBITS[name]
    for qu, pu in [("kpc", "kpc/Myr"), ("m", "m/s"), ("pc", "km/s")]:
        out = go.circulation(u.Q(q, qu), u.Q(p, pu))
        np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(go.circulation(q, p), expected)


def test_wrong_dimensions_raise():
    (q, p), _, _ = ORBITS["xy"]
    with pytest.raises(Exception, match=r"(?i)unit|dimension|convert"):
        go.circulation(u.Q(q, "deg"), u.Q(p, "kpc/Myr"))
    with pytest.raises(Exception, match=r"(?i)unit|dimension|convert"):
        go.circulation(u.Q(q, "kpc"), u.Q(p, "kpc"))


@pytest.mark.parametrize("q_quantity", [True, False])
def test_align_keeps_the_input_units(q_quantity):
    (q, p), _, _ = ORBITS["yz"]
    qi = u.Q(q, "pc") if q_quantity else q
    pi = p if q_quantity else u.Q(p, "km/s")
    q2, p2, axis = go.align_circulation(qi, pi)
    assert int(axis) == 0
    assert isinstance(q2, u.AbstractQuantity) == q_quantity
    assert isinstance(p2, u.AbstractQuantity) != q_quantity
    if q_quantity:
        assert q2.unit == u.unit("pc")
        np.testing.assert_array_equal(q2.value, q[:, [2, 1, 0]])
    else:
        assert p2.unit == u.unit("km/s")
        np.testing.assert_array_equal(p2.value, p[:, [2, 1, 0]])


# ===================================================================
# Shapes with circ, and per-component units


@pytest.mark.parametrize("shape", [(3,), (5, 4)])
def test_align_with_circ_checks_the_shape(shape):
    x = np.ones(shape)
    with pytest.raises(ValueError, match="shape"):
        go.align_circulation(x, x, np.array([True, False, False]))


def _units_of(v):
    return {c: str(u.unit_of(getattr(v, c))) for c in v.components}


def _cart(v, at=None):
    if at is None:
        return cx.vconvert(cx.CartesianPos3D, v)
    return cx.vconvert(cx.CartesianVel3D, v, at)


def _coordinate(rep):
    """Return a yz-plane orbit in mixed units, in the representation ``rep``."""
    (q, p), _, _ = ORBITS["yz"]
    qv = cx.CartesianPos3D(
        x=u.Q(q[:, 0], "kpc"), y=u.Q(q[:, 1] * 1e3, "pc"), z=u.Q(q[:, 2] * 2e8, "AU")
    )
    pv = cx.CartesianVel3D(
        x=u.Q(p[:, 0], "kpc/Myr"),
        y=u.Q(p[:, 1] * 977.8, "km/s"),
        z=u.Q(p[:, 2], "kpc/Myr"),
    )
    if rep is cx.CartesianPos3D:
        return qv, pv
    qs = cx.vconvert(cx.vecs.SphericalPos, qv)
    ps = cx.vconvert(cx.vecs.SphericalVel, pv, qv)
    return qs, ps


def _check_keeps_units(w, aligned):
    assert type(aligned.q) is type(w.q)
    assert type(aligned.p) is type(w.p)
    assert _units_of(aligned.q) == _units_of(w.q)
    assert _units_of(aligned.p) == _units_of(w.p)


@pytest.mark.parametrize("rep", [cx.CartesianPos3D, cx.vecs.SphericalPos])
def test_align_coordinate_keeps_component_units(rep):
    qv, pv = _coordinate(rep)
    w = gc.PhaseSpaceCoordinate(q=qv, p=pv, t=u.Q(T, "Myr"))
    aligned, axis = go.align_circulation(w)
    assert int(axis) == 0
    _check_keeps_units(w, aligned)
    # The values are those of the Cartesian swap.
    qc, pc = _cart(w.q), _cart(w.p, w.q)
    qa, pa = _cart(aligned.q), _cart(aligned.p, aligned.q)
    for a, b in [(qa.x, qc.z), (qa.y, qc.y), (qa.z, qc.x)]:
        np.testing.assert_allclose(
            u.ustrip("kpc", a), u.ustrip("kpc", b), rtol=1e-5, atol=1e-5
        )
    for a, b in [(pa.x, pc.z), (pa.y, pc.y), (pa.z, pc.x)]:
        np.testing.assert_allclose(
            u.ustrip("kpc/Myr", a), u.ustrip("kpc/Myr", b), rtol=1e-5, atol=1e-5
        )


def test_align_spherical_in_au_deg_mas_per_yr():
    qv, pv = _coordinate(cx.vecs.SphericalPos)
    qv = cx.vecs.SphericalPos(
        r=u.Q(u.ustrip("AU", qv.r), "AU"),
        theta=u.Q(u.ustrip("deg", qv.theta), "deg"),
        phi=u.Q(u.ustrip("deg", qv.phi), "deg"),
    )
    pv = cx.vecs.SphericalVel(
        r=u.Q(u.ustrip("AU/yr", pv.r), "AU/yr"),
        theta=u.Q(u.ustrip("mas/yr", pv.theta), "mas/yr"),
        phi=u.Q(u.ustrip("mas/yr", pv.phi), "mas/yr"),
    )
    w = gc.PhaseSpaceCoordinate(q=qv, p=pv, t=u.Q(T, "Myr"))
    aligned, _ = go.align_circulation(w)
    _check_keeps_units(w, aligned)


def test_align_composite_keeps_component_units():
    qv, pv = _coordinate(cx.CartesianPos3D)
    qs, ps = _coordinate(cx.vecs.SphericalPos)
    h = T.size // 2
    a = gc.PhaseSpaceCoordinate(q=qv[:h], p=pv[:h], t=u.Q(T[:h], "Myr"))
    b = gc.PhaseSpaceCoordinate(q=qs[h:], p=ps[h:], t=u.Q(T[h:], "Myr"))
    comp = gc.CompositePhaseSpaceCoordinate(a=a, b=b)
    aligned, axis = go.align_circulation(comp)
    assert int(axis) == 0
    for k in ("a", "b"):
        _check_keeps_units(comp[k], aligned[k])


def test_align_orbit_keeps_units_and_time(orbit):
    aligned, _ = go.align_circulation(orbit)
    _check_keeps_units(orbit, aligned)
    np.testing.assert_array_equal(aligned.t.value, orbit.t.value)
    assert aligned.t.unit == orbit.t.unit
