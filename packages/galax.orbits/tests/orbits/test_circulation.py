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
