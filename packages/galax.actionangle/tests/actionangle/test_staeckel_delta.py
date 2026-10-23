"""Tests for `staeckel_fudge_delta`."""

import functools as ft

import jax
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from ._helpers import ScaledKuzminKutuzovPotential, SeparablePotential, kk_delta
from .optional_deps import OptDeps
from galax.actionangle import staeckel_fudge_delta


def _kk(m_tot=1e11):
    return gp.KuzminKutuzovPotential(
        m_tot=u.Q(m_tot, "Msun"),
        a=u.Q(3.0, "kpc"),
        c=u.Q(1.0, "kpc"),
        units="galactic",
    )


def _pts(n=50, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(1.0, 9.0, size=(n, 3))
    q[:, 2] *= rng.choice([-1, 1], size=n) * 0.5
    return jnp.asarray(q)


def test_delta_recovers_kk():
    pot = _kk()
    delta = staeckel_fudge_delta(pot, _pts(), jnp.asarray(0.0))
    assert delta.shape == (50,)
    np.testing.assert_allclose(delta, kk_delta(pot), rtol=1e-8)


def test_delta_zero_gradient():
    q = _pts(5)
    t = jnp.asarray(0.0)
    g = jax.grad(lambda x: staeckel_fudge_delta(_kk(), x, t).sum())(q)
    assert jnp.all(g == 0)
    g = jax.grad(lambda m: staeckel_fudge_delta(_kk(m), q, t).sum())(1e11)
    assert g == 0


def test_delta_nan_at_axis_and_plane():
    q = jnp.array([[5.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    delta = staeckel_fudge_delta(_kk(), q, jnp.asarray(0.0))
    assert jnp.all(jnp.isnan(delta))
    g = jax.grad(lambda x: jnp.nansum(staeckel_fudge_delta(_kk(), x, 0.0)))(q)
    assert jnp.all(jnp.isfinite(g))


def test_delta_nan_for_separable_potential():
    """Phi = f(R) + g(z): d2Phi/dRdz == 0 everywhere, so NaN (not inf)."""
    pot = SeparablePotential(units="galactic")
    q = _pts(5)
    t = jnp.asarray(0.0)
    assert jnp.all(jax.vmap(lambda x: pot._gradient(x, t))(q) != 0)
    # the helper is separable: the (x, y)-z Hessian block vanishes exactly
    hess = jax.vmap(lambda x: pot._hessian(x, t))(q)
    assert jnp.all(hess[..., :2, 2] == 0)
    assert jnp.all(jnp.isnan(staeckel_fudge_delta(pot, q, t)))


def test_delta_overloads():
    pot = _kk()
    q = _pts(4)
    t = jnp.asarray(0.0)
    d = staeckel_fudge_delta(pot, q, t)
    assert d.shape == (4,)
    dq = staeckel_fudge_delta(pot, u.Q(q, "kpc"), t)
    assert dq.shape == (4,)
    assert dq.unit == "kpc"
    np.testing.assert_allclose(dq.value, d)
    assert staeckel_fudge_delta(pot, q[0], t).shape == ()
    # broadcasting of q and t
    ts = jnp.zeros(3)
    assert staeckel_fudge_delta(pot, q[:, None], ts).shape == (4, 3)

    v = jnp.tile(jnp.array([0.0, 0.1, 0.0]), (4, 1))
    psc = gc.PhaseSpaceCoordinate(
        q=u.Q(q, "kpc"), p=u.Q(v, "kpc/Myr"), t=u.Q(jnp.zeros(4), "Myr")
    )
    dw = staeckel_fudge_delta(pot, psc)
    assert dw.shape == (4,)
    assert dw.unit == "kpc"
    np.testing.assert_allclose(dw.value, d)

    orbit = gd.evaluate_orbit(pot, psc, u.Q(jnp.linspace(0.0, 10.0, 5), "Myr"))
    do = staeckel_fudge_delta(pot, orbit)
    assert do.shape == (4, 5)
    assert do.unit == "kpc"


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_delta_matches_gala():
    import astropy.units as au
    import gala.dynamics as gala_dyn
    import gala.potential as galap
    from gala.dynamics.actionangle import get_staeckel_fudge_delta
    from gala.units import galactic

    gpot = galap.MiyamotoNagaiPotential(
        m=1e11 * au.Msun, a=3 * au.kpc, b=0.5 * au.kpc, units=galactic
    )
    pot = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.5, "kpc"), units="galactic"
    )
    q = np.asarray(_pts(30, seed=3))
    w = gala_dyn.PhaseSpacePosition(
        pos=q.T * au.kpc, vel=np.ones((3, 30)) * au.kpc / au.Myr
    )
    ref = get_staeckel_fudge_delta(gpot, w, median=False).to_value(au.kpc)
    ours = staeckel_fudge_delta(pot, jnp.asarray(q), jnp.asarray(0.0))
    np.testing.assert_allclose(ours, ref, rtol=1e-8)


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_delta_matches_gala_composite():
    """A composite potential (disk + bulge + nucleus + halo); NaNs must match."""
    import astropy.units as au
    import gala.dynamics as gala_dyn
    import gala.potential as galap
    from gala.dynamics.actionangle import get_staeckel_fudge_delta

    rng = np.random.default_rng(7)
    q = rng.uniform(-15.0, 15.0, size=(500, 3))
    q[:, 2] *= 0.4
    w = gala_dyn.PhaseSpacePosition(
        pos=q.T * au.kpc, vel=np.ones((3, len(q))) * au.kpc / au.Myr
    )
    with np.errstate(invalid="ignore"):  # gala's sqrt of Delta^2 < 0
        ref = get_staeckel_fudge_delta(galap.MilkyWayPotential2022(), w, median=False)
    ours = staeckel_fudge_delta(
        gp.MilkyWayPotential2022(), jnp.asarray(q), jnp.asarray(0.0)
    )
    # assert_allclose's equal_nan also checks that the NaN locations agree.
    np.testing.assert_allclose(ours, ref.to_value(au.kpc), rtol=1e-9)


def _gala_delta(gpot, q):
    import astropy.units as au
    import gala.dynamics as gala_dyn
    from gala.dynamics.actionangle import get_staeckel_fudge_delta

    w = gala_dyn.PhaseSpacePosition(
        pos=q.T * au.kpc, vel=np.ones((3, len(q))) * au.kpc / au.Myr
    )
    with np.errstate(invalid="ignore"):  # gala's sqrt of Delta^2 < 0
        return get_staeckel_fudge_delta(gpot, w, median=False).to_value(au.kpc)


def _all_phi_pts(n=40, seed=11):
    """Points at all azimuths (x, y of both signs), off the plane."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-9.0, 9.0, size=(n, 3))
    q[:, 2] *= 0.4
    return q


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_delta_matches_gala_triaxial():
    """Non-axisymmetric: derivatives at (x, y, z), projected like gala."""
    import astropy.units as au
    import gala.potential as galap
    from gala.units import galactic

    gpot = galap.NFWPotential(
        m=1e12 * au.Msun, r_s=20 * au.kpc, a=1, b=0.8, c=0.6, units=galactic
    )
    # gala flattens the potential (gp.TriaxialNFWPotential flattens the density).
    nfw = gp.NFWPotential(m=u.Q(1e12, "Msun"), r_s=u.Q(20.0, "kpc"), units="galactic")
    pot = gp.TriaxialInThePotential(nfw, y_over_x=0.8, z_over_x=0.6)
    q = _all_phi_pts()
    ours = staeckel_fudge_delta(pot, jnp.asarray(q), jnp.asarray(0.0))
    np.testing.assert_allclose(ours, _gala_delta(gpot, q), rtol=1e-12)


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
def test_delta_matches_gala_translated():
    """An off-centre potential (galax translation vs gala ``origin``).

    Eq. 9 uses the origin's cylindrical coordinates, so Delta is not
    translation invariant: it is not Delta of the centred potential at x - d.
    The shift has d_z != 0, so Phi is not symmetric about z = 0 and
    d2Phi/dRdz != 0 there: the z = 0 points are finite, as in gala.
    """
    import astropy.units as au
    import gala.potential as galap
    from gala.units import galactic

    d = [1.0, -0.5, 0.3]
    gpot = galap.MiyamotoNagaiPotential(
        m=1e11 * au.Msun,
        a=3 * au.kpc,
        b=0.5 * au.kpc,
        units=galactic,
        origin=d * au.kpc,
    )
    mn = gp.MiyamotoNagaiPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), b=u.Q(0.5, "kpc"), units="galactic"
    )
    pot = gp.TranslatedPotential(mn, translation=u.Q(d, "kpc"))
    q = _all_phi_pts()
    q0 = q * np.array([1.0, 1.0, 0.0])  # the same points on the plane z = 0
    q = np.asarray(jnp.concat([q, q0, jnp.asarray([[8.0, 0, 0], [5.0, 3, 0]])]))
    ours = staeckel_fudge_delta(pot, jnp.asarray(q), jnp.asarray(0.0))
    ref = _gala_delta(gpot, q)
    assert np.all(np.isfinite(ref[-2:]))  # gala: 2.6379, 3.4356 kpc
    np.testing.assert_allclose(ours, ref, rtol=1e-12)


PC = u.unitsystem("pc", "Myr", "Msun", "radian")
_OFF_AXIS = np.array(
    [[300.0, 0, 200], [1000.0, 300, 700], [1.0, 0.5, 0.3], [8.0, 2, 1]]
)


@ft.cache
def _undeclared(cls):
    """``cls`` without its `Symmetry.SPHERICAL` declaration.

    Delta then goes through eq. 9 instead of the exact-zero shortcut: the tests
    of eq. 9's round-off threshold, and of the spherical built-ins' Hessians,
    use these.
    """
    assert cls.symmetry is gp.Symmetry.SPHERICAL
    return type(f"Undeclared{cls.__name__}", (cls,), {"symmetry": gp.Symmetry.NONE})


def _sph(kind, units, scale, *, declared=False):
    m, r = u.Q(1e12, "Msun"), u.Q(scale, "kpc")
    cls = {
        "hernquist": gp.HernquistPotential,
        "nfw": gp.NFWPotential,
        "isochrone": gp.IsochronePotential,
    }[kind]
    cls = cls if declared else _undeclared(cls)
    if kind == "nfw":
        return cls(m=m, r_s=r, units=units)
    return cls(m_tot=m, r_s=r, units=units)


def _composite(*, declared):
    return gp.CompositePotential(
        a=_sph("hernquist", "galactic", 3.0, declared=declared),
        b=_sph("nfw", "galactic", 20.0, declared=declared),
    )


# Off-axis points, then the plane, the axis and the origin (NaN through eq. 9,
# 0/0), then the NFW core at 1e-5 r_s, where its Hessian round-off makes the
# eq. 9 Delta spurious.
_SPH_PTS = jnp.concat(
    [
        jnp.asarray(_OFF_AXIS),
        jnp.asarray(
            [[5.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [8e-5, 3e-5, 5e-5]]
        ),
    ]
)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(ft.partial(_sph, k, "galactic", 10.0), id=k)
        for k in ("hernquist", "isochrone", "nfw")
    ]
    + [pytest.param(_composite, id="composite")],
)
def test_delta_exactly_zero_for_declared_spherical(make):
    """Exactly 0 everywhere, even where eq. 9 gives NaN or round-off.

    A composite of spherical parts derives `SPHERICAL`, so it is 0 too. The
    zeros take the broadcast batch shape of ``q`` and ``t``, and eq. 9's dtype.
    """
    pot = make(declared=True)
    assert pot.symmetry is gp.Symmetry.SPHERICAL
    q = jnp.asarray(_SPH_PTS)[:, None]
    delta = staeckel_fudge_delta(pot, q, jnp.asarray([0.0, 1.0]))
    assert delta.shape == (len(_SPH_PTS), 2)
    assert jnp.all(delta == 0)
    # The same potential undeclared is NaN at the plane, axis and origin.
    general = staeckel_fudge_delta(make(declared=False), q, jnp.asarray([0.0, 1.0]))
    assert jnp.all(jnp.isnan(general[len(_OFF_AXIS) : len(_OFF_AXIS) + 3]))
    assert delta.dtype == general.dtype


def test_delta_exactly_zero_skips_the_hessian():
    """No derivative of the potential is traced for a declared-spherical one.

    The NFW potential's log(1 + r / r_s) marks it in the jaxpr: present on the
    general path, absent on the declared-spherical one.
    """

    def jaxpr(pot):
        return str(
            jax.make_jaxpr(lambda q: staeckel_fudge_delta(pot, q, 0.0))(
                jnp.ones((3, 3))
            )
        )

    assert "log" in jaxpr(_sph("nfw", "galactic", 10.0))
    assert "log" not in jaxpr(_sph("nfw", "galactic", 10.0, declared=True))


@pytest.mark.parametrize("units", ["galactic", PC], ids=["galactic", "pc"])
@pytest.mark.parametrize(
    ("kind", "rmin"),
    [("hernquist", 0.1), ("isochrone", 1e-5), ("nfw", 1e-8)],
)
def test_delta_scale_free_spherical_zero(kind, rmin, units):
    # Radii in units of r_s, so both unit systems probe the same range.
    pot = _sph(kind, units, 10.0)
    r_s = float(u.ustrip(pot.units["length"], u.Q(10.0, "kpc")))
    r = np.geomspace(rmin, 100.0, 40) * r_s
    q = jnp.asarray(np.stack([r * 0.8, r * 0.3, r * 0.5], axis=-1))
    delta = staeckel_fudge_delta(pot, q, jnp.asarray(0.0))
    assert jnp.all(delta == 0)


def test_delta_threshold_zeroes_tiny_kk_delta():
    """Pins the threshold from below: this Delta would survive 1e2 eps.

    c = a (1 - 2.5e-13): the analytic Delta^2 / (eps S) is ~296, 316, 275 and
    306 at these points, under 1e3 and over 1e2. Unlike round-off noise, it
    does not depend on the evaluation order (eager vs jit).
    """
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"),
        a=u.Q(3.0, "kpc"),
        c=u.Q(3.0 * (1 - 2.5e-13), "kpc"),
        units="galactic",
    )
    q = jnp.asarray([[3.0, 1, 2], [2, 2, 2], [1, 3, 2.5], [3, 0, 2]])
    assert jnp.all(staeckel_fudge_delta(pot, q, jnp.asarray(0.0)) == 0)


@pytest.mark.parametrize("declared", [False, True], ids=["eq9", "declared"])
@pytest.mark.parametrize("kind", ["hernquist", "isochrone", "nfw"])
def test_delta_spherical_zero_float32_inputs(kind, declared):
    """Float32 q and t under x64: the derivatives, and Delta, are float32."""
    pot = _sph(kind, "galactic", 10.0, declared=declared)
    r = jnp.geomspace(1.0, 1000.0, 40, dtype=jnp.float32)
    q = jnp.stack([r * 0.8, r * 0.3, r * 0.5], axis=-1)
    delta = staeckel_fudge_delta(pot, q, jnp.asarray(0.0, dtype=jnp.float32))
    assert delta.dtype == jnp.float32
    assert jnp.all(delta == 0)


# The potentials' constants (G) are built under the suite's x64 at import, so
# a float32 potential truncates them.
_F32 = pytest.mark.filterwarnings(
    "ignore:Explicitly requested dtype float64:UserWarning"
)


@_F32
@pytest.mark.parametrize("declared", [False, True], ids=["eq9", "declared"])
@pytest.mark.parametrize("kind", ["hernquist", "isochrone", "nfw"])
def test_delta_spherical_zero_float32(kind, declared):
    """The spherical threshold scales with the working precision.

    Here with x64 off, the default a caller gets; the dtype check keeps the
    test honest.
    """
    with jax.enable_x64(False):  # noqa: FBT003
        pot = _sph(kind, "galactic", 10.0, declared=declared)
        r = jnp.geomspace(1.0, 1000.0, 40)
        q = jnp.stack([r * 0.8, r * 0.3, r * 0.5], axis=-1)
        delta = staeckel_fudge_delta(pot, q, jnp.asarray(0.0))
        assert delta.dtype == jnp.float32
        assert jnp.all(delta == 0)


@_F32
def test_delta_kk_float32():
    """The float32 threshold does not zero a genuine (KK) Delta."""
    with jax.enable_x64(False):  # noqa: FBT003
        pot = _kk()
        delta = staeckel_fudge_delta(pot, _pts().astype(jnp.float32), jnp.asarray(0.0))
        assert delta.dtype == jnp.float32
        expect = kk_delta(pot)
        # r ~ 100 kpc: Delta^2 / (eps S) ~ 2.4e3 to 2.9e3, above the 1e3 eps
        # threshold (a 1e4 one zeroes it); error ~ eps S / (2 Delta^2) ~ 2e-4.
        far = jnp.asarray(
            [[80.0, 0, 60], [99.0, 0, 14], [50.0, 50, 70]], dtype=jnp.float32
        )
        delta_far = staeckel_fudge_delta(pot, far, jnp.asarray(0.0))
    np.testing.assert_allclose(delta, expect, rtol=1e-4)
    np.testing.assert_allclose(delta_far, expect, rtol=1e-3)


@_F32
def test_delta_float32_si():
    """Delta is scale-free: float32 with SI lengths, where R^2 ~ 1e40 overflows.

    1e6 Msun, since a galaxy's mass in kg (> 3.4e38) is itself past float32.
    """
    with jax.enable_x64(False):  # noqa: FBT003
        pot = ScaledKuzminKutuzovPotential(
            m_tot=u.Q(1e6, "Msun"), a=u.Q(3.0, "kpc"), c=u.Q(1.0, "kpc"), units="si"
        )
        q = u.Q(_pts().astype(jnp.float32), "kpc")
        delta = staeckel_fudge_delta(pot, q, u.Q(jnp.float32(0), "s"))
        assert delta.value.dtype == jnp.float32
        expect = np.sqrt(8.0)
    np.testing.assert_allclose(delta.ustrip("kpc"), expect, rtol=1e-4)


class _NotAGradient(ScaledKuzminKutuzovPotential):
    """A `_gradient` that is not a gradient: its Jacobian is not symmetric."""

    def _gradient(self, xyz, t, /):
        g = super()._gradient(xyz, t)
        size = jnp.linalg.vector_norm(g) / jnp.linalg.vector_norm(xyz)
        return g + 1e-4 * size * jnp.stack([jnp.zeros_like(xyz[0]), xyz[0], xyz[2]])


def test_delta_nan_for_inconsistent_second_derivatives():
    """Delta is NaN, not finite and wrong, where d2Phi/dRdz != d2Phi/dzdR.

    That is the sign of an untrustworthy Hessian, e.g. a potential evaluated
    near underflow, whose autodiff intermediates go subnormal.
    """
    pot = _NotAGradient(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), c=u.Q(1.0, "kpc"), units="galactic"
    )
    delta = staeckel_fudge_delta(pot, _pts(), jnp.asarray(0.0))
    assert jnp.all(jnp.isnan(delta))


def _every_spherical(r_s):
    m = u.Q(1e12, "Msun")
    kw = {"units": "galactic"}
    return {
        "Kepler": _undeclared(gp.KeplerPotential)(m_tot=m, **kw),
        "Plummer": _undeclared(gp.PlummerPotential)(m_tot=m, r_s=r_s, **kw),
        "Hernquist": _undeclared(gp.HernquistPotential)(m_tot=m, r_s=r_s, **kw),
        "Jaffe": _undeclared(gp.JaffePotential)(m_tot=m, r_s=r_s, **kw),
        "Isochrone": _undeclared(gp.IsochronePotential)(m_tot=m, r_s=r_s, **kw),
        "NFW": _undeclared(gp.NFWPotential)(m=m, r_s=r_s, **kw),
        "HardCutoffNFW": _undeclared(gp.HardCutoffNFWPotential)(
            m=m, r_s=r_s, r_t=12.3 * r_s, **kw
        ),
        "Burkert": _undeclared(gp.BurkertPotential)(m=m, r_s=r_s, **kw),
        "Gaussian": _undeclared(gp.GaussianPotential)(m_tot=m, r_s=r_s, **kw),
        "gNFW0.5": _undeclared(gp.gNFWPotential)(m=m, r_s=r_s, gamma=0.5, **kw),
        "gNFW1.5": _undeclared(gp.gNFWPotential)(m=m, r_s=r_s, gamma=1.5, **kw),
        "Zhao": _undeclared(gp.ZhaoPotential)(
            m=m, r_s=r_s, alpha=0.5, beta=4.0, gamma=1.0, **kw
        ),
        "PowerLawCutoff": _undeclared(gp.PowerLawCutoffPotential)(
            m_tot=m, alpha=1.8, r_c=r_s, **kw
        ),
        "StoneOstriker15": _undeclared(gp.StoneOstriker15Potential)(
            m_tot=m, r_c=0.1 * r_s, r_h=r_s, **kw
        ),
        "Logarithmic": _undeclared(gp.LogarithmicPotential)(
            v_c=u.Q(200, "km/s"), r_s=r_s, **kw
        ),
    }


@pytest.mark.parametrize("name", _every_spherical(u.Q(10.0, "kpc")))
def test_delta_zero_for_every_spherical_builtin(name):
    """Delta == 0, exactly, from 1e-8 to 1e8 scale lengths in 64 directions.

    Through eq. 9: each potential has its `SPHERICAL` declaration stripped
    (`_undeclared`, which also checks every one declares it), so the shortcut
    does not hide the Hessian.

    This needs each potential's Hessian to a few ulp of its magnitude: NFW and
    Burkert used to give a spurious Delta in the core, and Burkert and gNFW
    far out.
    """
    pot = _every_spherical(u.Q(10.0, "kpc"))[name]
    rng = np.random.default_rng(0)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 10.0 * np.geomspace(1e-8, 1e8, 33)
    q = jnp.asarray((r[:, None, None] * d[None]).reshape(-1, 3))
    delta = staeckel_fudge_delta(pot, q, jnp.asarray(0.0))
    assert jnp.all(delta == 0), int(jnp.sum(delta != 0))


def test_delta_output_unit_is_potential_length():
    """Both overloads return Delta in the potential's length unit (here pc)."""
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"), a=u.Q(3.0, "kpc"), c=u.Q(1.0, "kpc"), units=PC
    )
    expect = kk_delta(pot)  # sqrt(8) kpc, in pc
    np.testing.assert_allclose(expect, np.sqrt(8.0) * 1e3, rtol=1e-12)
    q = u.Q(_pts(4), "kpc")
    dq = staeckel_fudge_delta(pot, q, u.Q(0.0, "Myr"))
    assert dq.unit == "pc"
    np.testing.assert_allclose(dq.value, expect, rtol=1e-8)

    psc = gc.PhaseSpaceCoordinate(
        q=q, p=u.Q(jnp.zeros((4, 3)), "km/s"), t=u.Q(jnp.zeros(4), "Myr")
    )
    dw = staeckel_fudge_delta(pot, psc)
    assert dw.unit == "pc"
    np.testing.assert_allclose(dw.value, expect, rtol=1e-8)


def test_delta_time_dependent():
    """Delta is evaluated at ``t``: KK with a(t) = 3 kpc + 0.1 kpc/Myr t."""
    a = gp.params.LinearParameter(
        slope=u.Q(0.1, "kpc/Myr"),
        point_time=u.Q(0.0, "Myr"),
        point_value=u.Q(3.0, "kpc"),
    )
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"), a=a, c=u.Q(1.0, "kpc"), units="galactic"
    )
    q = _pts(5)
    ts = jnp.asarray([0.0, 10.0])  # a = 3, 4 kpc
    delta = staeckel_fudge_delta(pot, q[:, None], ts)
    assert delta.shape == (5, 2)
    expect = jnp.asarray([kk_delta(pot, 0.0), kk_delta(pot, 10.0)])
    np.testing.assert_allclose(expect, np.sqrt([8.0, 15.0]), rtol=1e-12)
    np.testing.assert_allclose(delta, jnp.broadcast_to(expect, (5, 2)), rtol=1e-8)


def test_delta_near_spherical_kk_recovered():
    """A genuine Delta well below the cancelling terms is kept.

    c = a (1 - 1e-9): Delta^2 / S ~ 3.5e-11 .. 1.8e-10 over the points, far
    above the threshold 1e3 eps ~ 2.2e-13. The error is ~eps S / Delta^2.
    """
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"),
        a=u.Q(3.0, "kpc"),
        c=u.Q(3.0 * (1 - 1e-9), "kpc"),
        units="galactic",
    )
    delta = staeckel_fudge_delta(pot, _pts(), jnp.asarray(0.0))
    assert jnp.all(delta != 0)
    np.testing.assert_allclose(delta, kk_delta(pot), rtol=5e-6)


def test_delta_threshold_below_near_spherical_kk():
    """Pins the threshold from above: 1e4 eps would zero this genuine Delta.

    c = a (1 - 2e-11): Delta^2 / (eps S) ~ 3.2e3 .. 1.7e4 over the points,
    above 1e3 and partly below 1e4. Error ~ eps S / (2 Delta^2) <~ 1.6e-4.
    """
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"),
        a=u.Q(3.0, "kpc"),
        c=u.Q(3.0 * (1 - 2e-11), "kpc"),
        units="galactic",
    )
    delta = staeckel_fudge_delta(pot, _pts(), jnp.asarray(0.0))
    assert jnp.all(delta != 0)
    np.testing.assert_allclose(delta, kk_delta(pot), rtol=5e-4)


def test_delta_psc_time_dependent():
    """The (pot, w) overload evaluates at w.t: a(t) = 3, 4 kpc at t = 0, 10 Myr."""
    a = gp.params.LinearParameter(
        slope=u.Q(0.1, "kpc/Myr"),
        point_time=u.Q(0.0, "Myr"),
        point_value=u.Q(3.0, "kpc"),
    )
    pot = gp.KuzminKutuzovPotential(
        m_tot=u.Q(1e11, "Msun"), a=a, c=u.Q(1.0, "kpc"), units="galactic"
    )
    q = jnp.broadcast_to(_pts(5)[:, None], (5, 2, 3))
    psc = gc.PhaseSpaceCoordinate(
        q=u.Q(q, "kpc"),
        p=u.Q(jnp.zeros((5, 2, 3)), "km/s"),
        t=u.Q(jnp.asarray([0.0, 10.0]), "Myr"),
    )
    delta = staeckel_fudge_delta(pot, psc)
    assert delta.shape == (5, 2)
    expect = jnp.broadcast_to(jnp.sqrt(jnp.asarray([8.0, 15.0])), (5, 2))
    np.testing.assert_allclose(delta.value, expect, rtol=1e-8)


def test_delta_integer_q():
    """Integer positions are promoted to float, like an integer ``t``."""
    pot = _kk()
    q = jnp.asarray([[3, 1, 2], [5, -2, 1]])
    expect = staeckel_fudge_delta(pot, q.astype(float), jnp.asarray(0.0))
    np.testing.assert_array_equal(staeckel_fudge_delta(pot, q, 0), expect)
    np.testing.assert_allclose(expect, kk_delta(pot), rtol=1e-8)


def test_delta_jitted_no_retrace():
    """One trace across t types/units and potential parameter values."""
    from galax.actionangle._src.staeckel import _delta

    # `_cache_size` is private jax API: skip rather than fail if it goes away
    # (probed on a fresh jit, so an un-jitted `_delta` still fails).
    if not hasattr(jax.jit(lambda x: x), "_cache_size"):
        pytest.skip("jax no longer exposes the jit cache size")
    assert hasattr(_delta, "_cache_size"), "_delta must be jitted"

    q = _pts(6)
    staeckel_fudge_delta(_kk(), q, jnp.asarray(0.0))
    n = _delta._cache_size()
    staeckel_fudge_delta(_kk(), q + 0.1, 1.0)
    staeckel_fudge_delta(_kk(), q, 2)  # int t
    staeckel_fudge_delta(_kk(), q, np.float64(2.0))  # strong
    staeckel_fudge_delta(_kk(2e11), u.Q(q, "kpc"), u.Q(0.5, "Gyr"))
    staeckel_fudge_delta(_kk(), jnp.round(q).astype(int), 0.0)  # int q
    assert _delta._cache_size() == n


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_delta_half_precision_promoted_to_float32(dtype):
    """Half-precision inputs are computed in float32, not zeroed by the tol."""
    pot = _kk()
    q = _pts(4).astype(dtype)
    t = jnp.zeros((), dtype)
    expect = staeckel_fudge_delta(pot, q.astype("float32"), t.astype("float32"))
    assert jnp.all(expect > 0.1)
    delta = staeckel_fudge_delta(pot, q, t)
    assert delta.dtype == jnp.float32
    np.testing.assert_array_equal(delta, expect)

    psc = gc.PhaseSpaceCoordinate(
        q=u.Q(q, "kpc"), p=u.Q(jnp.zeros((4, 3), dtype), "kpc/Myr"), t=u.Q(t, "Myr")
    )
    dw = staeckel_fudge_delta(pot, psc)
    assert dw.dtype == jnp.float32
    np.testing.assert_array_equal(dw.value, expect)


@pytest.mark.parametrize("which", ["q", "t"])
def test_delta_rejects_complex(which):
    q, t = _pts(2), jnp.asarray(0.0)
    if which == "q":
        q = q.astype(complex)
    else:
        t = t.astype(complex)
    with pytest.raises(TypeError, match="complex"):
        staeckel_fudge_delta(_kk(), q, t)


def test_delta_psc_rejects_extra_t():
    """``w`` carries its own times; a stray ``t`` is a clear TypeError."""
    psc = gc.PhaseSpaceCoordinate(
        q=u.Q(_pts(2), "kpc"), p=u.Q(jnp.zeros((2, 3)), "km/s"), t=u.Q(0.0, "Myr")
    )
    for t in (0.0, u.Q(0.0, "Myr")):
        with pytest.raises(TypeError, match="already carries its times"):
            staeckel_fudge_delta(_kk(), psc, t)
