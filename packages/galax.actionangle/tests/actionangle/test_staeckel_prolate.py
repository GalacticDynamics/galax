"""Tests for the Staeckel-fudge prolate core: coordinates, integrals, F(tau)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

from ._helpers import (
    AXIS_W6,
    D2,
    _kk,
    _mn,
    _phi,
    _setup,
    _w6s,
    check_dF_limit_sites,
)
from galax.actionangle._src._staeckel_prolate import (
    LOG_TAU_MAX,
    integrals_of_motion,
    make_F,
    prolsph,
)


def _Rz_grid():
    R = np.array([0.5, 1.0, 2.0, 2.0, 5.0, 0.7, 9.0, 3.0, 0.1, 1.5])
    z = np.array([0.3, -2.0, 0.01, -4.0, 6.0, 2.5, -0.2, 1e-6, 1.0, -2.9])
    return jnp.asarray(R), jnp.asarray(z)


def test_prolsph_roundtrip():
    R, z = _Rz_grid()
    p = jax.vmap(prolsph, (0, 0, None))(R, z, D2)
    np.testing.assert_allclose(p.lam * p.absnu / D2, z**2, rtol=1e-14, atol=0)
    # lam - D2 is carried as lmd: a direct subtraction would lose up to
    # ulp(lam)/lmd relative precision near the axis inside the foci.
    np.testing.assert_array_less(np.abs(p.lam - D2 - p.lmd), np.spacing(p.lam) + 1e-300)
    np.testing.assert_allclose(p.lmd * (1 - p.absnu / D2), R**2, rtol=1e-14, atol=0)
    np.testing.assert_allclose(p.rho**2, p.lmd, rtol=1e-14, atol=0)
    np.testing.assert_allclose(p.sigma**2, p.absnu, rtol=1e-14, atol=0)
    np.testing.assert_allclose(p.lmd * p.dmn / D2, R**2, rtol=1e-14, atol=0)
    assert np.all(p.lam >= D2)
    assert np.all((p.absnu >= 0) & (p.absnu <= D2))


def test_sigma_sign_and_smoothness():
    R, z = _Rz_grid()
    p = jax.vmap(prolsph, (0, 0, None))(R, z, D2)
    np.testing.assert_array_equal(np.sign(p.sigma), np.sign(z))
    assert prolsph(jnp.asarray(2.0), jnp.asarray(0.0), D2).sigma == 0

    for Ri in (0.0, 0.5, 2.0, 10.0):
        for jac in (jax.jacfwd, jax.jacrev):
            dsdz = jac(lambda z_, R_=Ri: prolsph(jnp.asarray(R_), z_, D2).sigma)(
                jnp.asarray(0.0)
            )
            assert np.isfinite(dsdz)
            assert dsdz > 0
    # continuous across the |nu| = D2/2 branch switch
    zs = jnp.linspace(1.9, 2.1, 2001)
    sig = jax.vmap(lambda z_: prolsph(jnp.asarray(0.0), z_, D2).sigma)(zs)
    assert np.max(np.abs(np.diff(sig))) < 2 * (zs[1] - zs[0])


@pytest.mark.parametrize("jac", [jax.jacfwd, jax.jacrev])
@pytest.mark.parametrize(
    "Rz", [(2.0, 0.0), (0.0, 1.0), (0.0, -2.0), (0.0, 0.0), (0.0, 5.0), (1e-3, 0.0)]
)
def test_prolsph_grad_nan_free_axis_plane(jac, Rz):
    J = jac(lambda Rz_: prolsph(Rz_[0], Rz_[1], D2))(jnp.asarray(Rz))
    for leaf in jax.tree.leaves(J):
        assert np.all(np.isfinite(leaf)), J


def _cpp_reference(phi, w6, D2):
    """Independent numpy port of gala #649's `staeckel_fudge_single` setup."""
    x, y, z, vx, vy, vz = np.asarray(w6, dtype=float)
    R = np.hypot(x, y)
    Lz = x * vy - y * vx
    vR = (x * vx + y * vy) / R
    vphi = Lz / R
    R2, z2 = R * R, z * z
    signz = 1.0 if z >= 0 else -1.0
    sm, dif = R2 + z2 + D2, R2 + z2 - D2
    sqD = np.sqrt(dif**2 + 4 * R2 * D2)
    if dif >= 0:
        lmd = 0.5 * (sqD + dif)
        dmn = D2 * R2 / lmd
    else:
        dmn = 0.5 * (sqD - dif)
        lmd = D2 * R2 / dmn
    absnu = 2 * D2 / (sm + sqD) * z2
    if absnu * 2 > D2:
        absnu = D2 - dmn
    else:
        dmn = D2 - absnu
    lam = D2 + lmd
    lamdot = (R * 2 * lam * vR + z * 2 * lmd * vz) / sqD
    nudot = (-R * 2 * absnu * vR + z * 2 * dmn * vz) / sqD * signz

    def Phi(R_, z_):
        return float(phi(jnp.asarray(R_), jnp.asarray(z_)))

    P = Phi(R, abs(z))
    P0 = Phi(np.sqrt(lam - D2), 0.0)
    E = P + 0.5 * (vx**2 + vy**2 + vz**2)
    flam = -lam * P0
    fnu = flam + P * (lam - absnu)
    I3 = lam * (P - P0) + 0.5 * ((z * vphi) ** 2 + (R * vz - z * vR) ** 2 + vz**2 * D2)

    def f_tau(tau):  # C++ F_tau, value only
        if tau >= D2:
            lam_, nu_, I = tau, absnu, I3 - fnu
            mult = lam_ - nu_
        else:
            lam_, nu_, I = lam, tau, I3 - flam
            mult = nu_ - lam_
        Rt = np.sqrt(max(0.0, (lam_ - D2) * (1 - nu_ / D2)))
        zt = np.sqrt(lam_ * nu_ / D2)
        return (E * tau - I - Phi(Rt, zt) * mult) * (tau - D2) - Lz**2 / 2 * tau

    ref = {"E": E, "Lz": Lz, "I3": I3, "flam": flam, "fnu": fnu, "lam": lam}
    ref |= {"absnu": absnu, "lamdot": lamdot, "nudot": nudot}
    return ref, f_tau


@pytest.mark.parametrize("pot", [_kk(), _mn()], ids=["kk", "mn"])
def test_F_vanishes_consistency(pot):
    phi = _phi(pot)
    iom_fn = jax.jit(lambda w: integrals_of_motion(phi, w, D2))
    for w6 in _w6s(12):
        iom = iom_fn(w6)
        F = make_F(phi, iom)
        ref, f_tau = _cpp_reference(phi, w6, D2)
        for k in ("E", "Lz", "I3", "flam", "fnu", "lam", "absnu", "lamdot"):
            np.testing.assert_allclose(getattr(iom, k), ref[k], rtol=1e-12, err_msg=k)
        # sigma-dot = nudot / (2 sqrt|nu|)
        np.testing.assert_allclose(
            iom.sigdot, ref["nudot"] / (2 * np.sqrt(ref["absnu"])), rtol=1e-12
        )

        lam, nu = ref["lam"], ref["absnu"]
        # F = (E tau - I - Phi mult)(tau - D2) - Lz^2 tau/2 is an O(1)-O(1)
        # difference; compare relative to the size of its terms.
        scale = abs(ref["E"]) * lam * (lam - D2) + ref["Lz"] ** 2 * lam
        np.testing.assert_allclose(
            F.F_lam(iom.lam), f_tau(lam), rtol=0, atol=1e-12 * scale
        )
        scale_n = abs(ref["E"]) * nu * D2 + ref["Lz"] ** 2 * nu + abs(ref["I3"]) * D2
        np.testing.assert_allclose(
            F.F_nu(iom.absnu), f_tau(nu), rtol=0, atol=1e-12 * scale_n
        )
        # The momenta at the phase point: F(lam) = (lam-|nu|)^2 lamdot^2/(8 lam),
        # F(|nu|) = (lam-|nu|)^2 sigdot^2/2.
        np.testing.assert_allclose(
            F.F_lam(iom.lam),
            (lam - nu) ** 2 * ref["lamdot"] ** 2 / (8 * lam),
            rtol=0,
            atol=1e-12 * scale,
        )
        np.testing.assert_allclose(
            F.F_nu(iom.absnu),
            (lam - nu) ** 2 * iom.sigdot**2 / 2,
            rtol=0,
            atol=1e-12 * scale_n,
        )
        # Endpoint values used analytically by the solver.
        np.testing.assert_allclose(
            F.F_lam(jnp.asarray(D2)), -(ref["Lz"] ** 2) * D2 / 2, rtol=1e-13
        )
        np.testing.assert_allclose(
            F.F_nu(jnp.asarray(D2)), -(ref["Lz"] ** 2) * D2 / 2, rtol=1e-13
        )
        # F_nu_u is F_nu through the signed root, and even in u.
        uu = jnp.sqrt(iom.absnu)
        np.testing.assert_allclose(F.F_nu_u(uu), F.F_nu(iom.absnu), rtol=1e-14)
        np.testing.assert_allclose(F.F_nu_u(-uu), F.F_nu_u(uu), rtol=1e-12)


def test_F_semi_no_overflow():
    _, iom, F = _setup()
    lam_pos = iom.lam
    for phi_inf in (0.0, 0.0123):
        target = iom.E - phi_inf
        for s in (0.9975, 0.99999, 1.0):
            v = F.F_semi(jnp.asarray(s), lam_pos, jnp.asarray(phi_inf))
            assert np.isfinite(v)
            np.testing.assert_array_equal(v, target)
    # Just below the guard F/tau^2 is finite and close to E - Phi_inf = E (KK).
    s_g = LOG_TAU_MAX - np.log(float(lam_pos)) - 1.0
    s_g = s_g / (1 + s_g)
    for s in (0.99, s_g):
        v = F.F_semi(jnp.asarray(s), lam_pos, jnp.asarray(0.0))
        assert np.isfinite(v)
        np.testing.assert_allclose(v, iom.E, rtol=1e-12)
    # Where tau^2 does not overflow it equals F_lam(tau)/tau^2.
    for s in (0.0, 0.3, 0.7, 0.9):
        tau = lam_pos * np.exp(s / (1 - s))
        np.testing.assert_allclose(
            F.F_semi(jnp.asarray(s), lam_pos, jnp.asarray(0.0)),
            F.F_lam(tau) / tau**2,
            rtol=1e-12,
        )


def test_dF_limits_and_fd():
    phi, iom, F = _setup()
    E, Lz = float(iom.E), float(iom.Lz)
    lam, nu = float(iom.lam), float(iom.absnu)

    # lambda side, tau = D2
    lim_l = (
        E * D2
        - Lz**2 / 2
        - (iom.I3 - iom.fnu)
        - (D2 - nu) * phi(jnp.asarray(0.0), iom.sigma)
    )
    np.testing.assert_allclose(F.dF_lam(jnp.asarray(D2)), lim_l, rtol=1e-14)
    np.testing.assert_allclose(F.dF_lam(jnp.asarray(D2 * (1 + 1e-9))), lim_l, rtol=1e-6)
    assert np.isnan(F.d2F_lam(jnp.asarray(D2)))  # as in the C++ (0/0)
    # nu side, tau = D2
    lim_n = (
        E * D2
        - Lz**2 / 2
        - (iom.I3 - iom.flam)
        - (D2 - lam) * phi(jnp.asarray(0.0), jnp.sqrt(iom.lam))
    )
    np.testing.assert_allclose(F.dF_nu(jnp.asarray(D2)), lim_n, rtol=1e-14)
    np.testing.assert_allclose(F.dF_nu(jnp.asarray(D2 * (1 - 1e-9))), lim_n, rtol=1e-6)
    # u = 0: F'(0) = F_uu(0)/2, by central differences of the even F(u)
    h = 1e-4
    Fuu0 = (F.F_nu_u(h) - 2 * F.F_nu_u(0.0) + F.F_nu_u(-h)) / h**2
    np.testing.assert_allclose(F.dF_nu(jnp.asarray(0.0)), Fuu0 / 2, rtol=1e-6)
    np.testing.assert_allclose(F.dF_nu(jnp.asarray(1e-12)), Fuu0 / 2, rtol=1e-6)

    # generic points against central differences
    for tau in (1.3 * lam, 2.0 * D2, 40.0):
        hh = 1e-5 * tau
        fd1 = (F.F_lam(tau + hh) - F.F_lam(tau - hh)) / (2 * hh)
        np.testing.assert_allclose(F.dF_lam(jnp.asarray(tau)), fd1, rtol=1e-7)
        fd2 = (F.dF_lam(tau + hh) - F.dF_lam(tau - hh)) / (2 * hh)
        np.testing.assert_allclose(F.d2F_lam(jnp.asarray(tau)), fd2, rtol=1e-6)
    for tau in (0.5 * nu, 0.3 * D2, 0.9 * D2):
        hh = 1e-5 * tau
        fd1 = (F.F_nu(tau + hh) - F.F_nu(tau - hh)) / (2 * hh)
        np.testing.assert_allclose(F.dF_nu(jnp.asarray(tau)), fd1, rtol=1e-7)


@pytest.mark.parametrize("name", ["kk", "mn", "nfw", "mw2022"])
def test_dF_limit_sites_grad_nan_free(name):
    """See `check_dF_limit_sites`."""
    check_dF_limit_sites(name)


@pytest.mark.parametrize("pot", [_kk(), _mn()], ids=["kk", "mn"])
def test_I3_grad_on_axis_matches_fd(pot):
    """dI3/dw at R = 0: AD (jacrev) against central differences."""
    phi = _phi(pot)

    def I3(w):
        return integrals_of_motion(phi, w, D2).I3

    g = jax.jacrev(I3)(AXIS_W6)
    h = 1e-6
    fd = np.array(
        [
            (I3(AXIS_W6.at[i].add(h)) - I3(AXIS_W6.at[i].add(-h))) / (2 * h)
            for i in range(6)
        ]
    )
    x, y, z, vx, vy, vz = np.asarray(AXIS_W6)
    np.testing.assert_allclose(fd[:2], [-z * vz * vx, -z * vz * vy], rtol=1e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-10)


def test_I3_kinetic_form_unchanged():
    """The Cartesian kinetic term equals the C++ cylindrical one, to 1e-14."""
    phi = _phi(_mn())
    for w6 in _w6s(12):
        ref, _ = _cpp_reference(phi, w6, D2)
        np.testing.assert_allclose(
            integrals_of_motion(phi, w6, D2).I3, ref["I3"], rtol=1e-14
        )


def test_kk_closed_form_F_and_staeckel_constant():
    """KK is exactly Staeckel: Phi*mult = -(G(tau) - G(other)), G = GM sqrt(tau+c^2)."""
    pot = _kk()
    phi = _phi(pot)
    t = u.Q(0.0, "Myr")
    GM = pot.constants["G"].value * pot.m_tot(t, ustrip=pot.units["mass"])
    c2 = pot.c(t, ustrip=pot.units["length"]) ** 2

    def G(tau):
        return GM * np.sqrt(np.asarray(tau) + c2)

    for w6 in _w6s(8):
        iom = integrals_of_motion(phi, w6, D2)
        F = make_F(phi, iom)
        E, Lz, lam, nu = (float(getattr(iom, k)) for k in ("E", "Lz", "lam", "absnu"))
        I_lam = float(iom.I3 - iom.fnu)
        I_nu = float(iom.I3 - iom.flam)
        scale = abs(E) * lam * (lam - D2) + Lz**2 * lam + abs(I_lam) * D2
        # Staeckel constant: the two branches share one F(tau).
        np.testing.assert_allclose(
            I_lam + G(nu), I_nu + G(lam), rtol=0, atol=1e-14 * (abs(I_lam) + G(lam))
        )
        tl = np.linspace(D2, 3 * lam, 9)
        Fl = (E * tl - I_lam + G(tl) - G(nu)) * (tl - D2) - Lz**2 * tl / 2
        np.testing.assert_allclose(F.F_lam(tl), Fl, rtol=0, atol=1e-13 * scale)
        tn = np.linspace(0.0, D2, 9)
        Fn = (E * tn - I_nu + G(tn) - G(lam)) * (tn - D2) - Lz**2 * tn / 2
        np.testing.assert_allclose(F.F_nu(tn), Fn, rtol=0, atol=1e-13 * scale)


def test_iom_flags_degenerate_scales():
    """D2 below 1e-100 max(R^2 + z^2, 1), or R^2 + z^2 + D2 > 1e100: invalid.

    Such a lane is evaluated at the safe stand-in point, so its gradients
    are finite (zero); D2 down to ~1e-60 (R^2 + z^2) stays valid.
    """
    phi = _phi(_mn())
    gen = [3.0, 0.0, 1.0, 0.05, 0.2, 0.03]
    rows = [  # (w6, D2, valid)
        (gen, 4.0, True),
        (gen, 1e-60, True),
        (gen, 1e-8 * 10.0, True),  # the pipeline's fallback, (1e-4 r)^2
        ([0.0, 0.0, 0.0, 0.05, 0.2, 0.03], 1e-90, True),  # the origin
        ([0.0, 0.0, 0.0, 0.05, 0.2, 0.03], 1e-150, False),
        (gen, 1e-300, False),  # 1e-301 of R^2 + z^2
        (gen, 5e-324, False),  # subnormal
        ([1e60, 0.0, 1e60, 0.05, 0.2, 0.03], 4.0, False),  # R^2 + z^2 > 1e100
        (gen, 1e101, False),
    ]
    w6s = jnp.asarray([w for w, _, _ in rows])
    D2s = jnp.asarray([d for _, d, _ in rows])

    def f(w, d2):
        iom = integrals_of_motion(phi, w, d2)
        return jnp.stack([iom.E, iom.I3, iom.lam, iom.absnu]), iom.valid

    J, valid = jax.vmap(jax.jacrev(f, has_aux=True))(w6s, D2s)
    np.testing.assert_array_equal(valid, [v for _, _, v in rows])
    assert np.all(np.isfinite(J))
    np.testing.assert_array_equal(np.asarray(J)[~np.asarray(valid)], 0.0)
