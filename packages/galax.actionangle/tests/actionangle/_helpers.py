"""Test helpers: test potentials, KK Delta, and O2GF runs for the oracle tests."""

__all__ = [
    "HO",
    "HO_OMEGA",
    "HO_TOY",
    "ISO",
    "ISO_TOY",
    "KMS",
    "KPC",
    "MYR",
    "ScaledKuzminKutuzovPotential",
    "SeparablePotential",
    "angle_diff",
    "integrate",
    "kk_delta",
    "lmj09",
    "mw2022_parts",
    "o2gf_from_point",
]

import diffrax as dfx
import jax
import numpy as np

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.actionangle import (
    IsochroneToy,
    O2GFResult,
    OscillatorToy,
    o2gf_actions_angles_freqs,
)
from galax.actionangle._src._parse import parse_w
from galax.potential.params import AbstractParameter, ParameterField

ISO = gp.IsochronePotential(
    m_tot=u.Q(1e11, "Msun"), r_s=u.Q(2.0, "kpc"), units="galactic"
)
HO_OMEGA = np.array([0.7, 1.3, 2.1])
HO = gp.HarmonicOscillatorPotential(omega=u.Q(HO_OMEGA, "1/Myr"), units="galactic")
ISO_TOY = IsochroneToy(m=jnp.asarray(1e11), b=jnp.asarray(2.0))
HO_TOY = OscillatorToy(omega=jnp.asarray(HO_OMEGA))
KPC, MYR = u.Q(1.0, "kpc").ustrip("m"), u.Q(1.0, "Myr").ustrip("s")
KMS = u.Q(1.0, "km/s").ustrip("kpc/Myr")
# The default integrator tolerance (1e-7) would limit the O2GF tests.
_TIGHT = gd.integrate.Integrator(
    gd.OrbitSolver(
        solver=dfx.Dopri8(scan_kind="bounded"),
        stepsize_controller=dfx.PIDController(rtol=1e-12, atol=1e-12),
        max_steps=2**18,
    )
)


class ScaledKuzminKutuzovPotential(gp.AbstractSinglePotential):
    r"""Kuzmin-Kutuzov, Delta^2 = a^2 - c^2, evaluated with lengths in units of c.

    `galax.potential.KuzminKutuzovPotential` squares the input coordinates,
    which overflows float32 in SI units (R ~ 1e20 m); this one does not, to
    test that `staeckel_fudge_delta` itself is scale-free there.
    """

    m_tot: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    a: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    c: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]

    @jax.jit
    def _potential(self, xyz, t, /):
        t = u.Q.from_(t, self.units["time"])
        G = self.constants["G"].value
        m = self.m_tot(t, ustrip=self.units["mass"])
        a = self.a(t, ustrip=self.units["length"])
        c = self.c(t, ustrip=self.units["length"])
        xyz = u.ustrip(AllowValue, self.units["length"], xyz) / c
        R2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
        z2 = xyz[..., 2] ** 2
        a2 = (a / c) ** 2
        lam_nu = R2 + a2 * z2 + a2
        denom = jnp.sqrt(R2 + z2 + a2 + 1 + 2 * jnp.sqrt(lam_nu))
        return -G * m / c / denom


class SeparablePotential(gp.AbstractSinglePotential):
    r"""$\Phi = f(R) + g(z)$, so $\partial^2\Phi / \partial R \partial z = 0$.

    $f = \frac{1}{2} R^2$, $g = z^2$, in the potential's units.
    """

    @jax.jit
    def _potential(self, xyz, t, /):
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        return 0.5 * (xyz[..., 0] ** 2 + xyz[..., 1] ** 2) + xyz[..., 2] ** 2


def kk_delta(pot: gp.KuzminKutuzovPotential, t: float = 0.0) -> float:
    """Focal distance Delta = sqrt(a^2 - c^2) at time ``t``, in ``pot.units``."""
    t = u.Q(t, pot.units["time"])
    a = pot.a(t, ustrip=pot.units["length"])
    c = pot.c(t, ustrip=pot.units["length"])
    return float(jnp.sqrt(a**2 - c**2))


def mw2022_parts() -> list[tuple[str, dict[str, float]]]:
    """Return the parts of `galax.potential.MilkyWayPotential2022`, in Msun and kpc.

    Each is ``(kind, params)`` with kind ``"MiyamotoNagai"`` (``m``, ``a``,
    ``b``), ``"NFW"`` (``m``, ``r_s``) or ``"Hernquist"`` (``m``, ``r_s``).
    The disk is the three Miyamoto-Nagai disks that make up its MN3 sech^2
    fit, so galpy and agama can build the same potential from their own
    classes. NFW ``m`` is the scale mass in ``-G m ln(1 + r/r_s) / r``.
    """
    mw = gp.MilkyWayPotential2022()
    t = u.Q(0.0, "Myr")
    parts = [
        (
            "MiyamotoNagai",
            {
                "m": float(mn.m_tot(t, ustrip="Msun")),
                "a": float(mn.a(t, ustrip="kpc")),
                "b": float(mn.b(t, ustrip="kpc")),
            },
        )
        for mn in mw["disk"]._get_mn_components(t)
    ]
    halo = mw["halo"]
    parts.append(
        (
            "NFW",
            {
                "m": float(halo.m(t, ustrip="Msun")),
                "r_s": float(halo.r_s(t, ustrip="kpc")),
            },
        )
    )
    parts.extend(
        (
            "Hernquist",
            {
                "m": float(mw[k].m_tot(t, ustrip="Msun")),
                "r_s": float(mw[k].r_s(t, ustrip="kpc")),
            },
        )
        for k in ("bulge", "nucleus")
    )
    return parts


def lmj09(q2: float, q3: float) -> gp.LMJ09LogarithmicPotential:
    """Triaxial LMJ09 logarithmic potential with axis ratios ``q2`` and ``q3``."""
    return gp.LMJ09LogarithmicPotential(
        v_c=u.Q(200.0, "km/s"),
        r_s=u.Q(3.0, "kpc"),
        q1=1.0,
        q2=q2,
        q3=q3,
        phi=u.Q(0.0, "rad"),
        units="galactic",
    )


def angle_diff(a, b):
    """Difference of two angles, wrapped into [-pi, pi)."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def integrate(pot, q0, p0, tmax, n=2000, *, orbit=False):
    """Integrate from ``q0`` (kpc) and ``p0`` (kpc/Myr) at t = 0 to ``tmax`` Myr.

    ``q0`` and ``p0`` may carry batch axes. The orbit is sampled ``n`` times.
    Returns the orbit, or ``(q, p, t)`` as NumPy arrays in ``pot.units``.
    """
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(q0, "kpc"),
        p=u.Q(p0, "kpc/Myr"),
        t=u.Q(np.zeros(np.shape(q0)[:-1]), "Myr"),
    )
    ts = u.Q(jnp.linspace(0.0, tmax, n), "Myr")
    orb = gd.evaluate_orbit(pot, w0, ts, integrator=_TIGHT)
    return orb if orbit else tuple(np.asarray(x) for x in parse_w(pot, orb))


def o2gf_from_point(
    pot: gp.AbstractPotential, xv: np.ndarray, tmax: float, *, n_max: int = 8
) -> tuple[O2GFResult, np.ndarray]:
    """Integrate from ``xv`` (kpc, km/s) for ``tmax`` Myr and run O2GF on it.

    The orbit is sampled 2000 times. Returns the result and its actions in kpc
    km/s, ordered (J_r, J_phi, J_z).
    """
    orb = integrate(
        pot, xv[:3], u.ustrip("kpc/Myr", u.Q(xv[3:], "km/s")), tmax, orbit=True
    )
    res = o2gf_actions_angles_freqs(pot, orb, n_max=n_max)
    return res, np.asarray(u.ustrip("kpc km/s", res.aaf.actions))
