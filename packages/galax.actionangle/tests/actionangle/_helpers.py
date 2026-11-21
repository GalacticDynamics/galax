"""Test helpers: test potentials, KK Delta, and O2GF runs for the oracle tests."""

__all__ = [
    "ScaledKuzminKutuzovPotential",
    "SeparablePotential",
    "kk_delta",
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
from galax.actionangle import O2GFResult, o2gf_actions_angles_freqs
from galax.actionangle._src._parse import parse_w
from galax.potential.params import AbstractParameter, ParameterField


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


_TIGHT = gd.integrate.Integrator(
    gd.OrbitSolver(
        solver=dfx.Dopri8(scan_kind="bounded"),
        stepsize_controller=dfx.PIDController(rtol=1e-12, atol=1e-12),
        max_steps=2**18,
    )
)


def o2gf_from_point(
    pot: gp.AbstractPotential, xv: np.ndarray, tmax: float, *, n_max: int = 8
) -> tuple[O2GFResult, np.ndarray]:
    """Integrate from ``xv`` (kpc, km/s) for ``tmax`` Myr and run O2GF on it.

    The orbit is sampled 2000 times. Returns the result and its actions in kpc
    km/s, ordered (J_r, J_phi, J_z).
    """
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q(xv[:3], "kpc"), p=u.Q(xv[3:], "km/s"), t=u.Q(0.0, "Myr")
    )
    ts = u.Q(jnp.linspace(0.0, tmax, 2000), "Myr")
    q, p, t = parse_w(pot, gd.evaluate_orbit(pot, w0, ts, integrator=_TIGHT))
    res = o2gf_actions_angles_freqs(pot, q, p, t, n_max=n_max)
    unit = pot.units["length"] ** 2 / pot.units["time"]
    return res, np.asarray(u.ustrip("kpc km/s", u.Q(res.aaf.actions, unit)))
