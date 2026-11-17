"""Tests for the experimental kinematic distribution functions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.stats import ks_2samp

import unxt as u

import galax.coordinates as gc
import galax.dynamics as gd
import galax.potential as gp
from galax.dynamics._src.experimental.df import Fardal2015DF


def test_fardal_arms_drawn_independently() -> None:
    """Leading and trailing particles are separate draws from the DF.

    In the progenitor's frame the leading arm is the trailing DF rotated by 180
    deg about L, so undoing that rotation gives two samples of the same DF. They
    must agree in distribution but not be the same draw (gala and the Chen+24
    authors' agama code draw each arm on its own).
    """
    pot = gp.NFWPotential(m=1e12, r_s=15.0, units="galactic")
    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([30.0, 10, 20], "kpc"), p=u.Q([10.0, -150, -20], "km/s"), t=u.Q(0, "Myr")
    )
    n = 4000
    ts = jnp.linspace(-3000.0, 0.0, n)
    prog = gd.evaluate_orbit(pot, w0, u.Q(ts, "Myr"))
    pw = jnp.asarray(prog.w(units=pot.units))
    x, v = pw[:, :3], pw[:, 3:]
    msat = 1e5

    keys = jr.split(jr.key(0), n)
    sample = jax.vmap(Fardal2015DF().sample, in_axes=(0, None, 0, 0, 0, None))
    x_lead, v_lead, x_trail, v_trail = map(
        np.asarray, sample(keys, pot, ts, x, v, msat)
    )

    rt = np.asarray(gd.tidal_radius(pot, x, v, mass=msat, t=ts))[:, None]
    x, v = np.asarray(x), np.asarray(v)
    r = np.linalg.norm(x, axis=1, keepdims=True)
    xh = x / r
    zh = np.cross(x, v)
    omega = np.linalg.norm(zh, axis=1, keepdims=True) / r**2
    zh /= np.linalg.norm(zh, axis=1, keepdims=True)
    yh = np.cross(zh, xh)
    vscale = omega * rt  # the Fardal+15 velocity scale

    def offsets(xs: np.ndarray, vs: np.ndarray, sign: float, /) -> np.ndarray:
        """Position and velocity offsets, in units of r_t and omega r_t.

        For each: the size, then x, y, z. x and y are rotated back for the lead
        (z is unchanged by the rotation).
        """
        cols = []
        for d in ((xs - x) / rt, (vs - v) / vscale):
            cols += [np.linalg.norm(d, axis=1), sign * (d * xh).sum(1)]
            cols += [sign * (d * yh).sum(1), (d * zh).sum(1)]
        return np.column_stack(cols)

    trail = offsets(x_trail, v_trail, 1.0)
    lead = offsets(x_lead, v_lead, -1.0)
    for i in range(trail.shape[1]):
        if np.std(trail[:, i]) < 1e-8:  # e.g. the y position offset is 0
            continue
        # Same DF.
        assert ks_2samp(trail[:, i], lead[:, i]).pvalue > 1e-3, i
        # Separate draws: one shared draw gives correlation 1; with n = 4000
        # independent draws |corr| ~ 0.016.
        assert abs(np.corrcoef(trail[:, i], lead[:, i])[0, 1]) < 0.1, i
