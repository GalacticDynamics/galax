"""Make the gala reference outputs for the O2GF tests.

The data were made with gala 1.12.0 (git commit bebac7d72) from PyPI, NumPy
2.5.5, SciPy 1.18.1 and Astropy 8.0.3. gala is never a dependency of galax, so
run this in a separate environment, from the repository root:

    uv venv gala_venv && VIRTUAL_ENV=gala_venv uv pip install gala
    gala_venv/bin/python -I \
        packages/galax.actionangle/tests/actionangle/data/make_o2gf_gala_refs.py

It writes ``o2gf_gala_refs.npz`` next to this file. Each case integrates one
orbit with gala's DOPRI853 from t = 0, keeps every ``stride``-th sample, and
runs gala's ``find_actions_o2gf`` on it with gala's own toy fit. The npz holds
the samples, the toy gala fitted, and gala's actions, angles, frequencies,
S_n, dS_n/dJ and n-vectors. A case that makes gala warn (poor angle coverage,
NaN toy values) stops the script.

No case uses an exactly spherical potential or a planar orbit. In a spherical
potential gala's linear system is singular, and on a planar orbit galax drops
the rest axis, so in both cases galax differs from gala by design.
"""

import pathlib
import warnings

import astropy
import astropy.units as apyu
import gala
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import numpy as np
import scipy
from gala._version import commit_id
from gala.units import galactic

N_STEPS = 16_000
KPC_MYR = apyu.kpc / apyu.Myr

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # MilkyWayPotential2022
    MW = gp.MilkyWayPotential2022(units=galactic)

# name: (potential, q0 [kpc], p0 [kpc/Myr], dt [Myr], stride, N_max, why)
CASES = {
    "mw2022_loop": (
        MW,
        [8.0, 0.0, 0.5],
        [0.01, 0.225, 0.02],
        0.25,
        10,
        8,
        "z-tube in an axisymmetric disc plus spherical halo",
    ),
    "flat_nfw_loop": (
        gp.NFWPotential(m=1e12, r_s=20.0, a=1.0, b=1.0, c=0.8, units=galactic),
        [15.0, 0.0, 3.0],
        [0.02, 0.16, 0.04],
        0.5,
        10,
        6,
        "z-tube in an NFW halo flattened along z",
    ),
    "mn_loop": (
        gp.MiyamotoNagaiPotential(m=1e11, a=3.0, b=0.3, units=galactic),
        [6.0, 0.0, 0.3],
        [0.02, 0.2, 0.02],
        0.25,
        10,
        6,
        "z-tube in a Miyamoto-Nagai disc",
    ),
    "triaxial_nfw_xtube": (
        gp.NFWPotential(m=1e12, r_s=20.0, a=1.0, b=0.9, c=0.8, units=galactic),
        [3.0, 0.0, 10.0],
        [0.02, 0.15, 0.0],
        0.5,
        10,
        6,
        "long-axis (x) tube in a triaxial NFW halo, to test the alignment",
    ),
    "log_box": (
        gp.LogarithmicPotential(
            v_c=200 * apyu.km / apyu.s, r_h=3.0, q1=1.0, q2=0.9, q3=0.8, units=galactic
        ),
        [4.0, 3.0, 2.0],
        [0.0, 0.0, 0.0],
        0.25,
        10,
        6,
        "box orbit from rest in a triaxial cored logarithmic potential",
    ),
}


def run_case(pot, q0, p0, dt, stride, n_max):
    """Integrate one orbit and run gala's O2GF on it, refusing any warning."""
    w0 = gd.PhaseSpacePosition(pos=q0 * apyu.kpc, vel=p0 * KPC_MYR)
    orbit = pot.integrate_orbit(
        w0,
        dt=dt,
        n_steps=N_STEPS,
        Integrator=gi.DOPRI853Integrator,
        Integrator_kwargs={"atol": 1e-13, "rtol": 1e-13},
    )[::stride]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # This is what find_actions_o2gf does without a toy, keeping the toy.
        toy = gd.actionangle.fit_toy_potential(orbit)
        res = gd.actionangle.find_actions_o2gf(orbit, N_max=n_max, toy_potential=toy)
    # gala always warns "More unknowns than equations!": it compares the
    # number of unknowns with len(aa), which is 6 (the rows of aa) and not the
    # number of samples. That warning says nothing about the orbit.
    caught = [w for w in caught if "More unknowns" not in str(w.message)]
    if caught:
        msg = "; ".join(str(w.message) for w in caught)
        raise RuntimeError(msg)
    return orbit, res[0], toy


def main():
    """Run every case and write the npz."""
    out = {
        "gala_version": gala.__version__,
        "gala_commit": commit_id,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "astropy_version": astropy.__version__,
        "cases": np.array(list(CASES)),
    }
    for name, (pot, q0, p0, dt, stride, n_max, why) in CASES.items():
        orbit, row, toy = run_case(pot, q0, p0, dt, stride, n_max)
        circ = orbit.circulation()
        w = orbit.w(galactic)
        out[f"{name}__why"] = why
        out[f"{name}__q"] = w[:3].T
        out[f"{name}__p"] = w[3:].T
        out[f"{name}__t"] = orbit.t.decompose(galactic).value
        out[f"{name}__n_max"] = n_max
        out[f"{name}__circulation"] = circ
        if isinstance(toy, gp.IsochronePotential):
            out[f"{name}__toy_m"] = toy.parameters["m"].decompose(galactic).value
            out[f"{name}__toy_b"] = toy.parameters["b"].decompose(galactic).value
        else:
            out[f"{name}__toy_omega"] = (
                toy.parameters["omega"].decompose(galactic).value
            )
        out[f"{name}__actions"] = row["actions"].decompose(galactic).value
        out[f"{name}__angles"] = row["angles"].decompose(galactic).value
        out[f"{name}__freqs"] = row["freqs"].decompose(galactic).value
        out[f"{name}__sn"] = np.asarray(row["Sn"])
        out[f"{name}__dsn_dj"] = np.asarray(row["dSn_dJ"])
        out[f"{name}__nvecs"] = np.asarray(row["nvecs"])
        print(
            name,
            "T =",
            len(orbit.t),
            "circulation",
            circ,
            "toy",
            dict(toy.parameters),
            "J",
            out[f"{name}__actions"],
            "Omega",
            out[f"{name}__freqs"],
        )
    path = pathlib.Path(__file__).with_name("o2gf_gala_refs.npz")
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    main()
