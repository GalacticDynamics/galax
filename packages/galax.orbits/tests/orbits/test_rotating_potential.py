"""Orbits in a `galax.potential.RotatingPotential`.

These live in the orbits tree because they integrate orbits: `orbits` may
depend on `potential`, not the reverse.
"""

import numpy as np
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.coordinates as gc
import galax.orbits as go
import galax.potential as gp


def test_jacobi_energy_is_conserved() -> None:
    """In a rigidly rotating potential, ``E - Omega L_z`` is an integral of motion.

    The host is a disk and a halo (both axisymmetric) plus a static bar spun up
    by the transform, so the whole potential rotates rigidly at ``Omega``. The
    energy itself is not conserved, which the test also checks so that it would
    notice a transform that does not rotate.
    """
    omega = u.Q(40.0, "km/(s kpc)")
    bar = gp.MonariEtAl2016BarPotential(
        alpha=u.Q(0.05, ""),
        R0=u.Q(8.0, "kpc"),
        v0=u.Q(220.0, "km/s"),
        Rb=u.Q(3.5, "kpc"),
        Omega=u.Q(0.0, "km/(s kpc)"),
        phi_b=u.Q(0.0, "deg"),
        units="galactic",
    )
    pot = gp.CompositePotential(
        disk=gp.MiyamotoNagaiPotential(
            m_tot=u.Q(6e10, "Msun"),
            a=u.Q(3, "kpc"),
            b=u.Q(0.28, "kpc"),
            units="galactic",
        ),
        halo=gp.NFWPotential(m=u.Q(5e11, "Msun"), r_s=u.Q(16, "kpc"), units="galactic"),
        bar=gp.RotatingPotential(bar, Omega=omega, phi_0=u.Q(25, "deg")),
    )

    w0 = gc.PhaseSpaceCoordinate(
        q=u.Q([4.0, 0.5, 0.2], "kpc"),
        p=u.Q([30.0, 190.0, 10.0], "km/s"),
        t=u.Q(0.0, "Myr"),
    )
    ts = u.Q(jnp.linspace(0.0, 2_000.0, 201), "Myr")
    orbit = go.compute_orbit(pot, w0, ts)

    usys = pot.units
    xyz = convert(orbit.q, u.Q).ustrip(usys["length"])
    vxyz = convert(orbit.p, u.Q).ustrip(usys["speed"])
    energy = gp.total_energy(pot, orbit).ustrip(usys["specific energy"])
    l_z = xyz[:, 0] * vxyz[:, 1] - xyz[:, 1] * vxyz[:, 0]
    e_j = energy - omega.ustrip(usys["frequency"]) * l_z

    scale = np.abs(e_j[0])
    assert np.ptp(energy) / scale > 1e-3  # the bar does work on the orbit
    np.testing.assert_allclose(e_j, e_j[0], rtol=1e-6)
