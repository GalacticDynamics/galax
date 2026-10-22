"""Compare a galax potential against an agama potential at fixed points."""

__all__ = ["POINTS", "T0", "assert_matches_agama"]

from typing import Any

import numpy as np

import unxt as u

import galax.potential as gp

# kpc. Off-axis, radii ~0.1-50 kpc, nothing at the origin: one point in the
# z = 0 plane and the rest on no symmetry plane.
POINTS = np.array(
    [
        [0.1, 0.05, 0.02],
        [0.8, -0.3, 0.4],
        [2.0, 1.0, 0.0],
        [3.5, 2.0, 1.0],
        [-5.0, 4.0, -2.5],
        [8.0, -6.0, 3.0],
        [15.0, 10.0, -12.0],
        [30.0, -25.0, 28.0],
    ]
)

# Every comparison is of static potentials, evaluated at this one time.
T0 = u.Q(0, "Myr")

_UNITS = {
    "potential": "km2 / s2",
    "gradient": "km2 / (s2 kpc)",
    "density": "Msun / kpc3",
    "hessian": "km2 / (s2 kpc2)",
}


def _sym33(d6: np.ndarray) -> np.ndarray:
    # agama's `eval(..., der=True)` orders the 6 components (xx, yy, zz, xy, yz, zx).
    return d6[..., [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]


def _agama(agama_pot: Any, name: str, xyz: np.ndarray) -> np.ndarray:
    match name:
        case "potential":
            return agama_pot.potential(xyz)
        case "gradient":
            return -agama_pot.force(xyz)
        case "density":
            return agama_pot.density(xyz)
        case "hessian":
            return -_sym33(agama_pot.eval(xyz, der=True))


def assert_matches_agama(
    galax_pot: gp.AbstractPotential,
    agama_pot: Any,
    *,
    rtol: float,
    quantities: tuple[str, ...] = ("potential", "gradient", "density", "hessian"),
) -> None:
    """Assert each requested quantity agrees, naming the one that does not."""
    x, t = u.Q(POINTS, "kpc"), T0
    for name in quantities:
        galax = u.ustrip(_UNITS[name], getattr(galax_pot, name)(x, t=t))
        agama = _agama(agama_pot, name, POINTS)
        np.testing.assert_allclose(galax, agama, rtol=rtol, err_msg=name)
