"""Compare a galax potential against an agama potential at fixed points."""

__all__ = ["POINTS", "assert_matches_agama"]

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

_UNITS = {
    "potential": "km2 / s2",
    "gradient": "km2 / (s2 kpc)",
    "density": "Msun / kpc3",
    "hessian": "km2 / (s2 kpc2)",
}


def _sym33(d6: np.ndarray) -> np.ndarray:
    # agama's `eval(..., der=True)` orders the 6 components (xx, yy, zz, xy, yz, zx).
    xx, yy, zz, xy, yz, zx = np.moveaxis(d6, -1, 0)
    return np.stack(
        [
            np.stack([xx, xy, zx], -1),
            np.stack([xy, yy, yz], -1),
            np.stack([zx, yz, zz], -1),
        ],
        -2,
    )


def _agama(agama_pot: Any, name: str, xyz: np.ndarray) -> np.ndarray:
    if name == "potential":
        return agama_pot.potential(xyz)
    if name == "gradient":
        return -agama_pot.force(xyz)
    if name == "density":
        return agama_pot.density(xyz)
    return -_sym33(agama_pot.eval(xyz, der=True))


def assert_matches_agama(
    galax_pot: gp.AbstractPotential,
    agama_pot: Any,
    xyz: np.ndarray = POINTS,
    *,
    rtol: float,
    atol: float = 0.0,
    quantities: tuple[str, ...] = ("potential", "gradient", "density", "hessian"),
    up_to_constant: bool = False,
) -> None:
    """Assert each requested quantity agrees, naming the one that does not.

    ``up_to_constant`` compares the potential relative to its value at the
    first point, for models whose zero-point convention differs.
    """
    x = u.Q(xyz, "kpc")
    for name in quantities:
        galax = np.asarray(
            u.ustrip(_UNITS[name], getattr(galax_pot, name)(x, t=u.Q(0, "Myr")))
        )
        other = _agama(agama_pot, name, xyz)
        if up_to_constant and name == "potential":
            galax, other = galax - galax[0], other - other[0]
        np.testing.assert_allclose(
            galax,
            other,
            rtol=rtol,
            atol=atol,
            err_msg=name,
        )
