"""Expansions against agama's: same density in, two Poisson solvers compared.

Both multipole expansions are built from one Python density callable, so a
disagreement measures the solvers, not two model definitions. Only potential
and gradient are compared: density and hessian are reconstructions through
different interpolants, so a disagreement there would measure the interpolant.
"""

from types import ModuleType

import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from ..optional_deps import OptDeps
from ._compare import assert_matches_agama

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

R_MIN, R_MAX = 0.01, 300.0  # kpc
N_R = 128
M_TOT, R_S = 1e11, 2.0  # Msun, kpc


def _ellipsoidal_hernquist(p: float, q: float):
    """Hernquist density on the ellipsoidal radius ``x^2 + (y/p)^2 + (z/q)^2``."""

    def rho(xyz, t=None):
        m = jnp.sqrt(xyz[..., 0] ** 2 + (xyz[..., 1] / p) ** 2 + (xyz[..., 2] / q) ** 2)
        return M_TOT * R_S / (2 * jnp.pi * p * q * m * (m + R_S) ** 3)

    return rho


@pytest.mark.parametrize(
    ("p", "q", "l_max", "gx_sym", "ag_sym", "rtol"),
    [
        pytest.param(1.0, 1.0, 0, "spherical", "s", 1e-8, id="spherical"),
        pytest.param(1.0, 0.6, 8, "zrotation_zreflection", "a", 1e-6, id="oblate"),
        pytest.param(0.8, 0.6, 8, "plane_reflection", "t", 1e-6, id="triaxial"),
    ],
)
def test_multipole_matches_agama(
    agama: ModuleType,
    p: float,
    q: float,
    l_max: int,
    gx_sym: str,
    ag_sym: str,
    rtol: float,
) -> None:
    """`MultipoleProfilePotential` agrees with agama's `Multipole`.

    Measured worst elementwise agreement (``n_r=128``, agama 1.0.0):
    spherical 6.2e-10 (the potential's 1.6e-10 is the two libraries' G
    differing by 1.49e-10); oblate 6.0e-7 and triaxial 5.2e-7, both in the
    gradient. The tolerances pin the regime, not the digits.
    """
    rho = _ellipsoidal_hernquist(p, q)
    galax_pot = gp.MultipoleProfilePotential.from_density(
        rho,
        r_min=u.Q(R_MIN, "kpc"),
        r_max=u.Q(R_MAX, "kpc"),
        n_r=N_R,
        l_max=l_max,
        symmetry=gx_sym,
        units="galactic",
    )
    agama_pot = agama.Potential(
        type="Multipole",
        density=lambda xyz: np.asarray(rho(xyz)),
        rmin=R_MIN,
        rmax=R_MAX,
        gridSizeR=N_R,
        lmax=l_max,
        symmetry=ag_sym,
    )
    assert_matches_agama(
        galax_pot, agama_pot, rtol=rtol, quantities=("potential", "gradient")
    )


@pytest.mark.parametrize(
    ("pot", "beta", "gamma"),
    [
        pytest.param(
            gp.ZhaoPotential(
                m=1e11, r_s=2.0, alpha=1.0, beta=4.0, gamma=0.5, units="galactic"
            ),
            4.0,
            0.5,
            id="zhao",
        ),
        pytest.param(
            gp.gNFWPotential(m=1e11, r_s=15.0, gamma=1.3, units="galactic"),
            3.0,
            1.3,
            id="gnfw",
        ),
    ],
)
def test_double_power_law_matches_agama_spheroid(
    agama: ModuleType,
    pot: gp.AbstractPotential,
    beta: float,
    gamma: float,
) -> None:
    """Analytic double-power-law potentials agree with agama's `Spheroid`.

    `Spheroid` is a density profile whose potential agama computes with its
    own multipole solver, so this is held to an expansion's tolerance, not a
    closed form's. Its normalisation is taken from galax's own density at
    ``r_s``, which avoids matching mass conventions but means density itself
    is not compared.

    Measured worst elementwise agreement (agama 1.0.0): zhao 3.0e-7, gnfw
    1.3e-7, both in the gradient.
    """
    r_s = pot.r_s(u.Q(0, "Myr"), ustrip="kpc")
    rho_rs = u.ustrip(
        "Msun / kpc3", pot.density(u.Q([r_s, 0.0, 0.0], "kpc"), t=u.Q(0, "Myr"))
    )
    agama_pot = agama.Potential(
        type="Spheroid",
        densityNorm=float(rho_rs) * 2 ** (beta - gamma),
        scaleRadius=r_s,
        alpha=1,
        beta=beta,
        gamma=gamma,
    )
    assert_matches_agama(
        pot, agama_pot, rtol=1e-6, quantities=("potential", "gradient")
    )
