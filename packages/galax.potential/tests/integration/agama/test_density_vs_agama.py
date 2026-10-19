"""galax potentials against agama's Poisson solve of their own density.

agama is handed the galax ``density`` callable and computes the potential
itself, so these check galax's analytic potentials for models agama has no
implementation of, and `SCFPotential` against agama's basis-set solver.
"""

from types import ModuleType

import numpy as np
import pytest

import unxt as u

import galax.potential as gp
from ...unit.potential.optional_deps import OptDeps
from ._compare import assert_matches_agama

pytestmark = [
    pytest.mark.agama,
    pytest.mark.skipif(not OptDeps.AGAMA.installed, reason="requires agama"),
]

T0 = u.Q(0, "Myr")


def _density(pot: gp.AbstractPotential):
    return lambda xyz: np.asarray(
        u.ustrip("Msun / kpc3", pot.density(u.Q(xyz, "kpc"), t=T0))
    )


# Not here: `KuzminPotential`, a razor-thin disk whose volume density is zero
# off the plane, so there is nothing for agama to solve; and `SatohPotential`,
# for which agama's `CylSpline` reached only 3e-4 in the potential (80x80
# grid, ~2e6 density calls) while galax's density matches its own Laplacian
# to 1e-15, so the comparison would measure agama's grid.
@pytest.mark.parametrize(
    ("pot", "rtol"),
    [
        pytest.param(
            gp.BurkertPotential(m=1e11, r_s=5.0, units="galactic"), 5e-8, id="burkert"
        ),
        pytest.param(
            gp.StoneOstriker15Potential(
                m_tot=1e11, r_c=1.0, r_h=20.0, units="galactic"
            ),
            1e-8,
            id="stoneostriker15",
        ),
        # agama's multipole sits at ~2e-7 in the gradient here for every slope
        # and grid tried, while galax's potential matches direct quadrature of
        # this density to 3e-15: the tolerance is agama's.
        pytest.param(
            gp.PowerLawCutoffPotential(
                m_tot=1e11, alpha=1.0, r_c=10.0, units="galactic"
            ),
            1e-6,
            id="powerlawcutoff",
        ),
    ],
)
def test_matches_agama_poisson_solve(
    agama: ModuleType, pot: gp.AbstractPotential, rtol: float
) -> None:
    """The analytic potential agrees with agama's multipole of its density.

    Measured worst elementwise agreement (``gridSizeR=128``, agama 1.0.0):
    burkert 4.2e-9, stoneostriker15 9.3e-10, powerlawcutoff 1.8e-7, each in
    the gradient; the potentials agree to the 1.5e-10 by which the two
    libraries' G differ (2.7e-9 for powerlawcutoff).
    """
    agama_pot = agama.Potential(
        type="Multipole",
        density=_density(pot),
        rmin=1e-3,
        rmax=1e3,
        gridSizeR=128,
        lmax=0,
        symmetry="s",
    )
    assert_matches_agama(
        pot, agama_pot, rtol=rtol, quantities=("potential", "gradient")
    )


def test_scf_matches_agama_basisset(agama: ModuleType) -> None:
    """`SCFPotential` agrees with agama's ``BasisSet`` at ``eta=1``.

    agama's basis is Zhao's (1996), of which ``eta=1`` is the
    Hernquist-Ostriker basis that `SCFPotential` uses. Projecting a finite
    galax expansion onto it therefore recovers that expansion exactly, so the
    potentials agree to quadrature precision rather than to an expansion's
    truncation error. Measured (agama 1.0.0): 1.49e-10 in both potential and
    gradient -- exactly the two libraries' G offset -- also with the agama
    basis enlarged to ``nmax=10, lmax=6``; at ``eta=0.5`` or ``1.5`` the
    disagreement is O(1), so this does distinguish the basis.
    """
    nmax, lmax = 6, 4
    rng = np.random.default_rng(0)
    # m <= l only, and no sin(0 phi) terms: the coefficients a real field has.
    mask = np.arange(lmax + 1)[:, None] >= np.arange(lmax + 1)[None, :]
    Snlm = rng.normal(scale=0.05, size=(nmax + 1, lmax + 1, lmax + 1)) * mask
    Tnlm = rng.normal(scale=0.05, size=Snlm.shape) * mask
    Tnlm[..., 0] = 0.0
    Snlm[0, 0, 0] = 1.0
    pot = gp.SCFPotential(m_tot=1e11, r_s=3.0, Snlm=Snlm, Tnlm=Tnlm, units="galactic")

    agama_pot = agama.Potential(
        type="BasisSet",
        density=_density(pot),
        nmax=nmax,
        lmax=lmax,
        eta=1.0,
        r0=3.0,
        symmetry="n",
    )
    assert_matches_agama(
        pot, agama_pot, rtol=2e-10, quantities=("potential", "gradient")
    )
