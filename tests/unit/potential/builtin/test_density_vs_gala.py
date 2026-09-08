"""Cross-check the analytic ``_density`` formulas added this session against gala.

Most potentials already get this coverage for free from
`GalaIOMixin.test_method_gala` (see ``tests/unit/potential/io/test_gala.py``),
which every ``AbstractSinglePotential_Test`` subclass inherits. This module
makes that check explicit and consolidated for the closed-form densities
newly derived here (Poisson's equation applied to potentials that previously
had no ``_density`` override, or a stray-unit bug), so a regression in any of
them is easy to spot in one place.
"""

import astropy.units as apyu
import pytest
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
from galax.interop.optional_deps import GSL_ENABLED, OptDeps

X = u.Q([3.5, 2.0, 1.0], "kpc")  # generic off-axis evaluation point

POTENTIALS = [
    pytest.param(
        gp.MiyamotoNagaiPotential(m_tot=6e10, a=6.5, b=0.26, units="galactic"),
        id="miyamotonagai",
    ),
    pytest.param(gp.JaffePotential(m_tot=1e12, r_s=10.0, units="galactic"), id="jaffe"),
    pytest.param(
        gp.IsochronePotential(m_tot=1e11, r_s=5.0, units="galactic"), id="isochrone"
    ),
    pytest.param(
        gp.LogarithmicPotential(v_c=220.0, r_s=8.0, units="galactic"),
        id="logarithmic",
    ),
    pytest.param(
        gp.LMJ09LogarithmicPotential(
            v_c=220.0,
            r_s=12.0,
            q1=1.3,
            q2=1.0,
            q3=0.8,
            phi=u.Q(20, "deg"),
            units="galactic",
        ),
        id="lmj09logarithmic",
        # Not our bug: gala's compiled `logarithmic_density` (in
        # builtin_potentials.cpp) uses the *un*-rotated position `q[0],
        # q[1]` directly, while `logarithmic_value`/`logarithmic_gradient`
        # correctly rotate by `phi` first. So gala's own potential/gradient
        # and density disagree with each other whenever `phi != 0` and `q1
        # != q2`; galax's `density` (verified exactly self-consistent with
        # its own potential via Poisson's equation) is unaffected. Already
        # flagged as a TODO in test_lmj09logarithmic.py::test_method_gala.
        marks=pytest.mark.xfail(
            reason="gala's logarithmic_density omits the phi rotation "
            "(compiled builtin_potentials.cpp bug, not a galax issue)",
            strict=True,
        ),
    ),
    pytest.param(
        gp.SatohPotential(m_tot=1e11, a=6.0, b=0.3, units="galactic"), id="satoh"
    ),
    pytest.param(
        gp.KuzminPotential(m_tot=1e11, r_s=1.0, units="galactic"), id="kuzmin"
    ),
    pytest.param(
        gp.LongMuraliBarPotential(
            m_tot=1e10, a=5.0, b=1.0, c=0.3, alpha=u.Q(0.9, "rad"), units="galactic"
        ),
        id="longmuralibar",
    ),
    pytest.param(
        gp.PowerLawCutoffPotential(m_tot=1e11, alpha=1.2, r_c=10.0, units="galactic"),
        id="powerlawcutoff",
        marks=pytest.mark.skipif(not GSL_ENABLED, reason="requires gala + GSL"),
    ),
]


@pytest.mark.skipif(not OptDeps.GALA.installed, reason="requires gala")
@pytest.mark.parametrize("pot", POTENTIALS)
def test_density_matches_gala(pot: gp.AbstractSinglePotential) -> None:
    """`density` matches gala's reference implementation for the same model."""
    gala_pot = gp.io.convert_potential(gp.io.GalaLibrary, pot)

    galax_density = convert(pot.density(X, t=0), u.Quantity)
    gala_density = convert(gala_pot.density(convert(X, apyu.Quantity)), u.Quantity)
    assert jnp.allclose(galax_density, gala_density, atol=u.Q(1e-8, galax_density.unit))


# ---------------------------------------------------------------------------
# Not comparable to gala:
#
# - `MonariEtAl2016BarPotential`: this model has no gala counterpart.
# - `MultipoleInnerPotential` / `MultipoleOuterPotential` / `MultipolePotential`:
#   already tested directly (`_density` is exactly zero for every solid-harmonic
#   term, verified in test_multipole.py / test_innermultipole.py /
#   test_outermultipole.py); their `test_method_gala` is separately marked
#   xfail for reasons unrelated to density (see test_multipole.py).
