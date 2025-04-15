"""Test API of `galax.potential` package."""

import galax.potential as gp

expected_all = [
    # Modules
    "io",
    "params",
    "plot",
    # base
    "AbstractPotential",
    # core
    "AbstractSinglePotential",
    # composite
    "AbstractCompositePotential",
    "AbstractPreCompositedPotential",
    "CompositePotential",
    # builtin
    "LongMuraliBarPotential",
    "MonariEtAl2016BarPotential",
    "KuzminPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "SatohPotential",
    "LMJ09LogarithmicPotential",
    "LogarithmicPotential",
    "AbstractMultipolePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
    "LeeSutoTriaxialNFWPotential",
    "NFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
    "NullPotential",
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
    "BurkertPotential",
    "HernquistPotential",
    "IsochronePotential",
    "JaffePotential",
    "KeplerPotential",
    "PlummerPotential",
    "PowerLawCutoffPotential",
    "StoneOstriker15Potential",
    "TriaxialHernquistPotential",
    "HardCutoffNFWPotential",
    # xfm
    "AbstractTransformedPotential",
    "TransformedPotential",
    "TriaxialInThePotential",
    "TranslatedPotential",
    # funcs
    "potential",
    "gradient",
    "laplacian",
    "density",
    "hessian",
    "acceleration",
    "tidal_tensor",
    "local_circular_velocity",
    "spherical_mass_enclosed",
    "dpotential_dr",
    "d2potential_dr2",
]


def test_all() -> None:
    """Test the `galax.potential` package contents."""
    # Test detailed contents (not order)
    assert set(gp.__all__) == set(expected_all)
