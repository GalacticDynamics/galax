""":mod:`galax.potential`.

Each built-in potential has a module of plain functions named for its class in
lower case without ``Potential``, e.g. :mod:`galax.potential.plummer` for
``PlummerPotential``. The functions take the parameters ``p = pot.params(t)``
and bare arrays in the unit system of the potential. They work on one point at
a time (``r`` and ``t`` scalars, ``xyz`` of shape ``(3,)``); use ``jax.vmap``
for batches. Spherical modules take ``(p, r)`` for ``potential``, ``density``,
``laplacian``, ``mass_enclosed``, ``dpotential_dr`` and ``d2potential_dr2``, and
``(p, xyz)`` for ``gradient`` and ``hessian``. Non-spherical modules take
``(p, xyz)`` for ``potential``, ``density``, ``laplacian``, ``gradient`` and
``hessian``; ``monarietal2016bar`` adds a trailing scalar ``t``.
``galax.potential.plummer.potential(p, r)`` is a different function from the
dispatched ``galax.potential.potential(pot, xyz, t)``. At the origin a
spherical ``gradient`` is 0 and a spherical ``hessian`` is ``laplacian(p, 0) / 3``
times the identity (``+inf`` on the diagonal for a cusp, never NaN). Functions
of ``r`` expect ``r > 0``. Classes keep the base-class autodiff for the
functions that a shared helper builds.
"""

__all__ = [
    # Modules
    "burkert",
    "gaussian",
    "gnfw",
    "hardcutoffnfw",
    "harmonic",
    "hernquist",
    "io",
    "isochrone",
    "jaffe",
    "kepler",
    "kuzmin",
    "kuzminkutuzov",
    "leesutotriaxialnfw",
    "longmuralibar",
    "miyamotonagai",
    "monarietal2016bar",
    "multipole_profile",
    "nfw",
    "params",
    "plot",
    "plummer",
    "powerlawcutoff",
    "satoh",
    "scf",
    "stoneostriker15",
    "zhao",
    # ABCs
    "AbstractPotential",
    "AbstractSinglePotential",
    "AbstractCompositePotential",
    # symmetry
    "Symmetry",
    # composite
    "CompositePotential",
    # builtin
    "LongMuraliBarPotential",
    "MonariEtAl2016BarPotential",
    "KuzminPotential",
    "KuzminKutuzovPotential",
    "MiyamotoNagaiPotential",
    "MN3ExponentialPotential",
    "MN3Sech2Potential",
    "HarmonicOscillatorPotential",
    "HenonHeilesPotential",
    "SatohPotential",
    "LMJ09LogarithmicPotential",
    "LogarithmicPotential",
    "AbstractMultipolePotential",
    "AbstractMultipoleProfilePotential",
    "MultipoleInnerPotential",
    "MultipoleOuterPotential",
    "MultipolePotential",
    "MultipoleProfilePotential",
    "AxisymmetricGaussianPotential",
    "GaussianPotential",
    "TriaxialGaussianPotential",
    "LeeSutoTriaxialNFWPotential",
    "NFWPotential",
    "TriaxialNFWPotential",
    "Vogelsberger08TriaxialNFWPotential",
    "NullPotential",
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
    "gNFWPotential",
    "ZhaoPotential",
    "SCFPotential",
    # Pre-composited
    "AbstractPreCompositedPotential",
    "BovyMWPotential2014",
    "LM10Potential",
    "MilkyWayPotential",
    "MilkyWayPotential2022",
    # xfm
    "AbstractTransformedPotential",
    "FlattenedInThePotential",
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
    "potential_energy",
    "total_energy",
    "dpotential_dr",
    "d2potential_dr2",
]

from .setup_package import install_import_hook, load_interop_plugins

with install_import_hook("galax.potential"):
    from . import (
        burkert,
        gaussian,
        gnfw,
        hardcutoffnfw,
        harmonic,
        hernquist,
        io,
        isochrone,
        jaffe,
        kepler,
        kuzmin,
        kuzminkutuzov,
        leesutotriaxialnfw,
        longmuralibar,
        miyamotonagai,
        monarietal2016bar,
        multipole_profile,
        nfw,
        params,
        plot,
        plummer,
        powerlawcutoff,
        satoh,
        scf,
        stoneostriker15,
        zhao,
    )
    from ._src.api import (
        acceleration,
        d2potential_dr2,
        density,
        dpotential_dr,
        gradient,
        hessian,
        laplacian,
        local_circular_velocity,
        potential,
        potential_energy,
        spherical_mass_enclosed,
        tidal_tensor,
        total_energy,
    )
    from ._src.base import AbstractPotential
    from ._src.base_multi import (
        AbstractCompositePotential,
        AbstractPreCompositedPotential,
    )
    from ._src.base_single import AbstractSinglePotential
    from ._src.builtin import (
        AbstractMultipolePotential,
        AbstractMultipoleProfilePotential,
        AxisymmetricGaussianPotential,
        BovyMWPotential2014,
        BurkertPotential,
        GaussianPotential,
        HardCutoffNFWPotential,
        HarmonicOscillatorPotential,
        HenonHeilesPotential,
        HernquistPotential,
        IsochronePotential,
        JaffePotential,
        KeplerPotential,
        KuzminKutuzovPotential,
        KuzminPotential,
        LeeSutoTriaxialNFWPotential,
        LM10Potential,
        LMJ09LogarithmicPotential,
        LogarithmicPotential,
        LongMuraliBarPotential,
        MilkyWayPotential,
        MilkyWayPotential2022,
        MiyamotoNagaiPotential,
        MN3ExponentialPotential,
        MN3Sech2Potential,
        MonariEtAl2016BarPotential,
        MultipoleInnerPotential,
        MultipoleOuterPotential,
        MultipolePotential,
        MultipoleProfilePotential,
        NFWPotential,
        NullPotential,
        PlummerPotential,
        PowerLawCutoffPotential,
        SatohPotential,
        SCFPotential,
        StoneOstriker15Potential,
        TriaxialGaussianPotential,
        TriaxialHernquistPotential,
        TriaxialNFWPotential,
        Vogelsberger08TriaxialNFWPotential,
        ZhaoPotential,
        gNFWPotential,
    )
    from ._src.composite import CompositePotential
    from ._src.symmetry import Symmetry
    from ._src.xfm import (
        AbstractTransformedPotential,
        FlattenedInThePotential,
        TransformedPotential,
        TranslatedPotential,
        TriaxialInThePotential,
    )

    # Register functions by module import
    # isort: split
    from ._src import register_funcs


# Cleanup
del install_import_hook, register_funcs

# Interoperability with third-party libraries. Importing a registered module is
# what performs its `plum` dispatch registration; entry points let separately
# installed distributions extend `galax.potential` without it knowing they exist.
load_interop_plugins("galax.potential.interop")

del load_interop_plugins
