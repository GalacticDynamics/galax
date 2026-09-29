.. include:: references.txt

.. raw:: html

   <img src="_static/Gala_Logo_RGB.png" width="50%"
    style="margin-bottom: 32px;"/>

.. module:: galax

*****
Galax
*****

Galactic Dynamics is the study of the formation, history, and evolution of
galaxies using the *orbits* of objects — numerically-integrated trajectories of
stars, dark matter particles, star clusters, or galaxies themselves.

``galax`` is an Astropy-affiliated Python package that aims to provide efficient
tools for performing common tasks needed in Galactic Dynamics research.  This
library is written in JAX, a Python library for high-performance automatic
differentiation and numerical computation.  Common operations include
`gravitational potential and force evaluations <potential/index.html>`_, `orbit
integrations <integrate/index.html>`_, `dynamical coordinate transformations
<dynamics/index.html>`_, and computing `chaos indicators for nonlinear dynamics
<dynamics/nonlinear.html>`_. ``galax`` heavily uses the units and astronomical
coordinate systems defined in the Astropy core package (:ref:`astropy.units
<astropy-units>` and :ref:`astropy.coordinates <astropy-coordinates>`).

This package is being actively developed in `a public repository on GitHub
<https://github.com/adrn/gala>`_, and we are always looking for new
contributors! No contribution is too small, so if you have any trouble with this
code, find a typo, or have requests for new content (tutorials or features),
please `open an issue on GitHub <https://github.com/adrn/gala/issues>`_.

.. ---------------------
.. Nav bar (top of docs)

.. toctree::
   :maxdepth: 1
   :titlesonly:

   install
   getting_started
   tutorials
   user_guide
   contributing


Contributors
============

.. include:: ../AUTHORS.rst


Citation and Attribution
========================

Citing ``galax``
----------------

WIP

.. _galax-attribution-agama:

Agama
-----

The spline-interpolated multipole machinery
(:class:`~galax.potential.MultipoleProfilePotential` and the
:mod:`galax.potential.harmonic` support it is built on) follows `Agama
<https://github.com/GalacticDynamics-Oxford/Agama>`_ closely. In places it is a
direct port of Agama's algorithms from C++. If you use these potentials, please
cite Agama alongside ``galax``:

    Vasiliev, E. 2019, *AGAMA: action-based galaxy modelling architecture*,
    MNRAS, 482, 1525. `arXiv:1802.08239 <https://arxiv.org/abs/1802.08239>`_

What derives from Agama:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``galax``
     - Agama
   * - :mod:`galax.potential.harmonic` asymptotic continuation --- the
       :math:`W x^v + U x^s + Q x^2` form, the inward and outward exponents,
       the convergence guards and fallback slopes, and the four-parameter
       inner-monopole bracket
     - ``PowerLawMultipole`` and ``computeExtrapolationCoefs`` in
       ``src/potential_multipole.cpp``
   * - :class:`~galax.potential.Symmetry` --- the symmetry vocabulary
     - ``SymmetryType`` in ``src/coord.h``
   * - the real spherical-harmonic basis convention
     - Agama's convention, shared with :mod:`scipy`
   * - ``galax.dynamics`` Milky-Way/LMC interaction example
     - ``py/example_lmc_mw_interaction.py``

Agama was also the numerical reference while this machinery was developed: it
is what established that the boundary tail was the dominant error source, and
that an earlier interior quadrature was only second order. Those comparisons
were run offline --- Agama is **not** a dependency of ``galax``, at test time
or otherwise.

Where ``galax`` differs
^^^^^^^^^^^^^^^^^^^^^^^

The port is not bit-compatible with Agama, by design:

- The radial Poisson solve is the textbook Green's-function solution rather
  than a port, and uses a Hermite-corrected quadrature that is fourth order in
  :math:`\ln r` where Agama's is second.
- The radial grid is padded beyond the requested bracket before the solve, so
  the asymptotic fit is read off an interior region rather than the boundary.
- The inner and outer continuations are merged into the same evaluation path.

License
^^^^^^^

Agama's ``LICENSE`` states that the library as distributed is GPL *because it
links GSL*, but that

    the original source code of Agama itself is not subject to GPL and is
    provided under the less restrictive BSD or MIT licenses

and grants permission "to anyone to use this software for any purpose, and to
alter it and redistribute it freely", asking that acknowledgement of the
original author be given. ``galax`` ports algorithm source and does not link
GSL, which is compatible with ``galax``'s MIT license. The full text is
reproduced in ``licences/Agama.txt``.

GitHub's license auto-detection reports ``NOASSERTION`` for the Agama
repository, so this is recorded here rather than left to be re-derived.
