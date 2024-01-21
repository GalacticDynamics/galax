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

WIP
