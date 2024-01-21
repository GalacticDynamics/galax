.. include:: references.txt

.. _galax-install:

************
Installation
************

With ``pip`` (recommended)
==========================

To install the latest stable version using ``pip``, use::

    python -m pip install galax

This is the recommended way to install ``galax``.

To install the development version::

    python -m pip install git+https://github.com/GalacticDynamics/galax


From Source: Cloning, Building, Installing
==========================================

The latest development version of galax can be cloned from
`GitHub <https://github.com/>`_ using ``git``::

    git clone git://github.com/GalacticDynamics/galax.git

To build and install the project (from the root of the source tree, e.g., inside
the cloned ``galax`` directory)::

    python -m pip install .


Python Dependencies
===================

Explicit version requirements are specified in the project `pyproject.toml
<https://github.com/GalacticDynamics/galax/blob/main/pyproject.toml>`_. ``pip``
should install and enforce these versions automatically.

Optional
--------

Further dependencies are required for some optional features of ``galax``:

- ``matplotlib``
