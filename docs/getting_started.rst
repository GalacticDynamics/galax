.. _galax-getting-started:

***************
Getting Started
***************

Welcome to the :mod:`galax` documentation!

.. TODO: in the paragraph below, switch the matplotlib link to :mod:`matplotlib`
.. when they add a top-level module definition

For practical reasons, this documentation generally assumes that you are
familiar with the Python programming language, including numerical and
computational libraries like :mod:`numpy`, :mod:`scipy`, and `matplotlib
<https://matplotlib.org/>`_. If you need a refresher on Python programming, we
recommend starting with the `official Python tutorial
<https://docs.python.org/3/tutorial/>`_, but many other good resources are
available on the internet, such as tutorials and lectures specifically designed
for `using Python for scientific applications <https://scipy-lectures.org/>`_.

On this introductory page, we will demonstrate a few common use cases for :mod:`galax`
and give an overview of the package functionality. For the examples
below, we will assume that the following imports have already been executed
because these packages will be generally required::

    >>> import astropy.units as u
    >>> import jax.numpy as jnp


Computing your first stellar orbit
==================================

One of the most common use cases for :mod:`galax` is to compute an orbit for a star
within a mass model for the Milky Way. To do this, we need to specify two
things: (1) the model of the Milky Way that we would like to use to represent
the mass distribution, and (2) the initial conditions of the star's orbit.

Mass models in :mod:`galax` are specified using Python classes that represent
standard gravitational potential models. For example, most of the standard,
parametrized gravitational potential models introduced in :cite:`Binney2008` are
available as classes in the :mod:`galax.potential` module. The standard Milky
Way model recommended for use in :mod:`galax` is the
:class:`~galax.potential.MilkyWayPotential`, which is a pre-defined,
multi-component mass model with parameters set to fiducial values that match the
rotation curve of the Galactic disk and the mass profile of the dark matter
halo. We can create an instance of this model with the fiducial parameters by
instantiating the :class:`~galax.potential.MilkyWayPotential` class without any
input::

    >>> import galax.potential as gp
    >>> mw = gp.MilkyWayPotential()
    >>> mw
    MilkyWayPotential({'disk': MiyamotoNagaiPotential(...), 'halo': NFWPotential(...),
                       'bulge': HernquistPotential(...), 'nucleus': HernquistPotential(...)})

This model, by default, contains four distinct potential components as listed in
the output above: disk, bulge, nucleus, and halo components. You can configure
any of the parameters of these components, or create your own "composite"
potential model using other potential models defined in :mod:`galax.potential`,
but for now we will use the fiducial model as we defined it, the variable
``mw``.

All of the :mod:`galax.potential` class instances have a set of standard methods
that enable fast calculations of computed or derived quantities. For example,
we could compute the potential energy or the acceleration at a Cartesian
position near the Sun::

    >>> xyz = [-8., 0, 0] * u.kpc
    >>> mw.potential(xyz, t=0).to_units("kpc2 / Myr2")
    Quantity['specific energy'](Array(-0.16440296, dtype=float64), unit='kpc2 / Myr2')
    >>> mw.acceleration(xyz, t=0).to_units("kpc/Myr2")
    Quantity['acceleration'](Array([ 0.00702262, -0.        , -0.        ], dtype=float64), unit='kpc / Myr2')

The values that are returned by most methods in :mod:`galax` are provided as
Astropy :class:`~astropy.units.Quantity` objects, which represent numerical data
with associated physical units. :class:`~astropy.units.Quantity` objects can be
re-represented in any equivalent units, so, for example, we could display the
energy or acceleration in other units::

    >>> mw.potential(xyz, t=0).to_units("kpc2/Myr2")
    Quantity['specific energy'](Array(-0.16440296, dtype=float64), unit='kpc2 / Myr2')
    >>> mw.acceleration(xyz, t=0).to_units("kpc/Myr2")
    Quantity['acceleration'](Array([ 0.00702262, -0.        , -0.        ], dtype=float64), unit='kpc / Myr2')

Now that we have a potential model, if we want to compute an orbit, we need to
specify a set of initial conditions to initialize the numerical orbit
integration. In :mod:`galax`, initial conditions and other positions in
phase-space (locations in position and velocity space) are defined using the
:class:`~galax.coordinates.PhaseSpacePosition` class. This class allows a number of
possible inputs, but one of the most common inputs are Cartesian position and
velocity vectors. As an example orbit, we will use a position and velocity that
is close to the Sun's Galactocentric position and velocity::

    >>> import galax.coordinates as gc
    >>> psp = gc.PhaseSpacePosition(q=[-8.1, 0, 0.02] * u.kpc,
    ...                             p=[13, 245, 8.] * u.km/u.s)

By convention, I typically use the variable ``w`` to represent phase-space
positions, so here ``psp`` is meant to imply "initial conditions." Note that,
when passing in Cartesian position and velocity values, we typically have to
pass them in as :class:`~astropy.units.Quantity` objects (i.e., with units).
This is required whenever the potential class you are using has a unit system,
which you can check by calling the
:obj:`~galax.potential.AbstractPotentialBase.units` attribute of your potential
object::

    >>> mw.units
    UnitSystem(kpc, Myr, solMass, rad)

Here, our Milky Way potential model has a unit system with dimensional units.
Note that we could have used any length unit for the position and any velocity
unit for the velocity, because :mod:`galax` handles the unit conversions
internally.

Now with a potential model defined and a set of initial conditions, we are set
to compute an orbit! To do this, we use the numerical integration system defined
in :mod:`galax.integrate`, but do so using the convenience interface available
on any Potential object through the
:func:`~galax.potential.AbstractPotential.evaluate_orbit` method::

    >>> import galax.dynamics as gd
    >>> t = jnp.arange(0.0, 2.0, step=1/1000) # Gyr
    >>> orbit = gd.evaluate_orbit(mw, psp.w(units=mw.units), t=t)

By default, this method uses Leapfrog integration , which is a fast, symplectic
integration scheme. The returned object is an instance of the
:class:`~galax.dynamics.Orbit` class, which is similar to the
:class:`~galax.coordinates.PhaseSpacePosition` but represents a collection of
phase-space positions at times::

    >>> orbit
    Orbit(
      q=CartesianPosition3D(
        x=Quantity[PhysicalType('length')](value=f64[2000], unit=Unit("kpc")),
        ...

:class:`~galax.dynamics.Orbit` objects have many of their own useful methods for
performing common tasks, like plotting an orbit::

    >>> orbit.plot(['x', 'y'])  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    import galax.coordinates as gc
    import galax.dynamics as gd
    import galax.potential as gp

    mw = gp.MilkyWayPotential()
    psp = gc.PhaseSpacePosition(pos=[-8.1, 0, 0.02] * u.kpc,
                                vel=[13, 245, 8.] * u.km/u.s)
    orbit = gd.evaluate_orbit(psp.w(units=mw.units), dt=1*u.Myr, t1=0, t2=2*u.Gyr)

    orbit.plot(['x', 'y'])

:class:`~galax.dynamics.Orbit` objects by default assume and use Cartesian
coordinate representations, but these can also be transformed into other
representations, like Cylindrical coordinates. For example, we could
re-represent the orbit in cylindrical coordinates and then plot the orbit in the
"meridional plane"::

    >>> fig = orbit.cylindrical.plot(['rho', 'z'])  # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    fig = orbit.cylindrical.plot(['rho', 'z'])

.. TODO:
.. Or estimate the pericenter, apocenter, and eccentricity of the orbit::

..     >>> orbit.pericenter()
..     <Quantity 8.00498069 kpc>
..     >>> orbit.apocenter()
..     <Quantity 9.30721946 kpc>
..     >>> orbit.eccentricity()
..     <Quantity 0.07522087>

:mod:`galax.potential` ``Potential`` objects and :class:`~galax.dynamics.Orbit`
objects have many more possibilities, so please do check out the narrative
documentation for :mod:`galax.potential` and :mod:`galax.dynamics` if you would
like to learn more!


What else can ``galax`` do?
===========================

This page is meant to demonstrate a few initial things you may want to do with
:mod:`galax`. There is much more functionality that you can discover either
through the :ref:`tutorials <tutorials>` or by perusing the :ref:`user guide
<galax-user-guide>`. Some other commonly-used functionality includes:

* :ref:`Generating simulated "mock" stellar stream models <galax-mockstreams>`
* :ref:`Stellar stream and great circle coordinate systems <galax-coordinates>`
* :ref:`Transformations to action-angle coordinates <galax-actionangle>`
* :ref:`Nonlinear dynamics and chaos indicators <galax-nonlinear-dynamics>`


Where to go from here
=====================

The two places to learn more are the tutorials and the user guide:

* The :ref:`galax-tutorials` are narrative demonstrations of functionality that
  walk through simplified, real-world use cases for the tools available in
  ``galax``.
* The :ref:`galax-user-guide` contains more exhaustive descriptions of all of the
  functions and classes available in ``galax``, and should be treated more like
  reference material.


Bibliography
============

.. bibliography::
    :cited:
