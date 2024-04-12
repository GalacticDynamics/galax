.. include:: ../references.txt

.. module:: galax.potential

********************************************
Gravitational potentials (`galax.potential`)
********************************************

Introduction
============

This subpackage provides a number of classes for working with parametric models
of gravitational potentials. There are a number of built-in potentials
implemented in C and Cython (for speed), and there are base classes that allow
for easy creation of `new custom potential classes <define-new-potential.html>`_
in pure Python or by writing custom C/Cython extensions. The ``Potential``
objects have convenience methods for computing common dynamical quantities, for
example: potential energy, spatial gradient, density, or mass profiles. These
are particularly useful in combination with the :mod:`~galax.integrate` and
:mod:`~galax.dynamics` subpackages.

.. TODO: uncomment
.. Also defined in this subpackage are a set of reference frames which can be used
.. for numerical integration of orbits in non-static reference frames. See the page
.. on :ref:`hamiltonian-reference-frames` for more information. ``Potential``
.. objects can be combined with a reference frame and stored in a
.. :class:`~galax.potential.hamiltonian.Hamiltonian` object that provides an easy
.. interface to numerical orbit integration.

For the examples below the following imports have already been executed::

    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> from galax.units import galactic, solarsystem, dimensionless


Getting Started: Built-in Methods of Potential Classes
======================================================

Any of the built-in ``Potential`` classes are initialized by passing in keyword
argument parameter values as :class:`~astropy.units.Quantity` objects or as
numeric values in a specified unit system. To see what parameters are available
for a given potential, check the documentation for the individual classes below.
You must also specify a :class:`~galax.units.UnitSystem` when initializing a
potential.  A unit system is a set of non-reducible units that define (at
minimum) the length, mass, time, and angle units. A few common unit systems are
built in to the package (e.g., ``galactic``, ``solarsystem``,
``dimensionless``). For example, to create an object to represent a Kepler
potential (point mass) at the origin with mass = 1 solar mass, we would
instantiate a :class:`~gala.potential.KeplerPotential` object:::

    >>> ptmass = gp.KeplerPotential(m=1.*u.Msun, units=solarsystem)
    >>> ptmass
    KeplerPotential(
      units=UnitSystem(AU, yr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[])
    )

If you pass in parameters with different units, they will be converted to the
specified unit system::

    >>> gp.KeplerPotential(m=1047.6115*u.Mjup, units=solarsystem)
    KeplerPotential(
      units=UnitSystem(AU, yr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[])
    )

If no units are specified for a parameter (i.e. a parameter value is passed in
as a Python numeric value or array), it is assumed to be in the specified
:class:`~galax.units.UnitSystem`::

    >>> gp.KeplerPotential(m=1., units=solarsystem)
    KeplerPotential(
      units=UnitSystem(AU, yr, solMass, rad),
      m=ConstantParameter(unit=Unit("solMass"), value=f64[])
    )

The potential classes work well with the :mod:`astropy.units` framework, but to
ignore units you can use the :class:`~galax.units.DimensionlessUnitSystem` or
pass :obj:`~None` as the unit system::

    >>> gp.KeplerPotential(m=1., units=None)
    KeplerPotential(
      units=DimensionlessUnitSystem(),
      m=ConstantParameter(unit=Unit(dimensionless), value=f64[])
    )

All of the built-in potential objects have defined methods to evaluate the
potential energy and the gradient/acceleration at a given position or array of
positions. For example, to evaluate the potential energy at the 3D position
``(x, y, z) = (1, -1, 0) AU``::

    >>> ptmass.potential_energy([1., -1., 0.] * u.au, t=0)
    Array(-27.91440236, dtype=float64)

These functions also accept both :class:`~astropy.units.Quantity` objects or
plain :class:`~numpy.ndarray`-like objects (in which case the position is
assumed to be in the unit system of the potential)::

    >>> ptmass.potential_energy(jnp.asarray([1., -1., 0.]), t=0)
    Array(-27.91440236, dtype=float64)

This also works for multiple positions by passing in a 2D position (but see
:ref:`conventions` for a description of the interpretation of different axes)::

    >>> pos = jnp.array([[1., -1. ,0],
    ...                  [2., 3., 0]])
    >>> ptmass.potential_energy(pos * u.au, t=0)
    Array([-27.91440236, -10.94892941], dtype=float64)

We can also compute the gradient or acceleration::

    >>> ptmass.gradient([1., -1., 0] * u.au, t=0)
    Array([ 13.95720118, -13.95720118,   0.        ], dtype=float64)
    >>> ptmass.acceleration([1., -1., 0] * u.au, t=0)
    Array([-13.95720118,  13.95720118,  -0.        ], dtype=float64)

Most of the potential objects also have methods implemented for computing the
corresponding mass density and the Hessian of the potential (the matrix of 2nd
derivatives) at given locations. For example, with the
:class:`~galax.potential.HernquistPotential`, we can evaluate both the mass
density and Hessian at the position ``(x, y, z) = (1, -1, 0) kpc``::

    >>> pot = gp.HernquistPotential(m=1E9*u.Msun, c=1.*u.kpc, units=galactic)
    >>> pot.density([1., -1., 0] * u.kpc, t=0)
    Array(7997938.82200887, dtype=float64)
    >>> pot.hessian([1., -1., 0] * u.kpc, t=0)
    Array([[-4.68187913e-05,  5.92578618e-04,  0.00000000e+00],
           [ 5.92578618e-04, -4.68187913e-05,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  5.45759827e-04]],      dtype=float64)

.. Another useful method is :meth:`~galax.potential.PotentialBase.mass_enclosed`,
.. which numerically estimates the mass enclosed within a spherical shell defined
.. by the specified position. This numerically estimates :math:`\frac{d \Phi}{d r}`
.. along the vector pointing at the specified position and estimates the enclosed
.. mass simply as :math:`M(<r)\approx\frac{r^2}{G} \frac{d \Phi}{d r}`. This
.. function can be used to compute, for example, a mass profile::

..     >>> pot = gp.NFWPotential(m=1E11*u.Msun, r_s=20.*u.kpc, units=galactic)
..     >>> pos = jnp.zeros((3,100)) * u.kpc
..     >>> pos[0] = jnp.logspace(jnp.log10(20./100.), jnp.log10(20*100.), pos.shape[1]) * u.kpc
..     >>> m_profile = pot.mass_enclosed(pos)
..     >>> plt.loglog(pos[0], m_profile, marker='')
..     >>> plt.xlabel("$r$ [{}]".format(pos.unit.to_string(format='latex')))
..     >>> plt.ylabel("$M(<r)$ [{}]".format(m_profile.unit.to_string(format='latex')))

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    import astropy.units as u
    import numpy as np
    import gala.potential as gp
    from gala.units import galactic, solarsystem
    import matplotlib.pyplot as plt

    pot = gp.NFWPotential(m=1E11*u.Msun, r_s=20.*u.kpc, units=galactic)
    pos = jnp.zeros((3,100)) * u.kpc
    pos[0] = jnp.logspace(jnp.log10(20./100.), jnp.log10(20*100.), pos.shape[1]) * u.kpc
    m_profile = pot.mass_enclosed(pos)

    plt.figure()
    plt.loglog(pos[0], m_profile, marker='')
    plt.xlabel("$r$ [{}]".format(pos.unit.to_string(format='latex')))
    plt.ylabel("$M(<r)$ [{}]".format(m_profile.unit.to_string(format='latex')))
    plt.tight_layout()


.. Plotting Equipotential and Isodensity contours
.. ==============================================

.. Potential objects provide specialized methods for visualizing the isopotential
.. (:meth:`~galax.potential.PotentialBase.plot_contours`) or isodensity
.. (:meth:`~galax.potential.PotentialBase.plot_density_contours`) contours
.. of a given potential object. These methods plot either 1D slices or 2D contour
.. plots of isopotentials and isodensities. To plot a 1D slice over the dimension
.. of interest, pass in a grid of values for that dimension and numerical values
.. for the others. For example, to make a 1D plot of the potential value as a
.. function of :math:`x` position at :math:`y=0, z=1`::

..     >>> p = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
..     >>> fig, ax = plt.subplots()
..    >>> plot_contours(p, grid=(jnp.linspace(-15,15,100), 0., 1.), marker='', ax=ax)
..    >>> E_unit = p.units['energy'] / p.units['mass']
..    >>> ax.set_xlabel("$x$ [{}]".format(p.units['length'].to_string(format='latex')))
..    >>> ax.set_ylabel("$\Phi(x,0,1)$ [{}]".format(E_unit.to_string(format='latex')))

.. .. plot::
..     :align: center
..     :context: close-figs
..     :width: 90%

..     pot = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)

..     fig, ax = plt.subplots(1,1)
..     pot.plot_contours(grid=(jnp.linspace(-15,15,100), 0., 1.), marker='', ax=ax)
..     E_unit = pot.units['energy'] / pot.units['mass']
..     ax.set_xlabel("$x$ [{}]".format(pot.units['length'].to_string(format='latex')))
..     ax.set_ylabel("$\Phi(x,0,1)$ [{}]".format(E_unit.to_string(format='latex')))
..     fig.tight_layout()

.. To instead make a 2D contour plot over :math:`x` and :math:`z` along with
.. :math:`y=0`, pass in a 1D grid of values for :math:`x` and a 1D grid of values
.. for :math:`z` (the meshgridding is taken care of internally). Here, we choose to
.. draw on a pre-defined ``matplotlib`` axes object so we can set the labels and
.. aspect ratio of the plot::

..     >>> fig,ax = plt.subplots(1, 1, figsize=(12, 4))
..     >>> x = jnp.linspace(-15, 15, 100)
..     >>> z = jnp.linspace(-5, 5, 100)
..     >>> p.plot_contours(grid=(x, 1., z), ax=ax)
..     >>> ax.set_xlabel("$x$ [kpc]")
..     >>> ax.set_ylabel("$z$ [kpc]")

.. .. plot::
..     :align: center
..     :context: close-figs
..     :width: 60%

..     x = jnp.linspace(-15,15,100)
..     z = jnp.linspace(-5,5,100)

..     fig,ax = plt.subplots(1, 1, figsize=(12,4))
..     pot.plot_contours(grid=(x, 1., z), ax=ax)
..     ax.set_xlabel("$x$ [{}]".format(pot.units['length'].to_string(format='latex')))
..     ax.set_ylabel("$z$ [{}]".format(pot.units['length'].to_string(format='latex')))
..     fig.tight_layout()

.. Saving / loading potential objects
.. ==================================

.. Potential objects can be `pickled
.. <https://docs.python.org/3/library/pickle.html>`_ and can therefore be stored
.. for later use. However, pickles are saved as binary files. It may be useful to
.. save to or load from text-based specifications of Potential objects. This can be
.. done with the :meth:`~galax.potential.PotentialBase.save` method and the
.. :func:`~galax.potential.load` function, which serialize and de-serialize
.. (respectively) a ``Potential`` object to a `YAML <https://yaml.org/>`_ file::

..     >>> from gala.potential import load
..     >>> pot = gp.NFWPotential(m=6E11*u.Msun, r_s=20.*u.kpc,
..     ...                       units=galactic)
..     >>> pot.save("potential.yml")
..     >>> load("potential.yml")
..     <NFWPotential: m=6.00e+11, r_s=20.00, a=1.00, b=1.00, c=1.00 (kpc,Myr,solMass,rad)>

.. Exporting potentials as ``sympy`` expressions
.. =============================================

.. Most of the potential classes can be exported to a :mod:`~sympy` expression that can
.. be used to manipulate or evaluate the form of the potential. To access this
.. functionality, the potential classes have a
.. :meth:`~galax.potential.PotentialBase.to_sympy` classmethod (note: this
.. requires :mod:`sympy` to be installed):

.. .. doctest-requires:: sympy

..     >>> expr, vars_, pars = gp.LogarithmicPotential.to_sympy()
..     >>> str(expr)
..     '0.5*v_c**2*log(r_h**2 + z**2/q3**2 + y**2/q2**2 + x**2/q1**2)'

.. This method also returns a dictionary containing the coordinate variables used
.. in the expression as ``sympy`` symbols, here defined as ``vars_``:

.. .. doctest-requires:: sympy

..     >>> vars_
..     {'x': x, 'y': y, 'z': z}

.. A second dictionary containing the potential parameters as :mod:`sympy` symbols is
.. also returned, here defined as ``pars``:

.. .. doctest-requires:: sympy

..     >>> pars
..     {'v_c': v_c, 'r_h': r_h, 'q1': q1, 'q2': q2, 'q3': q3, 'phi': phi, 'G': G}

.. The expressions and variables returned can be used to perform operations on the
.. potential expression. For example, to create a :mod:`sympy` expression for the
.. gradient of the potential:

.. .. doctest-requires:: sympy

..     >>> import sympy as sy
..     >>> grad = sy.derive_by_array(expr, list(vars_.values()))
..     >>> grad[0]  # dPhi/dx
..     1.0*v_c**2*x/(q1**2*(r_h**2 + z**2/q3**2 + y**2/q2**2 + x**2/q1**2))


Using gala.potential
====================
More details are provided in the linked pages below:

.. toctree::
   :maxdepth: 1

   define-new-potential
   compositepotential

.. .. toctree::
..    :maxdepth: 1

..    define-new-potential
..    compositepotential
..    origin-rotation
..    hamiltonian-reference-frames
..    scf


API
===

.. automodapi:: galax.potential

.. TODO: uncomment
.. .. automodapi:: gala.potential.frame.builtin

.. TODO: uncomment
.. .. automodapi:: gala.potential.hamiltonian
