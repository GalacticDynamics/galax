.. _define-new-potential:

*********************************
Defining your own potential class
*********************************

Introduction
============

For the examples below the following imports have already been executed::

    >>> import numpy as np
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd
    >>> from galax.typing import FloatOrIntScalarLike, FloatScalar
    >>> from jaxtyping import Array, Float

========================================
Implementing a new potential with Python
========================================

New Python potentials are implemented by subclassing
:class:`~galax.potential.potential.PotentialBase` and defining functions that
compute (at minimum) the energy and gradient of the potential. We will work
through an example below for adding the `Henon-Heiles potential
<http://en.wikipedia.org/wiki/H%C3%A9non-Heiles_System>`_.

The expression for the potential is:

.. math::

    \Phi(x,y) = \frac{1}{2}(x^2 + y^2) + A\,(x^2 y - \frac{y^3}{3})

With this parametrization, there is only one free parameter (``A``), and the
potential is two-dimensional.

At minimum, the subclass must implement the following methods:

- ``__init__()``
- ``_energy()``
- ``_gradient()``

The ``_energy()`` method should compute the potential energy at a given position
and time. The ``_gradient()`` method should compute the gradient of the
potential. Both of these methods must accept two arguments: a position, and a
time. These internal methods are then called by the
:class:`~galax.potential.potential.PotentialBase` superclass methods
:meth:`~galax.potential.potential.PotentialBase.energy` and
:meth:`~galax.potential.potential.PotentialBase.gradient`. The superclass methods
convert the input position to an array in the unit system of the potential for
fast evaluation. The input to these superclass methods can be
:class:`~astropy.units.Quantity` objects,
:class:`~galax.dynamics.PhaseSpacePosition` objects, or :class:`~numpy.ndarray`.

Because this potential has a parameter, the ``__init__`` method must accept
a parameter argument and store this in the ``parameters`` dictionary attribute
(a required attribute of any subclass). Let's write it out, then work through
what each piece means in detail::

    >>> class CustomHenonHeilesPotential(gp.AbstractPotential):
    ...     A: gp.AbstractParameter = gp.ParameterField(dimensions="dimensionless")
    ...
    ...     def _potential_energy(self, q: Float[Array, "3"], t: FloatOrIntScalarLike) -> FloatScalar:
    ...         A = self.A(t=t)
    ...         x, y = xy
    ...         return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)

The internal energy and gradient methods compute the numerical value and
gradient of the potential. The ``__init__`` method must take a single argument,
``A``, and store this to a parameter dictionary. The expected shape of the
position array (``xy``) passed to the internal ``_energy()`` and ``_gradient()``
methods is always 2-dimensional with shape ``(n_points, n_dim)`` where
``n_points >= 1`` and ``n_dim`` must match the dimensionality of the potential
specified in the initializer. Note that this is different from the shape
expected when calling the public methods ``energy()`` and ``gradient()``!

Let's now create an instance of the class and see how it works. For now, let's
pass in ``None`` for the unit system to designate that we'll work with
dimensionless quantities::

    >>> pot = CustomHenonHeilesPotential(A=1., units=None)

That's it! We now have a potential object with all of the same functionality as
the built-in potential classes. For example, we can integrate an orbit in this
potential (but note that this potential is two-dimensional, so we only have to
specify four coordinate values)::

    >>> w0 = gd.PhaseSpacePosition(q=[0., 0.3], p=[0.38, 0.])
    >>> t = jnp.arange(0, 500, step=0.05)
    >>> orbit = pot.integrate_orbit(w0, t=t)
    >>> fig = orbit.plot(marker=',', linestyle='none', alpha=0.5) # doctest: +SKIP

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    import matplotlib.pyplot as pl
    import numpy as np
    import galax.dynamics as gd
    import galax.potential as gp

    class CustomHenonHeilesPotential(gp.PotentialBase):
        A = gp.PotentialParameter("A")
        ndim = 2
        def _energy(self, xy, t):
            A = self.parameters['A'].value
            x,y = xy.T
            return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)
        def _gradient(self, xy, t):
            A = self.parameters['A'].value
            x,y = xy.T
            grad = np.zeros_like(xy)
            grad[:,0] = x + 2*A*x*y
            grad[:,1] = y + A*(x**2 - y**2)
            return grad

    pot = CustomHenonHeilesPotential(A=1., units=None)
    w0 = gd.PhaseSpacePosition(pos=[0.,0.3],
                               vel=[0.38,0.])
    orbit = gp.Hamiltonian(pot).integrate_orbit(w0, dt=0.05, n_steps=10000)
    fig = orbit.plot(marker=',', linestyle='none', alpha=0.5)

We could also, for example, create a contour plot of equipotentials::

    >>> grid = np.linspace(-1., 1., 100)
    >>> from matplotlib import colors
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1, figsize=(5,5))
    >>> fig = pot.plot_contours(grid=(grid, grid),
    ...                         levels=np.logspace(-3, 1, 10),
    ...                         norm=colors.LogNorm(),
    ...                         cmap='Blues', ax=ax)

.. plot::
    :align: center
    :context: close-figs
    :width: 60%

    from matplotlib import colors
    import matplotlib.pyplot as plt

    grid = np.linspace(-1., 1., 100)
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    fig = pot.plot_contours(grid=(grid,grid), cmap='Blues',
                            levels=np.logspace(-3, 1, 10),
                            norm=colors.LogNorm(), ax=ax)

=====================================
Adding a custom potential with Cython
=====================================

Adding a new Cython potential class is a little more involved as it requires
writing C-code and setting it up properly to compile when the code is built.
For this example, we'll work through how to define a new C-implemented potential
class representation of a Keplerian (point-mass) potential. Because this example
requires using Cython to build code, we provide a separate
`demo GitHub repository <https://github.com/adrn/gala-cpotential-demo>`_ with an
implementation of this potential with a demonstration of a build system that
successfully sets up the code.

New Cython potentials are implemented by subclassing
:class:`~galax.potential.potential.CPotentialBase`, subclassing
:class:`~galax.potential.potential.CPotentialWrapper`, and defining C functions
that compute (at minimum) the energy and gradient of the potential. This
requires creating (at minimum) a Cython file (.pyx), a C header file (.h), and a
C source file (.c).
