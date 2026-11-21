.. _galax-potential-modules:

*******************************
Per-potential function modules
*******************************

Each built-in potential has a module of plain functions next to its class. The
module is named for the class in lower case, without ``Potential``, so
:class:`~galax.potential.PlummerPotential` has :mod:`galax.potential.plummer`.
The functions compute the same quantities as the class methods. They take bare
arrays instead of quantities, so they can be used inside a ``jax.jit`` or
``jax.vmap`` of your own code. This page calls these functions kernels, and
"function" and "kernel" mean the same thing here.

For these examples::

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp


The contract
============

A kernel takes the parameters ``p`` first, then the position. ``p`` is the
dictionary returned by ``pot.params(t)``, evaluated at the time of interest.
Kernels work on one point at a time, so ``r`` and ``t`` are scalars and ``xyz``
has shape ``(3,)``. To evaluate many points, use ``jax.vmap``.

.. list-table::
    :header-rows: 1

    * - Module
      - Functions
      - Arguments
    * - Spherical
      - ``potential``, ``density``, ``laplacian``, ``mass_enclosed``,
        ``dpotential_dr``, ``d2potential_dr2``
      - ``(p, r)``
    * - Spherical
      - ``gradient``, ``hessian``
      - ``(p, xyz)``
    * - Non-spherical
      - ``potential``, ``density``, ``laplacian``, ``gradient``, ``hessian``
      - ``(p, xyz)``
    * - Monari 2016 bar
      - the non-spherical functions
      - ``(p, xyz, t)``

Positions and parameters are bare arrays in the unit system of the potential,
and the caller keeps track of the units (see `Units`_ below). A module may export
more than these functions, for example ``mass_total`` or ``rho0_of_m``.

:func:`galax.potential.plummer.potential` is a different function from
:func:`galax.potential.potential`. The first takes ``(p, r)`` and returns a
number. The second is dispatched on the potential and takes ``(pot, xyz, t)``
with units.

The class methods and the module functions are the same where the module
function is hand-written. For the functions that a shared helper builds from
others, such as the ``gradient`` of a non-spherical potential by autodiff or the
``laplacian`` by Poisson's equation, the classes keep the base-class autodiff,
and a test checks that the two agree.

At the origin the spherical functions follow one rule. The ``gradient`` is
exactly 0. The ``hessian`` is ``laplacian(p, 0) / 3`` times the identity, which
is :math:`\Phi''(0)` times the identity for a cored profile. For a cusp the
diagonal is ``+inf`` and the off-diagonal entries are 0. It is never NaN. The
functions of ``r`` expect ``r > 0``, and several of them return NaN at exactly
``r = 0``. The class method ``pot.hessian`` at exactly the origin keeps its
older values and can differ from the module function. To differentiate at the
origin, differentiate the ``hessian``: ``jax.jacfwd`` of the ``gradient`` at 0
is 0, because the gradient is set to 0 there. The non-spherical functions have
no such rule, so evaluate them away from the origin.


An example
==========

Build a Plummer potential and take its parameters at :math:`t = 0`:

    >>> pot = gp.PlummerPotential(
    ...     m_tot=u.Quantity(1e11, "Msun"), r_s=u.Quantity(5.0, "kpc"),
    ...     units="galactic")
    >>> p = pot.params(0.0)
    >>> sorted(p)
    ['G', 'm_tot', 'r_s']

The kernel evaluates one radius, in kpc, and returns the potential in
:math:`\mathrm{kpc}^2/\mathrm{Myr}^2`:

    >>> phi = gp.plummer.potential(p, 2.0)
    >>> phi.shape
    ()

``jax.vmap`` maps it over an array of radii. ``p`` is shared, so its axis is
``None``:

    >>> r = jnp.array([1.0, 2.0, 4.0])
    >>> phis = jax.vmap(gp.plummer.potential, in_axes=(None, 0))(p, r)
    >>> phis.shape
    (3,)

These match the potential of the class evaluated on the same points on the
x axis:

    >>> xyz = u.Quantity(jnp.stack([r, 0 * r, 0 * r], axis=-1), "kpc")
    >>> bool(jnp.allclose(phis, pot.potential(xyz, 0.0).value))
    True


Units
=====

The module functions take bare arrays in the unit system of the potential,
``pot.units``, so that they stay cheap under ``jit``, ``vmap`` and ``grad``.
``pot.params(t)`` returns the parameters, and ``G``, in that system. For the
galactic potential above the numbers are in kpc, Myr and Msun:

    >>> float(p["r_s"])
    5.0
    >>> round(float(p["G"]) * 1e12, 3)
    4.499

Pass positions and times in the same system. A position in pc given to this
potential would be read as kpc, and nothing checks it, so all the numbers need
to come from one unit system. ``pot.params(t, ustrip=False)`` returns the same
values as quantities, which is useful for inspection:

    >>> str(pot.params(0.0, ustrip=False)["m_tot"].unit)
    'solMass'

If you build ``p`` by hand, pick one unit system and put ``G`` in it as well.
For dimensionless N-body units that means ``G = 1`` and masses and lengths in
those units.

The class methods take quantities and coordinate objects and convert for you, so
use them when the inputs carry units. The module functions are for code that
already holds arrays in one system, such as integrators, fits and N-body
kernels.


The modules
===========

Spherical
---------

``burkert``, ``gaussian``, ``gnfw``, ``hardcutoffnfw``, ``hernquist``,
``isochrone``, ``jaffe``, ``kepler``, ``nfw``, ``plummer``,
``powerlawcutoff``, ``stoneostriker15``, ``zhao``.

Non-spherical
-------------

``kuzmin``, ``kuzminkutuzov``, ``leesutotriaxialnfw``, ``longmuralibar``,
``miyamotonagai``, ``monarietal2016bar``, ``satoh``.
