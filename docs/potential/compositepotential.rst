.. _galax-compositepotential:

*************************************************
Creating a composite (multi-component ) potential
*************************************************

Potential objects can be combined into more complex *composite* potentials using
:class:`~galax.potential.CompositePotential`. This class operates like a
Python dictionary in that each component potential must be named, and the
potentials can either be passed in to the initializer or added after the
composite potential container is already created.

With either class, interaction with the class (e.g., by calling methods) is
identical to the individual potential classes. To compose potentials with unique
but arbitrary names, you can also simply add pre-defined potential class
instances::

    >>> import jax.numpy as jnp
    >>> import galax.potential as gp
    >>> from galax.units import galactic
    >>> disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    >>> bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
    >>> pot = disk + bulge
    >>> print(pot.__class__.__name__)
    CompositePotential
    >>> list(pot.keys())  # doctest: +SKIP
    ['c655f07d-a1fe-4905-bdb2-e8a202d15c81',
     '8098cb0b-ebad-4388-b685-2f93a874296e']

The two components are assigned unique names and composed into a
:class:`~galax.potential.CompositePotential` instance.

Alternatively, the potentials can be composed directly into the object by
treating it like an immutable dictionary. This allows you to specify the keys or
names of the components in the resulting
:class:`~galax.potential.CompositePotential` instance::

    >>> disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    >>> bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
    >>> pot = gp.CompositePotential(disk=disk, bulge=bulge)
    >>> list(pot.keys())
    ['disk', 'bulge']

The order of insertion is preserved, and sets the order that the potentials are
called. In the above example, the disk potential would always be called first
and the bulge would always be called second.

The resulting potential object has all of the same properties as individual
potential objects::

    >>> q = jnp.asarray([1., -1., 0.])
    >>> pot.potential_energy(q, t=0)
    Array(-0.12887588, dtype=float64)
    >>> pot.acceleration(q, t=0)
    Array([-0.02270876,  0.02270876, -0.        ], dtype=float64)

..    >>> grid = jnp.linspace(-3., 3., 100)
..    >>> fig = pot.plot_contours(grid=(grid, 0, grid))

.. plot::
    :align: center
    :width: 60%

    import jax.numpy as jnp
    import gala.dynamics as gd
    import galax.potential as gp
    from galax.units import galactic

    disk = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.27, units=galactic)
    bulge = gp.HernquistPotential(m=3E10, c=0.7, units=galactic)
    pot = gp.CompositePotential(disk=disk, bulge=bulge)

    grid = jnp.linspace(-3.,3.,100)
    fig = pot.plot_contours(grid=(grid,0,grid))
