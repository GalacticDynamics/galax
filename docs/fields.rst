.. _galax-fields:

**********************
Integrating fields
**********************

A field in :mod:`galax` is a vector field :math:`dy/dt = f(t, y, \mathrm{args})`
on any PyTree state :math:`y`. The state does not have to be a phase-space
position. :func:`galax.fields.integrate_field` integrates it with `diffrax
<https://docs.kidger.site/diffrax/>`_. The design follows the ``fields.py`` of
`streamsculptor
<https://github.com/jnibauer/streamsculptor/blob/main/streamsculptor/fields.py>`_,
where a field is any class with a ``term(t, coords, args)`` method.

For these examples::

    >>> import diffrax as dfx
    >>> import diffraxtra as dfxtra
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.fields as gf

The default unit system of a field is "galactic", so times are in Myr and
lengths in kpc. The default solver is Dopri8 with a minimum step of 0.05 Myr, so
the examples below use rates and frequencies that are slow on that scale.


Exponential decay
=================

:class:`~galax.fields.CustomField` makes a field from a function
``func(t, y, args)`` that returns :math:`dy/dt`. Here is :math:`dy/dt = -k y`
with :math:`k = 0.003` per Myr:

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)
    >>> ts = jnp.linspace(0, 1_000, 5)
    >>> soln = gf.integrate_field(field, jnp.array([1.0]), ts)
    >>> soln.ys.round(4)
    Array([[1.    ],
           [0.4724],
           [0.2231],
           [0.1054],
           [0.0498]], dtype=float64)

The exact solution is :math:`y_0 e^{-kt}`:

    >>> bool(jnp.allclose(soln.ys[:, 0], jnp.exp(-0.003 * ts), rtol=1e-5))
    True


A PyTree state
==============

The state can be any PyTree. Here the state of a harmonic oscillator is the
tuple ``(x, v)``, with :math:`\dot x = v` and :math:`\dot v = -\omega^2 x`,
where :math:`\omega^2 = 0.001` per Myr squared. The period is about 199 Myr, and
the times below cover five periods:

    >>> osc = gf.CustomField(lambda t, y, args: (y[1], -0.001 * y[0]))
    >>> y0 = (jnp.array(1.0), jnp.array(0.0))
    >>> ts = jnp.linspace(0, 1_000, 11)
    >>> soln = gf.integrate_field(osc, y0, ts)
    >>> x, v = soln.ys
    >>> x.round(4)
    Array([ 1.    , -0.9998,  0.9991, -0.9981,  0.9966, -0.9947,  0.9923,
           -0.9895,  0.9863, -0.9827,  0.9787], dtype=float64)

The solution is :math:`x = \cos(\omega t)`:

    >>> bool(jnp.allclose(x, jnp.cos(jnp.sqrt(0.001) * ts), atol=1e-4))
    True

The energy :math:`E = (v^2 + \omega^2 x^2) / 2` is conserved by the exact flow.
It starts at 0.0005 and the integration keeps it there:

    >>> E = 0.5 * v**2 + 0.5 * 0.001 * x**2
    >>> bool(jnp.allclose(E, 0.0005, rtol=1e-5))
    True


Passing ``args``
================

Parameters of the field go in ``args``, which can be any PyTree. It is passed as
the third argument of ``func``, so one field covers many parameter values. This
is the same oscillator with the squared frequency :math:`\omega^2`, which was
0.001 above, taken from the dictionary entry ``k``, here 0.004:

    >>> osc = gf.CustomField(lambda t, y, args: (y[1], -args["k"] * y[0]))
    >>> soln = gf.integrate_field(osc, y0, ts, args={"k": 0.004})
    >>> soln.ys[0].round(4)
    Array([1.    , 0.9991, 0.9966, 0.9923, 0.9863, 0.9787, 0.9694, 0.9584,
           0.9457, 0.9315, 0.9156], dtype=float64)

With :math:`k = 0.004` the period is about 99 Myr, so these samples, 100 Myr
apart, fall close to a full period each time.


Quantities and units
====================

Initial conditions and times can be :class:`unxt.Quantity` objects. Each
Quantity leaf is converted to the field's unit for its dimension and stripped
before the integration:

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)
    >>> y0 = u.Q([1.0], "kpc")
    >>> ts = u.Q(jnp.linspace(0, 1, 5), "Gyr")
    >>> soln = gf.integrate_field(field, y0, ts)
    >>> soln.ts
    Array([   0.,  250.,  500.,  750., 1000.], dtype=float64)
    >>> soln.ys.round(4)
    Array([[1.    ],
           [0.4724],
           [0.2231],
           [0.1054],
           [0.0498]], dtype=float64)

The solution is a plain array in the field's units, so ``soln.ts`` is in Myr and
``soln.ys`` is in kpc. Wrap the array to get a Quantity back:

    >>> u.Q(soln.ys, "kpc")[-1]
    Q([0.04978707], 'kpc')

The ``units`` keyword picks another unit system. In "solarsystem" units, time is
in years and length in AU, so the decay rate of 0.003 is now per year. Here the
initial condition is given in km (about 1 AU) and the times in kyr, and both are
converted to AU and years before the integration:

    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y, units="solarsystem")
    >>> ts = u.Q(jnp.linspace(0, 1, 5), "kyr")
    >>> soln = gf.integrate_field(field, u.Q([1.496e8], "km"), ts)
    >>> soln.ts
    Array([   0.,  250.,  500.,  750., 1000.], dtype=float64)
    >>> soln.ys.round(4)
    Array([[1.    ],
           [0.4724],
           [0.2231],
           [0.1054],
           [0.0498]], dtype=float64)

``soln.ts`` is in years and ``soln.ys`` in AU, and they match the Myr and kpc
numbers above.


Choosing the solver
===================

The ``solver`` keyword takes a :class:`diffraxtra.DiffEqSolver`, which bundles a
diffrax solver, a step size controller and the other options of
:func:`diffrax.diffeqsolve`. ``integrate_field`` always lets the solver choose
the first step (``dt0=None``), so the solver needs an adaptive step size
controller such as :class:`diffrax.PIDController`; a constant-step solver raises
an error. This one is Tsit5 with a tighter tolerance:

    >>> solver = dfxtra.DiffEqSolver(
    ...     dfx.Tsit5(),
    ...     stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8))
    >>> osc = gf.CustomField(lambda t, y, args: (y[1], -0.001 * y[0]))
    >>> y0 = (jnp.array(1.0), jnp.array(0.0))
    >>> ts = jnp.linspace(0, 1_000, 5)
    >>> soln = gf.integrate_field(osc, y0, ts, solver=solver)
    >>> soln.ys[0].round(4)
    Array([ 1.    , -0.0517, -0.9947,  0.1545,  0.9787], dtype=float64)

With ``dense=True`` the solution also carries an interpolation between the
first and last time, which :meth:`diffrax.Solution.evaluate` evaluates at any
:math:`t` in between:

    >>> soln = gf.integrate_field(osc, y0, ts, solver=solver, dense=True)
    >>> x, v = soln.evaluate(250.0)
    >>> bool(jnp.isclose(x, jnp.cos(jnp.sqrt(0.001) * 250.0), atol=1e-5))
    True


Times from a specification
==========================

:func:`galax.fields.parse_time_specification` builds ``ts`` from a start time,
an end time and a number of times, or from other combinations that its
docstring lists. The result is a Quantity, and when its values are floats it
goes straight into :func:`~galax.fields.integrate_field`. Integer times break
the integration, so the example gives the start and end times as floats:

    >>> ts = gf.parse_time_specification(units="galactic",
    ...                                  t0=u.Q(0.0, "Myr"), t1=u.Q(500.0, "Myr"), n_steps=5)
    >>> ts
    Q([  0., 125., 250., 375., 500.], 'Myr')
    >>> field = gf.CustomField(lambda t, y, args: -0.003 * y)
    >>> soln = gf.integrate_field(field, u.Q([1.0], "kpc"), ts)
    >>> soln.ys.round(4)
    Array([[1.    ],
           [0.6873],
           [0.4724],
           [0.3247],
           [0.2231]], dtype=float64)


Orbits
======

An orbit is the same kind of integration. The orbit field
:class:`galax.dynamics.HamiltonianField` takes a potential and has the state
``(q, p)``. It lives in :mod:`galax.dynamics` for now, and
:func:`~galax.fields.integrate_field` takes it like any other field. This is an
orbit in a Kepler potential:

    >>> import galax.dynamics as gd
    >>> import galax.potential as gp
    >>> pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
    >>> field = gd.HamiltonianField(pot)
    >>> y0 = (u.Q([8.0, 0, 0], "kpc"), u.Q([0, 150, 0], "km/s"))
    >>> ts = u.Q(jnp.linspace(0, 1_000, 5), "Myr")
    >>> soln = gf.integrate_field(field, y0, ts)
    >>> q, p = soln.ys
    >>> q.shape, p.shape
    ((5, 3), (5, 3))

The energy is conserved, because the potential does not depend on time:

    >>> E = (0.5 * jnp.sum(p**2, axis=-1)
    ...      + pot.potential(u.Q(q, "kpc"), u.Q(0, "Myr")).ustrip("kpc2/Myr2"))
    >>> bool(jnp.allclose(E, E[0], rtol=1e-5))
    True
