.. _galax-time-dependent-parameters:

***********************************
Time-dependent potential parameters
***********************************

Every parameter of a :mod:`galax` potential is a *callable of time*. A mass you
pass as a plain :class:`~unxt.Quantity` is wrapped into one that ignores its argument;
anything that varies is the same thing with a different rule inside. That one
idea covers all the cases below, and it is why a potential whose mass grows
needs no special potential class.

For these examples::

    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import galax.potential as gp

A constant
==========

The common case needs no ceremony: pass a :class:`~unxt.Quantity` and it becomes a
:class:`~galax.potential.params.ConstantParameter`.

    >>> pot = gp.HernquistPotential(
    ...     m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic")
    >>> type(pot.m_tot).__name__
    'ConstantParameter'

Being a callable of time is not a fiction even here -- it answers, and gives
the same answer at every time:

    >>> pot.m_tot(u.Q(0.0, "Gyr"))
    Q(1.e+12, 'solMass')

    >>> pot.m_tot(u.Q(5.0, "Gyr"))
    Q(1.e+12, 'solMass')

A function of time
==================

For a parameter you can write in closed form, use
:class:`~galax.potential.params.CustomParameter`. The function takes the time first,
and any data it needs after that:

    >>> def growing(t, m0, rate):
    ...     return m0 + rate * t

    >>> m_of_t = gp.params.CustomParameter(
    ...     func=growing, args=(u.Q(1e12, "Msun"), u.Q(1e11, "Msun/Gyr")))

    >>> m_of_t(u.Q(2.0, "Gyr"))
    Q(1.2e+12, 'solMass')

It drops into a potential like any other parameter, and the potential is now
time-dependent:

    >>> pot = gp.HernquistPotential(
    ...     m_tot=m_of_t, r_s=u.Q(5.0, "kpc"), units="galactic")
    >>> pot.m_tot(u.Q(2.0, "Gyr"))
    Q(1.2e+12, 'solMass')

.. _galax-parameters-args:

Why the data goes in ``args``
-----------------------------

You could have closed over ``m0`` and ``rate`` instead::

    def growing(t):                     # don't
        return u.Q(1e12, "Msun") + u.Q(1e11, "Msun/Gyr") * t

It gives the same numbers, and it is worse in two ways that do not announce
themselves. ``CustomParameter.func`` is a *static* field -- it is hashed, not
traced -- so arrays captured in a closure are invisible to JAX. They are not
leaves of the parameter's pytree.

That costs a recompile: a closure is hashed by identity, so rebuilding one
over the *same* data is a fresh :func:`jax.jit` cache entry. And it loses data:
:func:`equinox.tree_serialise_leaves` writes nothing for a closed-over array, so
saving a potential silently drops it. Put data in ``args`` (positionally) or
``kwargs`` (by name) and both work, because both are ordinary fields.

Keywords given at the call site override stored ones, so ``kwargs`` is a way
to supply defaults:

    >>> def ramp(t, *, m0, rate):
    ...     return m0 + rate * t

    >>> p = gp.params.CustomParameter(
    ...     func=ramp, kwargs={"m0": u.Q(1e12, "Msun"), "rate": u.Q(1e11, "Msun/Gyr")})
    >>> p(u.Q(2.0, "Gyr"))
    Q(1.2e+12, 'solMass')

    >>> p(u.Q(2.0, "Gyr"), rate=u.Q(0.0, "Msun/Gyr"))
    Q(1.e+12, 'solMass')

Everything in ``args`` and ``kwargs`` passes through JAX, so it must be a JAX
type: arrays, :class:`~unxt.Quantity`, ``None``, Python scalars, and nested
tuples or dicts of those. A string, a function, or a unit object belongs in a
closure over ``func`` instead -- which is what the static field is still for.

Tabulated values
================

When the parameter is known at a set of times rather than in closed form --
measurements, or the output of another simulation -- interpolate between
them. The grid and the table are *data*, so they go in ``args`` exactly like
anything else; nothing about this case is special.

:mod:`interpax` provides the interpolators, and which one you want is your
choice rather than :mod:`galax`'s. A cubic is the usual answer because it is
:math:`C^1`, so an integrator sees no kink at a knot:

    >>> import interpax

    >>> def tabulated(t, ts, values):
    ...     tq = u.ustrip("Gyr", t)
    ...     grid, table = u.ustrip("Gyr", ts), u.ustrip("Msun", values)
    ...     return u.Q(
    ...         interpax.interp1d(tq, grid, table, method="cubic2",
    ...                           extrap=(table[0], table[-1])),
    ...         "Msun")

    >>> ts = u.Q(jnp.asarray([0.0, 1.0, 2.0, 3.0]), "Gyr")
    >>> ms = u.Q(jnp.asarray([1.0, 1.4, 1.9, 2.1]) * 1e12, "Msun")
    >>> m_of_t = gp.params.CustomParameter(func=tabulated, args=(ts, ms))

On a knot it returns that knot's value:

    >>> m_of_t(u.Q(1.0, "Gyr"))
    Q(1.4e+12, 'solMass')

and between knots it interpolates:

    >>> m_of_t(u.Q(1.5, "Gyr"))
    Q(1.6625e+12, 'solMass')

The ``extrap`` pair above *clamps* to the end values rather than continuing
the edge cubic. That is a choice worth making deliberately: a table says
nothing about what happens beyond its last entry, and an extrapolated cubic
will invent something -- for a quantity feeding a potential, fast and
silently.

    >>> m_of_t(u.Q(99.0, "Gyr"))
    Q(2.1e+12, 'solMass')

Swap ``method`` for a different rule -- ``"akima"`` or ``"monotonic"`` when
the table should not overshoot between knots, ``"linear"`` when a kink is
acceptable -- or drop ``extrap`` to extrapolate. Because the table travels
in ``args`` it stays a pytree leaf throughout, so the parameter still
serialises, still avoids a recompile per rebuild, and is still
differentiable with respect to the tabulated values:

    >>> import jax
    >>> g = jax.grad(
    ...     lambda v: u.ustrip("Msun",
    ...         gp.params.CustomParameter(func=tabulated, args=(ts, v))(u.Q(1.5, "Gyr"))))
    >>> bool(jnp.any(jnp.abs(u.ustrip("Msun", g(ms))) > 0))
    True

Writing your own
================

:class:`~galax.potential.params.CustomParameter` is the quick route. When a
parameter has its own named quantities, its own validation, or is something
you will reuse, a small class of your own is clearer -- and it is no harder,
because a parameter is just a callable of time with typed fields.

    >>> import functools as ft
    >>> import equinox as eqx
    >>> import jax
    >>> from unxt.quantity import AllowValue
    >>> from galax.potential.params import AbstractParameter

    >>> class ExponentialParameter(AbstractParameter):
    ...     '''A quantity growing as exp(t / t_grow).'''
    ...
    ...     m0: u.AbstractQuantity = eqx.field(converter=u.Q.from_)
    ...     t_grow: u.AbstractQuantity = eqx.field(converter=u.Q.from_)
    ...
    ...     @ft.partial(jax.jit, static_argnames=("ustrip",))
    ...     def __call__(self, t, *, ustrip=None, **_):
    ...         out = self.m0 * jnp.exp(
    ...             u.ustrip("Gyr", t) / u.ustrip("Gyr", self.t_grow))
    ...         return out if ustrip is None else u.ustrip(AllowValue, ustrip, out)

    >>> m_of_t = ExponentialParameter(m0=u.Q(1e12, "Msun"), t_grow=u.Q(5.0, "Gyr"))
    >>> m_of_t(u.Q(5.0, "Gyr"))
    Q(2.71828183e+12, 'solMass')

``ustrip`` is part of the contract every parameter honours -- a caller can
ask for a bare number in a unit of their choosing, and the ``AllowValue``
overload is what lets the same line work whether ``out`` carries units or
is already a plain array:

    >>> round(float(m_of_t(u.Q(5.0, "Gyr"), ustrip=u.unit("Msun"))) / 1e12, 6)
    2.718282

The fields are ordinary :mod:`equinox` fields, so they are pytree leaves with all
the properties ``args`` buys above -- and they are *named*, which
``args[1]`` is not:

    >>> m_of_t.t_grow
    Q(5., 'Gyr')

A class also gives you a type to check for, which a factory does not:

    >>> isinstance(m_of_t, ExponentialParameter)
    True

Use whichever fits. A one-off rule is a :class:`~galax.potential.params.CustomParameter`;
something with a name, invariants, and a second user is a class.

Evaluating every parameter at once
==================================

A potential keeps its parameters as the objects described above, and
``param_objs`` maps each name to its object:

    >>> pot = gp.HernquistPotential(
    ...     m_tot=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), units="galactic")
    >>> list(pot.param_objs)
    ['m_tot', 'r_s']

To get the values at a time, call ``params``. It evaluates every parameter at
that time and returns a dictionary with one entry per parameter, plus ``"G"``
from the potential's constants:

    >>> sorted(pot.params(u.Q(0.0, "Gyr")))
    ['G', 'm_tot', 'r_s']

By default each value is a plain array in the potential's unit system, which is
the form the potential's own jitted functions take, so this dictionary is also
the quickest way to see the numbers a potential is using at a given time. Pass
``ustrip=False`` to keep the units:

    >>> pot.params(u.Q(0.0, "Gyr"), ustrip=False)["m_tot"]
    Q(1.e+12, 'solMass')

A bare number is read as a time in the potential's time unit, which is Myr in
the ``"galactic"`` unit system used here.

For a time-dependent parameter the value follows the time. Here the mass is the
growing :class:`~galax.potential.params.CustomParameter` from above:

    >>> pot = gp.HernquistPotential(
    ...     m_tot=gp.params.CustomParameter(
    ...         func=growing, args=(u.Q(1e12, "Msun"), u.Q(1e11, "Msun/Gyr"))),
    ...     r_s=u.Q(5.0, "kpc"), units="galactic")
    >>> [round(float(pot.params(u.Q(t, "Gyr"))["m_tot"]) / 1e12, 3) for t in (0.0, 2.0)]
    [1.0, 1.2]

For a composite potential the result holds one such dictionary per component,
keyed by the component's name:

    >>> bulge = gp.KeplerPotential(m_tot=u.Q(1e10, "Msun"), units="galactic")
    >>> comp = gp.CompositePotential(halo=pot, bulge=bulge)
    >>> {name: sorted(values) for name, values in comp.params(u.Q(0.0, "Gyr")).items()}
    {'halo': ['G', 'm_tot', 'r_s'], 'bulge': ['G', 'm_tot']}

An orbit in a changing potential
================================

A time-dependent parameter changes the orbits too. Here a triaxial NFW halo
flattens along :math:`y`: its ``q1`` goes from 1 to 0.6 over 0.7 Gyr, a
:class:`~galax.potential.params.CustomParameter` carrying its end value and
duration in ``args``:

    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd

    >>> def flattening(t, q_end, t_end):
    ...     frac = jnp.clip(u.ustrip("Gyr", t) / u.ustrip("Gyr", t_end), 0.0, 1.0)
    ...     return u.Q(1.0 + (q_end - 1.0) * frac, "")

    >>> q1 = gp.params.CustomParameter(func=flattening, args=(0.6, u.Q(0.7, "Gyr")))
    >>> q1(u.Q(0.35, "Gyr"))
    Q(0.8, '')

    >>> def halo(q1):
    ...     return gp.TriaxialNFWPotential(
    ...         m=u.Q(5.4e11, "Msun"), r_s=u.Q(15.0, "kpc"),
    ...         q1=q1, q2=u.Q(1.0, ""), units="galactic")

The same starting point, integrated in the static halo and in the flattening one:

    >>> w0 = gc.PhaseSpacePosition(q=u.Q([8.0, 0, 0], "kpc"), p=u.Q([0, 135, 0], "km/s"))
    >>> ts = u.Q(jnp.linspace(0.0, 0.7, 400), "Gyr")
    >>> orbit_static = gd.evaluate_orbit(halo(u.Q(1.0, "")), w0, ts)
    >>> orbit_flat = gd.evaluate_orbit(halo(q1), w0, ts)

    >>> round(float(u.ustrip("kpc", orbit_flat.q.y[-1] - orbit_static.q.y[-1])), 1)
    -0.9

Drawn as they are integrated, the static orbit is blue and the flattening one is
dotted red; the shaded ellipses sketch the halo's shape at that time.

.. raw:: html

   <iframe src="_static/flattening_orbit.html" loading="lazy"
           title="An orbit in a static triaxial NFW halo (blue) and in the same halo flattening with time (dotted red)"
           style="display:block; width:100%; max-width:560px; aspect-ratio:1000/840; border:0; margin:1em auto;"></iframe>

The animation is made by ``scripts/make_flattening_orbit.py``, which runs this
example.
