.. _galax-variational-equations:

**************************************************
Variational equations and Lyapunov exponents
**************************************************

A small change :math:`\delta w` to an orbit's initial phase-space position grows
or shrinks along the orbit. To first order it follows the variational equations

.. math::

    \frac{d\,\delta w}{dt} = J(t)\,\delta w,
    \qquad
    J = \begin{pmatrix} 0 & I \\ \partial a/\partial x & \partial a/\partial v \end{pmatrix},

where :math:`a` is the acceleration and :math:`J` is evaluated along the orbit.
The state-transition matrix :math:`M(t) = \partial w(t) / \partial w_0` solves
the same equations with :math:`M = I` at the start. :mod:`galax` integrates
these deviations together with the orbit, under the same solver and the same
error control. It takes the Jacobian from the field with automatic
differentiation, so every potential works without extra code, and so do
single-particle fields with velocity-dependent forces such as dynamical
friction. N-body fields are not supported yet.

For these examples::

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.MilkyWayPotential()
    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Q([8.0, 0, 0.5], "kpc"),
    ...                              p=u.Q([10, 220, 20], "km/s"),
    ...                              t=u.Q(0, "Myr"))
    >>> ts = u.Q(jnp.linspace(0, 500, 6), "Myr")


The state-transition matrix
===========================

:func:`~galax.dynamics.orbit.compute_variational` returns the orbit and the
matrix at each output time:

    >>> orbit, M = gd.orbit.compute_variational(pot, w0, ts)
    >>> M.shape
    (6, 6, 6)

The last two axes are the rows and columns of :math:`M`. Both are in Cartesian
order :math:`(x, y, z, v_x, v_y, v_z)` and in the potential's unit system, here
kpc and kpc/Myr. :math:`M` starts as the identity:

    >>> bool(jnp.allclose(M[0], jnp.eye(6)))
    True

For a conservative force the flow preserves phase-space volume, so
:math:`\det M = 1` at every time:

    >>> bool(jnp.allclose(jnp.linalg.det(M), 1.0, atol=1e-6))
    True

Column :math:`j` of :math:`M(t)` is how the orbit at time :math:`t` moves when
we nudge the :math:`j`-th initial coordinate. For example, to first order a
star started 0.01 kpc further out in :math:`x` is displaced at 500 Myr by

    >>> dw = M[-1] @ jnp.array([0.01, 0, 0, 0, 0, 0])
    >>> dw.shape
    (6,)

The matrix is relative to ``w0`` at its own time ``w0.t``, even when the output
times start later. Integrating backward in time works the same way.

The step-size control covers the deviations as well as the orbit, so it takes
more steps than :func:`~galax.dynamics.orbit.compute_orbit` and the returned
orbit can differ from it slightly at the same tolerances. On a chaotic orbit
:math:`M` grows exponentially, and over a long enough time it overflows. That
shows up as diffrax's "maximum number of solver steps" error, which a larger
``max_steps`` does not fix. For long chaotic runs we use
:func:`~galax.dynamics.orbit.compute_lyapunov` instead.


Deviation vectors
=================

Often we only care about a few directions. ``dw0`` takes them as the columns of
a ``(6, k)`` array, and the result has one column per direction:

    >>> dw0 = jnp.array([[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0]]).T
    >>> _, D = gd.orbit.compute_variational(pot, w0, ts, dw0=dw0)
    >>> D.shape
    (6, 6, 2)
    >>> bool(jnp.allclose(D, M @ dw0, atol=1e-6))
    True


Symplectic solvers
==================

With a symplectic solver the deviations go through the same drift and kick
steps as the orbit. The result is the exact tangent map of the integrator, so
at the end of each step :math:`\det M = 1` to rounding error. Here every
output time is a multiple of ``dt0``, so every one lands on a step:

    >>> import diffrax as dfx
    >>> solver = gd.OrbitSolver(gd.experimental.Leapfrog(),
    ...                         stepsize_controller=dfx.ConstantStepSize())
    >>> _, M = gd.orbit.compute_variational(pot, w0, ts, solver=solver, dt0=0.5)
    >>> bool(jnp.allclose(jnp.linalg.det(M), 1.0, atol=1e-10))
    True

Between steps the solver interpolates the state linearly, and the interpolated
:math:`M` is not exactly symplectic. To check symplecticity, we choose output
times on the step grid.


Lyapunov exponents
==================

:func:`~galax.dynamics.orbit.compute_lyapunov` integrates ``k`` deviation
vectors, re-orthonormalizes them at ``n_renorm`` evenly spaced times, and
returns running estimates of the exponents. ``k=1`` gives the largest
exponent and ``k=6`` gives the full spectrum:

    >>> t_i, lam = gd.orbit.compute_lyapunov(pot, w0, u.Q(2, "Gyr"),
    ...                                      n_renorm=20, k=6)
    >>> lam.shape
    (20, 6)

The exponents are in the inverse of the potential's time unit. For a
conservative force they add up to zero, because :math:`\det M = 1`:

    >>> bool(jnp.abs(lam[-1].sum()) < 1e-6)
    True

For a regular orbit like this one the estimates keep falling toward zero, as
roughly :math:`\ln t / t`. For a chaotic orbit the largest one levels off at a
positive value. The renormalization keeps the deviations from overflowing,
which is why we use this function rather than the raw state-transition matrix
over long times. Each interval still has to be short enough: if the largest
exponent times the interval length nears 700, the deviations overflow within
it, and we need a larger ``n_renorm``.


Derivatives with respect to potential parameters
================================================

The variational equations give derivatives with respect to the initial
conditions. For derivatives with respect to the parameters of a potential, we
differentiate :func:`~galax.dynamics.orbit.compute_orbit` directly with
:func:`jax.jacfwd`.
