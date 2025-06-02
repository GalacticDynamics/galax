"""Orbit interfacing with `diffrax`.

This is private API.

"""

__all__: list[str] = []


import diffrax as dfx

import coordinax as cx
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity as FastQ

import galax.coordinates as gc
from .interp import PhaseSpaceInterpolation
from .orbit import Orbit


@gc.PhaseSpaceCoordinate.from_.dispatch
def from_(
    cls: type[gc.PhaseSpaceCoordinate],
    soln: dfx.Solution,
    /,
    *,
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    units: u.AbstractUnitSystem,  # not dispatched on, but required
    unbatch_time: bool = True,
) -> gc.PhaseSpaceCoordinate:
    r"""Convert a solution to a phase-space position.

    This assumes that the solution is shaped as:
    - t0, t1: (*tbatch,)
    - ts: (*tbatch, [T])
    - *ys: (*tbatch, [T], *ybatch, 3)

    Examples
    --------
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)
    >>> solver = gd.OrbitSolver()  # defaults to Dopri8
    >>> w0 = gc.PhaseSpaceCoordinate(
    ...     q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"),
    ...     t=u.Quantity(0, "Gyr"))
    >>> t1 = u.Quantity(1, "Gyr")
    >>> soln = solver.solve(field, w0, t1)
    >>> soln.ts.shape, soln.ys[0].shape
    ((1,), (1, 2, 3))

    >>> w = gc.PhaseSpaceCoordinate.from_(soln, units=pot.units, frame=w0.frame)
    >>> print(w, w.shape, sep='\n')
    PhaseSpaceCoordinate(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[-5.151 -6.454 -5.795]
             [ 4.277  4.633  1.426]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[ 0.225 -0.068  0.253]
             [-0.439 -0.002 -0.146]]>,
        t=Quantity(Array(1000., dtype=float64), unit='Myr'),
        frame=SimulationFrame())
    (2,)

    >>> w = gc.PhaseSpaceCoordinate.from_(soln, units=pot.units, frame=w0.frame,
    ...                                   unbatch_time=False)
    >>> print(w.shape)  # (*batch, [time])
    (2, 1)

    """
    # Reshape (*tbatch, T, *ybatch, *shape) to (*tbatch, *ybatch, T, *shape)
    t = soln.ts  # already in the shape (*tbatch, T)
    n_tbatch = soln.t0.ndim
    q, p = soln.ys
    if n_tbatch < q.ndim - 1:
        q = jnp.moveaxis(q, n_tbatch, -2)
        p = jnp.moveaxis(p, n_tbatch, -2)

    # Reshape (*tbatch, *ybatch, T) to (*tbatch, *ybatch) if T == 1
    if unbatch_time and t.ndim > 0 and t.shape[-1] == 1:
        t = t[..., -1]
        q = q[..., -1, :]
        p = p[..., -1, :]

    # Convert the solution to a phase-space position
    return cls(
        q=cx.CartesianPos3D.from_(q, units["length"]),
        p=cx.CartesianVel3D.from_(p, units["speed"]),
        t=FastQ(t, units["time"]),
        frame=frame,
    )


@gc.AbstractPhaseSpaceObject.from_.dispatch
def from_(
    cls: type[Orbit],
    soln: dfx.Solution,
    *,
    units: u.AbstractUnitSystem,  # not dispatched on, but
    frame: cx.frames.AbstractReferenceFrame,  # not dispatched on, but required
    unbatch_time: bool = True,
) -> Orbit:
    r"""Create a `galax.dynamics.Orbit` from a `diffrax.Solution`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import galax.coordinates as gc
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> pot = gp.HernquistPotential(m_tot=1e12, r_s=5, units="galactic")
    >>> field = gd.fields.HamiltonianField(pot)

    >>> solver = gd.OrbitSolver()
    >>> t1 = u.Quantity(1, "Gyr")
    >>> saveat=u.Quantity(jnp.linspace(0.2, 1, 71), "Gyr")

    Solving scalar initial conditions:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([8, 0, 0], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"), t=u.Quantity(0, "Gyr"))

    >>> soln = solver.solve(field, w0, t1, dense=True, saveat=saveat)
    >>> (soln.ts.shape, jax.tree.map(lambda x: x.shape, soln.ys))
    ((71,), ((71, 3), (71, 3)))

    >>> orbit = gd.Orbit.from_(soln, frame=w0.frame, units=pot.units)
    >>> print(orbit[..., :3])
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[-2.78  -7.464  0.   ]
             [-0.082 -7.06   0.   ]
             [ 2.406 -2.454  0.   ]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[ 0.199 -0.114  0.   ]
             [ 0.257  0.196  0.   ]
             [ 0.102  0.644  0.   ]]>,
        t=Quantity(Array([200. , 211.42857143, 222.85714286], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        interpolant=PhaseSpaceInterpolation( ... ))

    Solving batched initial conditions:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"), t=u.Quantity(0, "Gyr"))

    >>> soln = solver.solve(field, w0, t1, dense=True, saveat=saveat)
    >>> (soln.ts.shape, jax.tree.map(lambda x: x.shape, soln.ys))
    ((71,), ((71, 2, 3), (71, 2, 3)))

    >>> orbit = gd.Orbit.from_(soln, frame=w0.frame, units=pot.units)
    >>> print(orbit[..., :3])
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-1.715 -3.596 -1.929]
              [ 3.373 -1.433  3.794]
              [ 5.663  3.143  6.371]]
             [[ 4.418  5.691  1.473]
              [-0.859  3.25  -0.286]
              [-3.383 -3.917 -1.128]]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[[ 0.454 -0.097  0.511]
              [ 0.327  0.395  0.368]
              [ 0.099  0.373  0.112]]
             [[-0.376 -0.026 -0.125]
              [-0.496 -0.48  -0.165]
              [ 0.043 -0.549  0.014]]]>,
        t=Quantity(Array([200. , 211.42857143, 222.85714286], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        interpolant=PhaseSpaceInterpolation( ... ))

    Solving batched times and initial conditions:

    >>> w0 = gc.PhaseSpaceCoordinate(q=u.Quantity([[8, 0, 9], [9, 0, 3]], "kpc"),
    ...     p=u.Quantity([0, 220, 0], "km/s"), t=u.Quantity([0, 0.1], "Gyr"))

    >>> soln = solver.solve(field, w0, t1, dense=True, saveat=saveat)
    >>> (soln.ts.shape, jax.tree.map(lambda x: x.shape, soln.ys))
    ((71,), ((2, 71, 3), (2, 71, 3)))

    >>> orbit = gd.Orbit.from_(soln, frame=w0.frame, units=pot.units)
    >>> print(orbit[..., :3])
    Orbit(
        q=<CartesianPos3D (x[kpc], y[kpc], z[kpc])
            [[[-1.715e+00 -3.596e+00 -1.929e+00]
              [ 3.373e+00 -1.433e+00  3.794e+00]
              [ 5.663e+00  3.143e+00  6.371e+00]]
             [[-8.248e-03  5.207e+00 -2.749e-03]
              [-4.030e+00  7.323e+00 -1.343e+00]
              [-6.566e+00  6.488e+00 -2.189e+00]]]>,
        p=<CartesianVel3D (x[kpc / Myr], y[kpc / Myr], z[kpc / Myr])
            [[[ 0.454 -0.097  0.511]
              [ 0.327  0.395  0.368]
              [ 0.099  0.373  0.112]]
             [[-0.39   0.377 -0.13 ]
              [-0.295  0.033 -0.098]
              [-0.143 -0.167 -0.048]]]>,
        t=Quantity(Array([200. , 211.42857143, 222.85714286], dtype=float64), unit='Myr'),
        frame=SimulationFrame(),
        interpolant=PhaseSpaceInterpolation( ... ))

    """  # noqa: E501
    # TODO: don't double construct?
    w = gc.PhaseSpaceCoordinate.from_(
        soln, frame=frame, units=units, unbatch_time=unbatch_time
    )
    return cls(
        q=w.q,
        p=w.p,
        t=w.t,
        frame=w.frame,
        interpolant=None
        if soln.interpolation is None
        else PhaseSpaceInterpolation(soln.interpolation, units=units),
    )
