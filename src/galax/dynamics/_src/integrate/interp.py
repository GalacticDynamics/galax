__all__ = ["Interpolant"]

from typing import Any

import diffrax
import equinox as eqx
import jax

import quaxed.numpy as xp
import unxt as u

import galax.coordinates as gc


class Interpolant(eqx.Module):  # type: ignore[misc]#
    """Wrapper for `diffrax.DenseInterpolation`.

    This satisfies the `galax.coordinates.PhaseSpacePositionInterpolant`
    Protocol.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> from unxt.unitsystems import galactic
    >>> import galax.coordinates as gc
    >>> import galax.dynamics as gd
    >>> import galax.potential as gp

    We define initial conditions and a potential:

    >>> w0 = gc.PhaseSpacePosition(q=u.Quantity([10., 0., 0.], "kpc"),
    ...                            p=u.Quantity([0., 200., 0.], "km/s"))
    >>> pot = gp.HernquistPotential(m_tot=u.Quantity(1e12, "Msun"),
    ...                             r_s=u.Quantity(5, "kpc"), units="galactic")

    We can integrate the phase-space position for 1 Gyr, getting the final
    position.  The integrator accepts any function for the equations of motion.
    Here we will reproduce what happens with orbit integrations.

    >>> integrator = gd.integrate.Integrator()
    >>> t0, t1 = u.Quantity(0, "Gyr"), u.Quantity(1, "Gyr")
    >>> w = integrator(pot._vector_field, w0, t0, t1, units=galactic,
    ...                interpolated=True)
    >>> type(w)
    <class 'galax.coordinates...InterpolatedPhaseSpacePosition'>

    >>> isinstance(w.interpolant, gc.PhaseSpacePositionInterpolant)
    True

    """

    interpolant: diffrax.DenseInterpolation
    """:class:`diffrax.DenseInterpolation` object.

    This object is the result of the integration and can be used to evaluate the
    interpolated solution at any time. However it does not understand units, so
    the input is the time in ``units["time"]``. The output is a 6-vector of
    (q, p) values in the units of the integrator.
    """

    units: u.AbstractUnitSystem = eqx.field(static=True, converter=u.unitsystem)
    """The :class:`unxt.AbstractUnitSystem`.

    This is used to convert the time input to the interpolant and the phase-space
    position output.
    """

    def __call__(self, t: u.Quantity["time"], **_: Any) -> gc.PhaseSpacePosition:
        """Evaluate the interpolation."""
        # Parse t
        t_ = xp.atleast_1d(u.ustrip(self.units["time"], t))

        # Evaluate the interpolation
        ys = jax.vmap(self.interpolant.evaluate)(t_)

        # Reshape (T, *batch) to (*batch, T)
        # ts is already in the correct shape
        ys = xp.moveaxis(ys, 0, -2)
        # Reshape (*batch,T=1,6) to (*batch,6) if t is a scalar
        t_dim_sel = -1 if t.shape == () else slice(None)
        ys = ys[..., t_dim_sel, :]

        # Construct and return the result
        return gc.PhaseSpacePosition(
            q=u.Quantity(ys[..., 0:3], self.units["length"]),
            p=u.Quantity(ys[..., 3:6], self.units["speed"]),
            t=t,
        )
