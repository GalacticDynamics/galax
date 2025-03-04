"""MW + LMC field."""

__all__ = [
    "RigidMWandLMCField",
    "make_mw_lmc_potential",
    "radial_velocity_dispersion_helper",
]


from collections.abc import Callable
from dataclasses import KW_ONLY
from functools import partial
from typing import Any, TypeAlias, final
from typing_extensions import override

import equinox as eqx
import interpax as ipx
import jax
import jax.tree as jtu
from jax import lax
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, Real
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless
from unxt.quantity import AllowValue

import galax._custom_types as gt
import galax.dynamics._src.custom_types as gdt
import galax.potential as gp
from .uniform_acceleration import UniformAcceleration
from galax.dynamics._src.fields import AbstractField

QPQParr: TypeAlias = tuple[gdt.QParr, gdt.QParr]
QPQP: TypeAlias = tuple[gdt.QP, gdt.QP]


@partial(jax.jit, static_argnames=("log10r_max", "log10r_num"))
def radial_velocity_dispersion_helper(  # TODO: better name
    pot: gp.AbstractPotential,
    /,
    r: Real[Array, "R"],
    beta: gt.LikeSz0,
    *,
    log10r_max: float = 4.0,  # [kpc]
    log10r_num: int = 2_000,  # [kpc]
) -> Real[Array, "R"]:
    r"""Compute the radial velocity dispersion from spherical Jean's.

    Assuming constant anisotropy $\beta$ and spherical symmetry, the radial
    velocity dispersion is given by

    $$

        \sigma_r^2 = \frac{r^{-2\beta}}{\rho(r)} \int\limits_{r}^{\infty}
                     dr^\prime r^{\prime{2\beta} } \rho(r^\prime) \frac{d\Phi}{dr}

    $$

    Examples
    --------
    This helper function is mostly for use in constructing a spline for the
    velocity dispersion function. For example

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import interpax
    >>> import galax.potential as gp
    >>> mw_pot = gp.MilkyWayPotential(units="galactic")

    >>> r_interp = jnp.logspace(-1, 3., 1000)  # [kpc]
    >>> beta = .6
    >>> sigma_r_interp = jax.vmap(radial_velocity_dispersion_helper,in_axes=(None,0,None))(mw_pot, r_interp,beta)
    >>> sigma_r_func = interpax.Interpolator1D(x=r_interp, f=sigma_r_interp, method='cubic2')

    """  # noqa: E501
    r_grid = jnp.logspace(jnp.log10(r), log10r_max, num=log10r_num)
    r_eval = jnp.stack(
        [r_grid, jnp.zeros_like(r_grid), jnp.zeros_like(r_grid)], axis=-1
    )

    density = jax.vmap(pot.density, in_axes=(0, None))(r_eval, 0.0)
    innermost_density = density[0]
    dphi_dr = jax.vmap(pot.dpotential_dr, in_axes=(0, None))(r_eval, 0.0)

    integrand = jnp.power(r_grid, 2 * beta) * (density * dphi_dr)
    integral = trapezoid(integrand, x=r_grid)

    sigma2 = jnp.power(r, -2 * beta) / innermost_density * integral
    return jnp.sqrt(sigma2)


@final
class RigidMWandLMCField(AbstractField):
    """Dually evolving MW and LMC potentials.

    This field evolves the MW and LMC potentials simultaneously, treating the MW
    and LMC as rigid body potentials. The centroids are evolved in response to
    each other, with Chandrasekhar dynamical friction for the LMC.

    This is based on the script from AGAMA:
    https://github.com/GalacticDynamics-Oxford/Agama/blob/c507fc3e703513ae4a41bb705e171a4d036754a8/py/example_lmc_mw_interaction.py

    Parameters
    ----------
    mw_pot
        MW Potential.
    lmc_pot
        LMC Potential.
    sigma_func
        Velocity dispersion function of the MW at the LMC position. Maps
        (Array[float, (3)]) -> Array[float, ()].
    b_coulomb_min
        Minimum impact parameter for Coulomb logarithm.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp
    >>> import galax.dynamics as gd

    >>> mw_pot = gp.MilkyWayPotential(units="galactic")
    >>> lmc_pot = gp.NFWPotential(m=1e11, r_s=5, units="galactic")

    >>> def sigma_fn(xyz):
    ...     return u.Quantity(130, "km/s").ustrip("kpc/Myr")

    >>> b_coulomb_min=u.Quantity(1, "kpc")

    >>> field = gd.examples.RigidMWandLMCField(mw_pot=mw_pot, lmc_pot=lmc_pot,
    ...     sigma_func=sigma_fn, b_coulomb_min=b_coulomb_min)

    >>> y0 = ((u.Quantity([0, 0, 0], "kpc"), u.Quantity([0, 0, 0], "kpc/Myr")),
    ...       (u.Quantity([-0.8, -41.5, -26.8], "kpc"), u.Quantity([-56, -219, 186], "km/s")))
    >>> ts = u.Quantity(jnp.linspace(0, -14_000, 100), "Myr")

    >>> soln = gd.integrate_field( field, y0, ts )

    >>> soln.ys[0][0][:10, 0]
    Array([ 0.        ,  0.13036419,  1.02986725,  2.63339191,  4.62211495,
           6.83076014,  9.17829616, 11.62045687, 14.13056816, 16.69127387], dtype=float64)

    """  # noqa: E501

    mw_pot: gp.AbstractPotential
    """MW Potential."""

    lmc_pot: gp.AbstractPotential
    """LMC Potential."""

    sigma_func: Callable[[gt.Sz3], gt.Sz0] = eqx.field(
        converter=Unless(eqx.Partial, eqx.Partial)
    )
    """Velocity dispersion function of the MW at the LMC position."""

    b_coulomb_min: u.Quantity["length"]
    """Minimum impact parameter for Coulomb logarithm."""

    _: KW_ONLY

    units: u.AbstractUnitSystem = eqx.field(
        converter=u.unitsystem, default="galactic", static=True
    )

    def __check_init__(self) -> None:
        units = eqx.error_if(
            self.units,
            self.units != self.mw_pot.units,
            "Units of MW and LMC potentials must match.",
        )
        units = eqx.error_if(
            units,
            self.units != self.lmc_pot.units,
            "Units of MW and LMC potentials must match.",
        )

    @property
    def mass_lmc(self) -> Array:
        """Mass of the LMC."""
        xyz = jnp.array([100.0, 0, 0])
        t = jnp.array(0.0)
        return gp.spherical_mass_enclosed(self.lmc_pot, xyz, t)

    @override  # specify the signature of the `__call__` method.
    @dispatch.abstract
    def __call__(self, *_: Any, **kw: Any) -> tuple[Any, Any]:
        """Evaluate the field at a given coordinate."""
        raise NotImplementedError  # pragma: no cover

    @AbstractField.parse_inputs.dispatch  # type: ignore[misc]
    def parse_inputs(
        self: "RigidMWandLMCField",
        t0: gt.LikeSz0 | gt.QuSz0,
        y0: QPQParr | QPQP,
        /,
        *,
        ustrip: bool = False,  # noqa: ARG002
    ) -> tuple[gt.Sz0, QPQParr]:
        t0 = u.ustrip(AllowValue, self.units["time"], t0)
        y0 = _parse_y0(y0, self.units)
        return t0, y0


# ============================================================================
# Parse inputs


def _parse_y0(y0: QPQParr | QPQP, /, units: u.AbstractUnitSystem) -> QPQParr:
    to_array_fn = (
        lambda x: u.ustrip(units[u.dimension_of(x)], x)
        if u.quantity.is_any_quantity(x)
        else x
    )
    is_leaf = lambda x: eqx.is_array(x) or u.quantity.is_any_quantity(x)
    y0: QPQParr = jtu.map(to_array_fn, y0, is_leaf=is_leaf)
    return y0


# ============================================================================
# Call


@RigidMWandLMCField.__call__.dispatch
@partial(eqx.filter_jit)
def __call__(
    self: RigidMWandLMCField, t: gt.LikeSz0, coords: QPQParr, _: gt.OptArgs = None, /
) -> QPQParr:
    mw_x0, mw_v0 = coords[0]  # MW position and velocity
    lmc_x1, lmc_v1 = coords[1]  # LMC position and velocity

    delta_x = lmc_x1 - mw_x0  # relative position - from MW center
    delta_v = lmc_v1 - mw_v0  # relative velocity - from MW center

    dist = jnp.linalg.vector_norm(delta_x, axis=-1)
    vmag = jnp.linalg.vector_norm(delta_v, axis=-1)

    t = jnp.asarray(t)
    mw_a = self.lmc_pot.acceleration(-delta_x, t)  # force from LMC on MW center
    f_lmc_from_mw = self.mw_pot.acceleration(delta_x, t)  # force from MW on LMC
    rho = self.mw_pot.density(delta_x, t)  # MW density at LMC position

    # Distance-dependent Coulomb logarithm
    # (an approximation that best matches the results of N-body simulations)
    sigma = self.sigma_func(dist)  # velocity dispersion of MW at LMC position
    b_min = self.b_coulomb_min.ustrip(self.units["length"])
    couLog = jnp.maximum(0.0, jnp.sqrt(jnp.log(dist) - jnp.log(b_min)))
    X = vmag / (sigma * lax.sqrt(2.0))
    G = self.mw_pot.constants["G"].value
    drag = -(  # dynamical friction force
        (4 * jnp.pi * G * rho)
        * (delta_v / vmag)
        * (lax.erf(X) - 2 / lax.sqrt(jnp.pi) * X * jnp.exp(-X * X))
        * (G * self.mass_lmc / vmag**2)
        * couLog
    )
    lmc_a = f_lmc_from_mw + drag

    return ((mw_v0, mw_a), (lmc_v1, lmc_a))


@RigidMWandLMCField.__call__.dispatch
@partial(eqx.filter_jit)
def __call__(
    self: RigidMWandLMCField, t: gt.QuSz0, coords: QPQP, args: gt.OptArgs = None, /
) -> QPQParr:
    t = u.ustrip(AllowValue, self.units["time"], t)
    coords = jtu.map(
        partial(u.ustrip, AllowValue, self.units["length"]),
        coords,
        is_leaf=u.quantity.is_any_quantity,
    )
    return self(t, coords, args)


# ============================================================================


def _sigma_fn(_: gt.Sz3, /) -> gt.Sz0:
    """Velocity dispersion function of the MW at the LMC position.

    Maps (Array[float, (3)]) -> Array[float, ()].

    """
    return u.Quantity(130, "km/s").ustrip("kpc/Myr")


t_interp = u.Quantity(jnp.linspace(0, -14, 10_000), "Gyr")
b_coulomb_min_default = u.Quantity(1, "kpc")


def make_mw_lmc_potential(
    mw_pot: gp.AbstractPotential,
    lmc_pot: gp.AbstractPotential,  # LMC potential
    mw_w0: gdt.QP | gdt.QParr,  # MW present-day phase-space position
    lmc_w0: gdt.QP | gdt.QParr,  # LMC present-day phase-space position
    sigma_func: Callable[[gt.Sz3], gt.Sz0] = _sigma_fn,
    b_coulomb_min: u.Quantity["length"] = b_coulomb_min_default,
    t_interp: u.Quantity["time"] = t_interp,
    solver_kwargs: dict[str, Any] | None = None,
) -> gp.CompositePotential:
    """Build a MW + LMC potential.

    Parameters
    ----------
    mw_pot
        Milky Way Potential.
    lmc_pot
        LMC Potential.
    mw_w0
        MW present-day phase-space position.
    lmc_w0
        LMC present-day phase-space position.
    sigma_func
        Velocity dispersion function of the MW at the LMC position.
        Maps (Array[float, (3)]) -> Array[float, ()].
    b_coulomb_min
        Minimum impact parameter for Coulomb logarithm.

    Examples
    --------
    >>> def sigma_fn(xyz):
    ...     return u.Quantity(130, "km/s").ustrip("kpc/Myr")

    >>> mw_pot = gp.MilkyWayPotential(units="galactic")
    >>> lmc_pot = gp.NFWPotential(m=1e11, r_s=5, units="galactic")

    >>> mw_w0 = (u.Quantity([0, 0, 0], "kpc"), u.Quantity([0, 0, 0], "kpc/Myr"))
    >>> lmc_w0 = (u.Quantity([-0.8, -41.5, -26.8], "kpc"),
    ...           u.Quantity([-56, -219, 186], "km/s"))

    >>> mw_lmc_pot = make_mw_lmc_potential(mw_pot, lmc_pot, mw_w0, lmc_w0,
    ...      sigma_func=sigma_fn, b_coulomb_min=u.Quantity(1, "kpc"),
    ...      t_interp=u.Quantity(jnp.linspace(-0, -14, 1_000), "Gyr"))
    >>> mw_lmc_pot
    CompositePotential({'mw': TranslatedPotential(
            base_potential=MilkyWayPotential( ... ),
            translation=TimeDependentTranslationParameter( ... )
        ),
        'lmc': TranslatedPotential(
            base_potential=NFWPotential( ... ),
            translation=TimeDependentTranslationParameter( ... )
        ),
        'mw_acc': UniformAcceleration( ... )})

    """
    from galax.dynamics import integrate_field

    # Check that the units of the potentials match
    units = eqx.error_if(
        mw_pot.units,
        mw_pot.units != lmc_pot.units,
        "Units of MW and LMC potentials must match.",
    )

    mw_lmc_field = RigidMWandLMCField(
        mw_pot=mw_pot,
        lmc_pot=lmc_pot,
        sigma_func=sigma_func,
        b_coulomb_min=b_coulomb_min,
    )
    y0 = (mw_w0, lmc_w0)

    soln = integrate_field(mw_lmc_field, y0, t_interp, **(solver_kwargs or {}))

    mw_moving = gp.TranslatedPotential(
        mw_pot,
        gp.params.TimeDependentTranslationParameter.from_(
            soln.ts, soln.ys[0][0], units=units
        ),
    )
    lmc_moving = gp.TranslatedPotential(
        lmc_pot,
        gp.params.TimeDependentTranslationParameter.from_(
            soln.ts, soln.ys[1][0], units=units
        ),
    )

    unif_acc = UniformAcceleration(
        velocity_func=ipx.CubicSpline(soln.ts[::-1], soln.ys[0][1][::-1]),
        units=units,
        func_supports_units=False,
    )

    return gp.CompositePotential(mw=mw_moving, lmc=lmc_moving, mw_acc=unif_acc)
