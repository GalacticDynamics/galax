"""Private module that defines `MonariEtAl2016BarPotential` and its functions.

The public API is `galax.potential.monarietal2016bar`.
"""

__all__ = [
    # class
    "MonariEtAl2016BarPotential",
    # functions
    "density",
    "gradient",
    "hessian",
    "laplacian",
    "potential",
]

import functools as ft
from dataclasses import KW_ONLY

from typing import final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

import galax.potential.custom_types as gt
from galax.potential._src.base_single import (
    AbstractSinglePotential,
    LaplacianFromDensityMixin,
)
from galax.potential._src.jax import vectorize_method
from galax.potential._src.kernels import (
    cartesian_gradient,
    cartesian_hessian,
    poisson_laplacian,
)
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField


@final
class MonariEtAl2016BarPotential(LaplacianFromDensityMixin, AbstractSinglePotential):
    r"""Monari et al. (2016) bar potential.

    The model is a rigidly rotating bar that adds a quadrupole perturbation to
    an axisymmetric Galaxy potential. It is the three-dimensional version of the
    quadrupole bar of Dehnen (2000) [Dehnen2000]_, with Dehnen's cylindrical
    radius replaced by the spherical radius :math:`r`, as written in
    [Monari2016]_. The perturbation is a pure :math:`l = 2`, :math:`m = 2`
    quadrupole whose radial profile is a cubic inside the bar length ``Rb`` and
    a source-free :math:`r^{-3}` tail outside. The parameter ``alpha`` is the
    bar strength, ``R0`` the Galactocentric radius of the Sun, ``v0`` the
    circular velocity at ``R0``, ``Rb`` the length of the bar, ``phi_b`` the bar
    angle at :math:`t = 0` and ``Omega`` the pattern speed. The parameters
    ``R0`` and ``v0`` have no default. The defaults ``alpha = 0.01``,
    ``Rb = 3.5 kpc`` and ``Omega = 52.2 km/(s kpc)`` are the values Monari et
    al. (2016) use in their simulations. They fix the bar angle differently:
    their 25 degrees is :math:`\phi_b + \Omega t_e`, the angle between the bar
    major axis and the Sun-centre line at the end of the run :math:`t_e`, so the
    default ``phi_b = 25 deg`` is that angle at :math:`t = 0`.

    The potential is

    .. math::

        \Phi(r, \theta, \phi, t) = A_b \, U(r / R_b) \sin^2\theta \,
            \cos\gamma_b,
        \qquad
        A_b = \frac{\alpha v_0^2}{3} \left(\frac{R_0}{R_b}\right)^3,

    with the bar angle :math:`\gamma_b = 2(\phi - \phi_b - \Omega t)` and the
    radial function

    .. math::

        U(s) = \begin{cases}
            s^3 - 2, & s < 1, \\
            -s^{-3}, & s \geq 1,
        \end{cases}

    where :math:`r^2 = x^2 + y^2 + z^2`, :math:`\sin^2\theta = (x^2 + y^2) / r^2`
    and :math:`\phi` is the azimuth in the :math:`x`-:math:`y` plane. In the
    plane :math:`z = 0` this is Dehnen's bar potential. The density follows from
    Poisson's equation, :math:`\nabla^2 \Phi = 4 \pi G \rho`, and is

    .. math::

        \rho(r, \theta, \phi, t) = \frac{A_b}{4 \pi G} \, L(r) \sin^2\theta \,
            \cos\gamma_b,
        \qquad
        L(r) = \begin{cases}
            \dfrac{12}{r^2} + \dfrac{6 r}{R_b^3}, & r < R_b, \\[4pt]
            0, & r \geq R_b,
        \end{cases}

    where :math:`G` is the gravitational constant. The density is exactly zero
    outside the bar, :math:`r \geq R_b`. The total mass is zero. The mass in the
    positive lobes, which equals minus the mass in the negative lobes, is

    .. math::

        M_+ = \frac{9 A_b R_b}{\pi G}.

    The strength :math:`\alpha` is the ratio of the radial force of the bar to
    the axisymmetric radial force :math:`v_0^2 / R_0`, along the long axis of the
    bar at :math:`(R, z) = (R_0, 0)` for :math:`R_0 \geq R_b`.

    Notes
    -----
    The potential and its first derivative are continuous at :math:`r = R_b`,
    and the density jumps there. The density is a perturbation: it is positive
    along the bar axis and negative along the perpendicular axis, with equal
    mass in the two, so the potential is meant to be added to an axisymmetric
    one. The amplitude is constant in time, so the smooth switch-on of Dehnen
    (2000) is not included. This class is time dependent through
    :math:`\gamma_b`, and the module functions take the time as a third argument,
    ``potential(p, xyz, t)``.

    Equation 3 of [Monari2016]_ as printed gives :math:`U = +(r / R_b)^{-3}` for
    :math:`r \geq R_b`, which is discontinuous at :math:`r = R_b`. This class
    uses :math:`-(r / R_b)^{-3}`, which is the sign of Dehnen's equation 4.

    References
    ----------
    .. [Monari2016] Monari, G., Famaey, B., Siebert, A., Grand, R. J. J.,
       Kawata, D., & Boily, C. (2016). The effects of bar-spiral coupling on
       stellar kinematics in the Galaxy. Monthly Notices of the Royal
       Astronomical Society, 461, 3835-3846.
       `ADS <https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.3835M>`__;
       `doi:10.1093/mnras/stw1564 <https://doi.org/10.1093/mnras/stw1564>`__;
       `arXiv:1606.06785 <https://arxiv.org/abs/1606.06785>`__.
       The bar is their eqs. 1-3.

    .. [Dehnen2000] Dehnen, W. (2000). The Effect of the Outer Lindblad
       Resonance of the Galactic Bar on the Local Stellar Velocity
       Distribution. The Astronomical Journal, 119, 800-812.
       `ADS <https://ui.adsabs.harvard.edu/abs/2000AJ....119..800D>`__;
       `doi:10.1086/301226 <https://doi.org/10.1086/301226>`__;
       `arXiv:astro-ph/9911161 <https://arxiv.org/abs/astro-ph/9911161>`__.
       The planar quadrupole bar is his eq. 4 and the strength his eq. 8.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> pot = gp.MonariEtAl2016BarPotential(
    ...     alpha=0.01,
    ...     R0=u.Q(8.0, "kpc"),
    ...     v0=u.Q(220.0, "km/s"),
    ...     Rb=u.Q(3.5, "kpc"),
    ...     phi_b=u.Q(25, "deg"),
    ...     Omega=u.Q(52.2, "km/(s kpc)"),
    ...     units="galactic",
    ... )
    >>> pot(u.Q([8.0, 0.0, 0.0], "kpc"), u.Q(0.0, "Gyr"))
    Q(-0.00010847, 'kpc2 / Myr2')

    """

    _: KW_ONLY

    alpha: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(0.01, ""),
        dimensions="dimensionless",
        doc="""The amplitude.

    the ratio between the bar's and axisymmetric contribution to the radial
    force, along the bar's long axis at (R,z) = (R0,0).
    """,
    )

    R0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="length", doc="The Galactocentric radius of the Sun."
    )

    v0: AbstractParameter = ParameterField(  # type: ignore[assignment]
        dimensions="speed", doc="The circular velocity at R0."
    )

    Rb: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(3.5, "kpc"),
        dimensions="length",
        doc="The length of the bar.",
    )

    phi_b: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(25, "deg"), dimensions="angle", doc="Bar angle."
    )

    Omega: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.StaticQuantity(52.2, "km/(s kpc)"),
        dimensions="frequency",
        doc="Bar pattern speed.",
    )

    @ft.partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _potential(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        # Compute parameters
        params = self.params(t)
        return potential(params, xyz, t)  # type: ignore[no-any-return]

    @ft.partial(jax.jit)
    @vectorize_method(signature="(3),()->()")
    def _density(self, xyz: gt.QuSz3 | gt.Sz3, t: gt.QuSz0 | gt.Sz0) -> gt.Sz0:
        # Parse inputs
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        t = u.ustrip(AllowValue, self.units["time"], t)
        # Compute parameters
        params = self.params(t)
        return density(params, xyz, t)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def U_of_r(s: gt.Sz0, /) -> gt.Sz0:
    r"""Radial profile of the bar potential, as a function of :math:`s = r / R_b`.

    .. math::

        U(s) = \begin{cases} s^3 - 2, & s < 1, \\ -s^{-3}, & s \geq 1, \end{cases}

    which is continuous at :math:`s = 1` together with its first derivative. The
    printed eq. 3 of Monari et al. (2016) has :math:`+s^{-3}` for :math:`s \geq 1`;
    the minus sign is the one of Dehnen (2000, eq. 4).
    """

    # M+2016 eq.3, modified to work on s=r/Rb
    def gtr_func(s: gt.Sz0) -> gt.Sz0:
        return -(s**-3)

    def less_func(s: gt.Sz0) -> gt.Sz0:
        return s**3 - 2.0

    pred = s >= 1
    return jax.lax.cond(pred, gtr_func, less_func, s)  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def L_of_r(r: gt.Sz0, r_b: gt.Sz0, /) -> gt.Sz0:
    r"""Radial part of the density, the :math:`l = 2` radial Laplacian of :math:`U`.

    .. math::

        L(r) = \frac{\mathrm{d}^2 U}{\mathrm{d}r^2}
            + \frac{2}{r}\frac{\mathrm{d}U}{\mathrm{d}r} - \frac{6}{r^2} U
            = \begin{cases}
                \dfrac{12}{r^2} + \dfrac{6 r}{R_b^3}, & r < R_b, \\
                0, & r \geq R_b,
            \end{cases}

    with :math:`U = U(r / R_b)`. It is exactly zero outside the bar, since
    :math:`U = -(r/R_b)^{-3}` there is a source-free (harmonic) :math:`l = 2`
    radial solution.
    """
    return jnp.where(r < r_b, 12 / r**2 + 6 * r / r_b**3, 0.0)  # type: ignore[no-any-return]


# ===================================================================


@ft.partial(jax.jit)
def potential(p: gt.Params, xyz: gt.Sz3, t: gt.Sz0, /) -> gt.Sz0:
    r"""Specific potential energy (Monari et al. 2016, eq. 1).

    .. math::

        \Phi = \frac{\alpha v_0^2}{3} \left(\frac{R_0}{R_b}\right)^3
            U(r / R_b) \frac{R^2}{r^2} \cos\gamma_b,
        \qquad
        \gamma_b = 2(\phi - \phi_b - \Omega t),

    with :math:`R^2 = x^2 + y^2`, :math:`r^2 = R^2 + z^2` and :math:`\phi` the
    azimuth of ``xyz``. ``t`` is the time in the unit system of the potential.

    """
    R2 = xyz[0] ** 2 + xyz[1] ** 2
    r2 = R2 + xyz[2] ** 2

    prefactor = p["alpha"] * (p["v0"] ** 2 / 3) * (p["R0"] / p["Rb"]) ** 3
    u_of_r = U_of_r(jnp.sqrt(r2) / p["Rb"])
    phi = jnp.arctan2(xyz[1], xyz[0])
    gamma_b = 2 * (phi - p["phi_b"] - p["Omega"] * t)  # M+2016 eq.2

    energy = prefactor * u_of_r * (R2 / r2) * jnp.cos(gamma_b)  # M+2016 eq.1
    return energy  # type: ignore[no-any-return]


@ft.partial(jax.jit)
def density(p: gt.Params, xyz: gt.Sz3, t: gt.Sz0, /) -> gt.Sz0:
    r"""Density derived from the potential above via Poisson's equation.

    .. math::

        \rho = \frac{1}{4 \pi G} \frac{\alpha v_0^2}{3}
            \left(\frac{R_0}{R_b}\right)^3 L(r) \frac{R^2}{r^2} \cos\gamma_b,

    with :math:`L(r) = 12 / r^2 + 6 r / R_b^3` for :math:`r < R_b` and 0 outside
    (see ``L_of_r``), :math:`\gamma_b = 2(\phi - \phi_b - \Omega t)` and
    :math:`G` the gravitational constant.

    """
    R2 = xyz[0] ** 2 + xyz[1] ** 2
    r2 = R2 + xyz[2] ** 2
    r = jnp.sqrt(r2)

    prefactor = p["alpha"] * (p["v0"] ** 2 / 3) * (p["R0"] / p["Rb"]) ** 3
    phi = jnp.arctan2(xyz[1], xyz[0])
    gamma_b = 2 * (phi - p["phi_b"] - p["Omega"] * t)  # M+2016 eq.2

    lap_ang = prefactor * L_of_r(r, p["Rb"]) * (R2 / r2) * jnp.cos(gamma_b)
    return lap_ang / (4 * jnp.pi * p["G"])  # type: ignore[no-any-return]


# Derived quantities, built from the kernels above.
laplacian = poisson_laplacian(density)
gradient = cartesian_gradient(potential)
hessian = cartesian_hessian(potential)
