"""Sormani et al. (2022) model of the Milky Way's bar."""

__all__ = ["SormaniEtAl2022BarPotential"]

from typing import Any, Final, final

import jax

import quaxed.numpy as jnp
import unxt as u

import galax.potential.custom_types as gt
from .multipole_profile import (
    AbstractMultipoleProfilePotential,
    MultipoleProfilePotential,
)
from galax.potential._src.base import default_constants
from galax.potential._src.symmetry import Symmetry

# Sormani et al. (2022) Table 1, at the precision of the authors' own
# implementation (Agama's ``py/example_mw_bar_potential.py``), which rounds to
# every entry of the table. kpc and Msun / kpc^3.
_BAR_1: Final = {  # eqs 2-5
    "rho": 3.16273226e09,
    "x": 4.90209137e-01,
    "y": 3.92017253e-01,
    "z": 2.29482096e-01,
    "c_par": 1.99110223e00,
    "c_perp": 2.23179266e00,
    "m": 8.73227940e-01,
    "r_cut": 4.36983774e00,
    "alpha": 6.25670015e-01,
    "c": 1.34152138e00,
    "n": 1.94025114e00,
    "x_c": 7.50504078e-01,
    "y_c": 4.68875471e-01,
}
_BAR_2: Final = {  # eqs 6-8, i = 2
    "rho": 4.95381575e08,
    "x": 5.36363324e00,
    "y": 9.58522229e-01,
    "z": 6.10542494e-01,
    "n": 3.05125124e00,
    "c_perp": 9.69645220e-01,
    "R_out": 3.19043585e00,
    "R_in": 5.58255674e-01,
    "n_out": 1.67310332e01,
    "n_in": 3.19575493e00,
}
_BAR_3: Final = {  # eqs 6-8, i = 3: the long bar
    "rho": 1.74304936e13,
    "x": 4.77961423e-01,
    "y": 2.66853061e-01,
    "z": 2.51516920e-01,
    "n": 9.80136710e-01,
    "c_perp": 1.87882599e00,
    "R_out": 2.20415408e00,
    "R_in": 7.60708626e00,
    "n_out": -2.72907665e01,
    "n_in": 1.62966434e00,
}

M_TOT: Final = 1.8252e10
"""Total mass of the three components [Msun], by direct integration."""


def _sech(x: gt.Sz0) -> gt.Sz0:
    """``sech`` that does not overflow at large ``|x|``."""
    e = jnp.exp(-jnp.abs(x))
    out: gt.Sz0 = 2 * e / (1 + e * e)
    return out


def _components(xyz: gt.BtSz3, /) -> tuple[gt.BtSz0, gt.BtSz0, gt.BtSz0]:
    """Return the three barred components of Sormani et al. (2022), eqs 2-8.

    ``xyz`` in kpc, bar long axis along x; returns Msun / kpc^3.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    ax, ay, az = jnp.abs(x), jnp.abs(y), jnp.abs(z)
    R = jnp.hypot(x, y)
    r = jnp.sqrt(R**2 + z**2)

    p = _BAR_1
    a = (
        ((ax / p["x"]) ** p["c_perp"] + (ay / p["y"]) ** p["c_perp"])
        ** (p["c_par"] / p["c_perp"])
        + (az / p["z"]) ** p["c_par"]
    ) ** (1 / p["c_par"])
    a_p = jnp.hypot((x + p["c"] * z) / p["x_c"], y / p["y_c"])
    a_m = jnp.hypot((x - p["c"] * z) / p["x_c"], y / p["y_c"])
    rho_1 = (
        p["rho"]
        * _sech(a ** p["m"])
        * (1 + p["alpha"] * (jnp.exp(-(a_p ** p["n"])) + jnp.exp(-(a_m ** p["n"]))))
        * jnp.exp(-((r / p["r_cut"]) ** 2))
    )

    def long(p: dict[str, float]) -> gt.BtSz0:
        a = ((ax / p["x"]) ** p["c_perp"] + (ay / p["y"]) ** p["c_perp"]) ** (
            1 / p["c_perp"]
        )
        # At R = 0 one of the cutoffs is exp(-inf) = 0, the density's limit.
        cut = (R / p["R_out"]) ** p["n_out"] + (p["R_in"] / R) ** p["n_in"]
        out: gt.BtSz0 = (
            p["rho"] * jnp.exp(-(a ** p["n"]) - cut) * _sech(z / p["z"]) ** 2
        )
        return out

    return rho_1, long(_BAR_2), long(_BAR_3)


@jax.jit
def _density(xyz: gt.BtSz3, t: gt.BBtSz0, /) -> gt.BtSz0:  # noqa: ARG001
    rho_1, rho_2, rho_3 = _components(xyz)
    return rho_1 + rho_2 + rho_3


_R_MIN: Final = u.Q(1e-2, "kpc")
_R_MAX: Final = u.Q(100.0, "kpc")

# Coefficients proportional to the density; the rest are exponents and knots.
_LINEAR: Final = (
    "phi_lm",
    "dphi_lm",
    "d2phi_lm",
    "phi_asympt_scales",
    "rho_residual_lm",
    "drho_residual_lm",
    "rho_amplitude",
)
_SHAPE: Final = ("r_knots", "phi_asympt_powers", "rho_alpha")


@final
class SormaniEtAl2022BarPotential(AbstractMultipoleProfilePotential):
    r"""The Milky Way's bar of Sormani et al. (2022).

    An analytic fit to the stellar density of the made-to-measure bar of
    Portail et al. (2017), and the bar of the Milky Way potential of Hunter et
    al. (2024). The density is the sum of the three barred components of
    Sormani et al. (2022), eqs 2-8 with their Table 1. In their words the
    first two together are the X-shaped boxy bar and the third is the long
    bar. Their axisymmetric disc (eq. 9) is not part of the bar and is not
    included. The bar's long axis is along :math:`x`, and
    the density is symmetric under :math:`x \to -x`, :math:`y \to -y` and
    :math:`z \to -z`.

    The potential has no closed form. It is a `MultipoleProfilePotential`
    expansion of that density, with the even-:math:`l`, even-:math:`m` modes
    of `Symmetry.PLANE_REFLECTION`. Build it with `build`; `model_density` is
    the analytic density it expands, while ``density`` is the expansion's own.

    The parameters are the authors' implementation in Agama
    (``py/example_mw_bar_potential.py``, which Hunter et al. 2024 also use),
    whose values round to every entry of Table 1. They give component masses
    of 1.277, 0.333 and 0.215 :math:`\times 10^{10}\,M_\odot`, the paper's
    1.28, 0.33 and 0.22, and :math:`1.825 \times 10^{10}\,M_\odot` in total.
    The rounded Table 1 values give 1.274, 0.337 and 0.217.

    The bar is static and points along :math:`x`. Give it an orientation, or a
    pattern speed, by wrapping it in a frame transformation; see the example.

    References
    ----------
    - Sormani, M. C., Gerhard, O., Portail, M., Vasiliev, E., & Clarke, J.
      2022, MNRAS, 514, L1, doi:10.1093/mnrasl/slac046, arXiv:2204.13114
    - Portail, M., Gerhard, O., Wegg, C., & Ness, M. 2017, MNRAS, 465, 1621,
      doi:10.1093/mnras/stw2819
    - Hunter, G. H., Sormani, M. C., Beckmann, J. P., et al. 2024, A&A, 692,
      A216, doi:10.1051/0004-6361/202450000, arXiv:2403.18000

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> bar = gp.SormaniEtAl2022BarPotential.build()
    >>> bar.l_max, bar.symmetry
    (24, <Symmetry.PLANE_REFLECTION: 'plane_reflection'>)

    The potential is deeper along the bar's long axis, x, than along y:

    >>> t = u.Q(0, "Gyr")
    >>> x, y = u.Q([3.0, 0, 0], "kpc"), u.Q([0, 3.0, 0], "kpc")
    >>> bool(bar.potential(x, t) < bar.potential(y, t))
    True

    ``m_tot`` rescales it:

    >>> heavy = gp.SormaniEtAl2022BarPotential.build(m_tot=u.Q(2e10, "Msun"))

    A fixed orientation is a rotation. Here the long axis is turned 25 degrees
    counter-clockwise from x:

    >>> import coordinax as cx
    >>> import quaxed.numpy as jnp
    >>> rot = cx.ops.GalileanRotation.from_euler("z", u.Q(25, "deg"))
    >>> tilted = gp.TransformedPotential(bar, rot)
    >>> phi = u.Q(25, "deg")
    >>> along = u.Q(3.0, "kpc") * jnp.stack([jnp.cos(phi), jnp.sin(phi), 0])
    >>> tilted.potential(along, t)
    Q(-0.03453026, 'kpc2 / Myr2')
    >>> bar.potential(x, t)
    Q(-0.03453026, 'kpc2 / Myr2')

    A rotating bar wraps it the same way, with a time-dependent rotation.

    """

    @classmethod
    def build(
        cls,
        m_tot: Any = None,
        *,
        l_max: int = 24,
        n_r: int = 128,
        n_theta: int = 96,
        n_phi: int = 128,
        r_min: Any = _R_MIN,
        r_max: Any = _R_MAX,
        units: Any = "galactic",
        constants: Any = default_constants,
    ) -> "SormaniEtAl2022BarPotential":
        """Expand the bar's density in multipoles.

        Parameters
        ----------
        m_tot : Quantity["mass"], optional
            Total mass of the bar. The density is rescaled to it; the default
            is the model's own, 1.8252e10 Msun. Traceable: the expansion is
            built once at that mass and its coefficients are scaled, so
            ``m_tot`` can be a `jax.jit` argument or be differentiated through.
        l_max : int, optional
            Maximum multipole order. See the Notes for the accuracy.
        n_r : int, optional
            Number of log-spaced radial knots.
        n_theta, n_phi : int, optional
            Angular quadrature nodes. The defaults are set by the thin long
            bar, not by ``l_max``: see the Notes.
        r_min, r_max : Quantity["length"], optional
            The radial grid. Outside it the potential continues as a power
            law per mode, which is exact beyond the bar.
        units : AbstractUnitSystem, optional
            Unit system of the result.
        constants : Mapping, optional
            Physical constants.

        Notes
        -----
        The X-shaped bar is thick and converges quickly in :math:`l`. The long
        bar is a thin sheet (scale height 0.25 kpc) with sharp cylindrical
        cutoffs, so it needs high :math:`l` near it. Relative error of the
        force against direct summation of the density (to ~1e-3), for the
        default grid:

        ================ ======= ======= ======= =======
        position (kpc)   l=8     l=16    l=24    l=32
        ================ ======= ======= ======= =======
        (2, 1, 0.1)      1.9e-2  2.1e-3  7.8e-4
        (3, 0, 0)        1.7e-2  1.3e-2  5.0e-3  2.1e-3
        (3, 0, 0.3)      2.0e-1  4.6e-2  3.2e-3  5.3e-3
        (4, 2, 0.3)      3.6e-2  1.6e-2  5.7e-3
        (5, 0, 0.5)      1.3e-1  3.7e-2  9.6e-3  1.6e-2
        (6, 3, 0.5)      2.9e-3  9.5e-4  4.5e-4
        (8, 0, 1)        5.8e-3  1.5e-3  1.2e-3  7.0e-4
        ================ ======= ======= ======= =======

        Beyond the bar the expansion converges fast: ``l_max = 16`` agrees
        with ``l_max = 32`` to 6e-7 at (10, 5, 2) kpc. The default, 24, is
        good to 1% at every position measured. Its density, the Laplacian of
        the potential, matches the analytic one to 1e-5 within 1 kpc and 1e-2
        at 2 kpc, but rings around the thin long bar, by tens of percent at
        these ``l_max``.

        The angular rule has to resolve the long bar's thickness even for
        the monopole. With the 10 x 18 nodes ``l_max = 4`` would get by
        default the total mass is 8% short; the defaults here get it to 2e-4.
        The build takes about 20 s.
        """
        base = MultipoleProfilePotential.from_density(
            _density,
            r_min=r_min,
            r_max=r_max,
            n_r=n_r,
            l_max=l_max,
            n_theta=n_theta,
            n_phi=n_phi,
            symmetry=Symmetry.PLANE_REFLECTION,
            units="galactic",
            constants=constants,
        )
        scale = 1.0 if m_tot is None else u.ustrip("Msun", m_tot) / M_TOT
        t0 = u.Q(0.0, "Gyr")
        return cls(
            **{k: getattr(base, k)(t0) * scale for k in _LINEAR},
            **{k: getattr(base, k)(t0) for k in _SHAPE},
            l_max=l_max,
            symmetry=Symmetry.PLANE_REFLECTION,
            units=units,
            constants=constants,
        )

    @staticmethod
    def model_density(xyz: Any, /, m_tot: Any = None) -> u.Quantity:
        """Return the analytic density, eqs 2-8 of Sormani et al. (2022).

        ``xyz`` is a position Quantity with shape ``(..., 3)``; ``m_tot``
        rescales as in `build`.

        >>> import unxt as u
        >>> import galax.potential as gp
        >>> gp.SormaniEtAl2022BarPotential.model_density(u.Q([0.0, 0, 0], "kpc"))
        Q(7.12038574e+09, 'solMass / kpc3')

        """
        rho = _density(u.ustrip("kpc", xyz), None)
        scale = 1.0 if m_tot is None else u.ustrip("Msun", m_tot) / M_TOT
        return u.Q(rho * scale, "Msun / kpc3")
