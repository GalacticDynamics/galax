r"""A flattened generalized-NFW halo, and the Koposov et al. (2023) halo."""

__all__ = ["doi_10_1093_mnras_stad551", "flattened_gnfw_halo", "log_density"]

import functools as ft

from jaxtyping import Array
from typing import Any, Final

import jax

import coordinax.ops as cxo
import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .multipole_profile import MultipoleProfilePotential
from galax.potential._src.base import default_constants
from galax.potential._src.xfm.xop import TransformedPotential

DEFAULT_R_Q: Final = u.Q(30.0, "kpc")
DEFAULT_R_CUT: Final = u.Q(500.0, "kpc")
DEFAULT_R_MIN: Final = u.Q(1e-2, "kpc")
DEFAULT_R_MAX: Final = u.Q(2e3, "kpc")


def log_density(p: dict[str, Any], xyz: gt.BBtSz3, t: gt.BBtSz0, /) -> gt.BBtSz0:
    r"""Log of the flattened generalized-NFW density.

    .. math::

        \rho = \frac{\rho_0}{(m/r_s)^\gamma (1 + m/r_s)^{\beta - \gamma}}
               \exp\left[-(m / r_\mathrm{cut})^2\right],
        \qquad
        m^2 = x^2 + y^2 + \frac{z^2}{q(r)^2},
        \qquad
        \rho_0 = \frac{m_\mathrm{s}}{4\pi r_s^3},

    with the axis ratio changing with the spherical radius :math:`r` as

    .. math::

        q(r) = \frac{q_\mathrm{in} + q_\mathrm{out} (r / r_q)^2}
                    {1 + (r / r_q)^2}.

    ``p`` holds ``m`` (:math:`m_\mathrm{s}`), ``r_s``, ``gamma``, ``beta``,
    ``q_in``, ``q_out``, ``r_q`` and ``r_cut`` as bare arrays in one unit
    system; ``xyz`` has shape ``(..., 3)`` and ``t`` is ignored. The flattened
    radius is written as :math:`\ln m = \ln r + \tfrac12 \ln[1 + (q^{-2} - 1)
    z^2 / r^2]` and the density as a sum of logs, as
    `MultipoleProfilePotential.from_density` recommends for derivatives.
    """
    del t
    r2 = jnp.sum(xyz**2, axis=-1)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    q = p["q_in"] + (p["q_out"] - p["q_in"]) * r2 / (r2 + p["r_q"] ** 2)
    log_m = 0.5 * jnp.log(r2) + 0.5 * jnp.log1p(
        (q**-2 - 1) * jnp.where(r2 > 0, xyz[..., 2] ** 2 / safe_r2, 0.0)
    )
    log_s = log_m - jnp.log(p["r_s"])
    log_rho0 = jnp.log(p["m"]) - jnp.log(4 * jnp.pi) - 3 * jnp.log(p["r_s"])
    out: gt.BBtSz0 = (
        log_rho0
        - p["gamma"] * log_s
        - (p["beta"] - p["gamma"]) * jnp.logaddexp(0.0, log_s)
        - jnp.exp(2 * (log_m - jnp.log(p["r_cut"])))
    )
    return out


def flattened_gnfw_halo(
    m: Any,
    r_s: Any,
    /,
    *,
    gamma: Any = 1.0,
    beta: Any = 3.0,
    q_in: Any = 1.0,
    q_out: Any = 1.0,
    r_q: Any = DEFAULT_R_Q,
    r_cut: Any = DEFAULT_R_CUT,
    l_max: int = 12,
    n_r: int = 128,
    r_min: Any = DEFAULT_R_MIN,
    r_max: Any = DEFAULT_R_MAX,
    units: Any,
    constants: Any = default_constants,
) -> MultipoleProfilePotential:
    r"""Generalized-NFW halo flattened along :math:`z`, as a multipole expansion.

    The density is `log_density`: a generalized NFW with a Gaussian cutoff,
    flattened in the *density* along :math:`z` with an axis ratio that changes
    with radius as in Vasiliev, Belokurov & Erkal (2021, Eq. 8). ``q < 1`` is
    oblate and ``q > 1`` prolate; ``q_in == q_out`` is a constant flattening.
    With ``gamma = 1``, ``beta = 3``, ``q_in = q_out = 1`` and ``r_cut ->
    inf`` it is `NFWPotential` with the same ``m`` and ``r_s``. To tilt the
    halo, wrap it in a `TransformedPotential` with a rotation, as
    `galax.potential.papers.doi_10_1093_mnras_stad551` does.

    There is no closed-form potential. The density is expanded with
    `MultipoleProfilePotential.from_density`, keeping even :math:`l` and
    :math:`m = 0` (``symmetry="zrotation_zreflection"``). The returned
    potential's ``density`` is the expansion's reconstructed density, which is
    consistent with its potential.

    Accuracy is set mostly by ``l_max``. For a strongly flattened halo
    (``q_in = 0.55``, ``q_out = 0.9``) on the default grid, the tests check
    that :math:`\nabla^2\Phi / 4\pi G\rho - 1` stays within 5e-2, 2e-3 and
    3e-4 at ``l_max`` 4, 8 and 12, at positions 0.37 to 180 kpc from the
    centre. ``n_r`` matters much less than ``l_max``. The build is compiled once per
    grid shape and unit system; building again with new parameter values
    reuses it.

    Parameters
    ----------
    m : Quantity['mass']
        Characteristic mass :math:`4 \pi \rho_0 r_s^3`, as in `gNFWPotential`
        and `NFWPotential`; not the total mass.
    r_s : Quantity['length']
        Scale radius.
    gamma, beta : float, optional
        Inner and outer slopes (defaults 1 and 3, the NFW values).
    q_in, q_out : float, optional
        Axis ratio along :math:`z` well inside and well outside ``r_q``
        (default 1, spherical).
    r_q : Quantity['length'], optional
        Radius where the axis ratio changes over (default 30 kpc).
    r_cut : Quantity['length'], optional
        Gaussian cutoff radius (default 500 kpc).
    l_max : int, optional
        Maximum multipole order (default 12); only even orders are nonzero.
    n_r : int, optional
        Number of radial knots (default 128).
    r_min, r_max : Quantity['length'], optional
        Radial range of the expansion (default 0.01 to 2000 kpc). Outside it
        the potential continues as a fitted power law per mode; see
        `MultipoleProfilePotential`.
    units : AbstractUnitSystem
        Unit system.
    constants : Mapping, optional
        Physical constants (default from `default_constants`).

    Every argument except ``l_max``, ``n_r``, ``units`` and ``constants`` may
    be a `jax.jit` argument or be differentiated through; those four set array
    shapes or are static.

    References
    ----------
    - Vasiliev, E., Belokurov, V., & Erkal, D. 2021, MNRAS, 501, 2279,
      `arXiv:2009.10726 <https://arxiv.org/abs/2009.10726>`_.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    The spherical, uncut limit is the NFW potential:

    >>> halo = gp.flattened_gnfw_halo(u.Q(5.4e11, "Msun"), u.Q(15.0, "kpc"),
    ...     r_cut=u.Q(1e8, "kpc"), units="galactic")
    >>> nfw = gp.NFWPotential(m=u.Q(5.4e11, "Msun"), r_s=u.Q(15.0, "kpc"),
    ...                       units="galactic")
    >>> x = u.Q([8.0, 0.0, 1.0], "kpc")
    >>> bool(u.ustrip("", halo.potential(x, 0) / nfw.potential(x, 0) - 1) < 1e-6)
    True
    >>> halo.symmetry
    <Symmetry.ZROTATION_ZREFLECTION: 'zrotation_zreflection'>
    """
    usys = u.unitsystem(units)

    def val(x: Any, dim: str) -> Array:
        out: Array = jnp.asarray(u.ustrip(AllowValue, usys[dim], x), dtype=float)
        return out

    p = {
        "m": val(m, "mass"),
        "r_s": val(r_s, "length"),
        "gamma": val(gamma, "dimensionless"),
        "beta": val(beta, "dimensionless"),
        "q_in": val(q_in, "dimensionless"),
        "q_out": val(q_out, "dimensionless"),
        "r_q": val(r_q, "length"),
        "r_cut": val(r_cut, "length"),
    }
    out: MultipoleProfilePotential = _build(
        p,
        val(r_min, "length"),
        val(r_max, "length"),
        n_r=n_r,
        l_max=l_max,
        units=usys,
        constants=ImmutableMap(constants),
    )
    return out


@ft.partial(jax.jit, static_argnames=("n_r", "l_max", "units", "constants"))
def _build(
    p: dict[str, Any],
    r_min: Any,
    r_max: Any,
    /,
    *,
    n_r: int,
    l_max: int,
    units: u.AbstractUnitSystem,
    constants: ImmutableMap[str, Any],
) -> MultipoleProfilePotential:
    """Expand the density; jitted so a rebuild with new values is cached.

    ``constants`` is static: galax builds its constants as static quantities,
    which cannot hold traced values.
    """
    return MultipoleProfilePotential.from_density(
        log_rho=ft.partial(log_density, p),
        r_min=r_min,
        r_max=r_max,
        n_r=n_r,
        l_max=l_max,
        symmetry="zrotation_zreflection",
        units=units,
        constants=constants,
    )


def doi_10_1093_mnras_stad551(
    m_nfw: Any,
    r_s: Any,
    /,
    *,
    q: Any = 1.0,
    gamma: Any = 1.0,
    beta: Any = 3.0,
    axis: Any = (0.0, 0.0, 1.0),
    l_max: int = 12,
    n_r: int = 128,
    units: Any,
    constants: Any = default_constants,
) -> TransformedPotential:
    r"""Milky Way dark-matter halo of Koposov et al. (2023).

    Koposov, Erkal, Li et al. (2023, MNRAS 521, 4936; Sec. 5.1, Eqs. 7-12)
    model the halo as a generalized NFW with a Gaussian cutoff at
    :math:`r_\mathrm{cut} = 500` kpc, flattened in the density by a constant
    axis ratio ``q`` along a free symmetry axis. This returns
    `flattened_gnfw_halo` with ``q_in = q_out = q``, rotated so that its
    :math:`z`-axis points along ``axis``.

    **Mass.** ``m_nfw`` is their :math:`M_\mathrm{NFW}` (Eq. 8), which fixes
    :math:`c_\mathrm{NFW} = 15`, so the characteristic mass of
    `flattened_gnfw_halo` is

    .. math::

        m = \frac{M_\mathrm{NFW}}
                 {\ln(1 + c_\mathrm{NFW}) - c_\mathrm{NFW} / (1 + c_\mathrm{NFW})}
          = M_\mathrm{NFW} / 1.8351.

    **Axis.** ``axis`` is their :math:`(x_\mathrm{NFW}, y_\mathrm{NFW},
    z_\mathrm{NFW})`; it need not be normalized but must be nonzero, and
    ``axis`` and ``-axis`` give the same halo. The returned
    `TransformedPotential` moves the z-aligned halo by the smallest active
    rotation (Rodrigues' formula) that turns its :math:`z`-axis onto ``axis``
    or ``-axis``, whichever is in the upper hemisphere. Its inverse, like
    their :math:`R` (Eq. 12, :math:`\mathbf{x}' = R\,\mathbf{x}` in Eq. 9),
    takes that axis to :math:`z'`; the two differ only by a rotation about the
    axis, which leaves the axisymmetric halo unchanged. Unlike the angles of
    their Eqs. 10-11, this rotation is smooth at :math:`\pm\hat{z}`, so
    derivatives with respect to ``axis`` stay finite there.

    Koposov et al. evaluate the halo with GalPot (Dehnen & Binney 1998); here
    it is a multipole expansion (see `flattened_gnfw_halo` for its accuracy).

    Every argument except ``l_max``, ``n_r``, ``units`` and ``constants`` may
    be traced by `jax.jit` or differentiated through.

    References
    ----------
    - Koposov, S. E., Erkal, D., Li, T. S., et al. 2023, MNRAS, 521, 4936,
      `doi:10.1093/mnras/stad551 <https://doi.org/10.1093/mnras/stad551>`_,
      `arXiv:2211.04495 <https://arxiv.org/abs/2211.04495>`_.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    Their Table 2's oblate :math:`M_\mathrm{NFW}`, :math:`r_s`, ``q``,
    ``gamma`` and ``beta``, with an illustrative axis:

    >>> halo = gp.papers.doi_10_1093_mnras_stad551(
    ...     u.Q(11.22e11, "Msun"), u.Q(16.1, "kpc"), q=0.55, gamma=1.27,
    ...     beta=2.97, axis=[0.2, 0.4, 0.9], units="galactic")
    >>> halo.potential(u.Q([8.0, 0.0, 1.0], "kpc"), u.Q(0.0, "Gyr")).round(5)
    Q(-0.1026, 'kpc2 / Myr2')
    """
    c = 15.0
    halo = flattened_gnfw_halo(
        m_nfw / (jnp.log(1 + c) - c / (1 + c)),
        r_s,
        gamma=gamma,
        beta=beta,
        q_in=q,
        q_out=q,
        l_max=l_max,
        n_r=n_r,
        units=units,
        constants=constants,
    )
    n = jnp.asarray(axis, dtype=float)
    n = n / jnp.linalg.norm(n)
    n = jnp.where(n[2] < 0, -n, n)  # same halo; keeps away from the pole at -z
    k = jnp.array([[0.0, 0.0, n[0]], [0.0, 0.0, n[1]], [-n[0], -n[1], 0.0]])
    rot = jnp.eye(3) + k + k @ k / (1.0 + n[2])  # active, takes z to n
    return TransformedPotential(base_potential=halo, xop=cxo.GalileanRotation(rot))
