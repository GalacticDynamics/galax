r"""The Koposov et al. (2023) Milky Way dark-matter halo."""

__all__ = ["KoposovEtAl2023HaloPotential"]

import functools as ft

from jaxtyping import Array
from typing import Any, Final, final

import jax

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue
from xmmutablemap import ImmutableMap

import galax.potential.custom_types as gt
from .multipole_profile import MultipoleProfilePotential
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractSinglePotential
from galax.potential._src.jax import vectorize_method
from galax.potential._src.params.base import AbstractParameter
from galax.potential._src.params.field import ParameterField

DEFAULT_R_Q: Final = u.Q(30.0, "kpc")
DEFAULT_R_CUT: Final = u.Q(500.0, "kpc")
DEFAULT_R_MIN: Final = u.Q(1e-2, "kpc")
DEFAULT_R_MAX: Final = u.Q(2e3, "kpc")


@final
class KoposovEtAl2023HaloPotential(AbstractSinglePotential):
    r"""Flattened generalized-NFW halo of Koposov et al. (2023).

    The Milky Way dark-matter halo of Koposov, Erkal, Li et al. (2023, Sec.
    5.1, Eqs. 7-11): a generalized NFW density with a Gaussian cutoff,
    flattened in the *density* along a free symmetry axis,

    .. math::

        \rho = \frac{\rho_0}{(m/r_s)^\gamma (1 + m/r_s)^{\beta - \gamma}}
               \exp\left[-(m / r_\mathrm{cut})^2\right],
        \qquad
        m^2 = x'^2 + y'^2 + \frac{z'^2}{q(r)^2},

    where :math:`\mathbf{x}' = R\,\mathbf{x}` is the position in the halo's
    frame, :math:`R` turns ``axis`` onto :math:`z'`, and :math:`r =
    |\mathbf{x}|`. The axis ratio changes with radius as in Vasiliev,
    Belokurov & Erkal (2021, Eq. 8),

    .. math::

        q(r) = \frac{q_\mathrm{in} + q_\mathrm{out} (r / r_q)^2}
                    {1 + (r / r_q)^2},

    so ``q_in == q_out`` is Koposov et al.'s constant flattening. ``q < 1`` is
    oblate along ``axis`` and ``q > 1`` prolate.

    **Mass scale.** ``m`` is galax's characteristic mass,
    :math:`m = 4 \pi \rho_0 r_s^3`, as in `gNFWPotential` and `NFWPotential`:
    with ``gamma = 1``, ``beta = 3``, ``q_in = q_out = 1`` and ``r_cut ->
    inf`` this is `NFWPotential` with the same ``m`` and ``r_s``. Koposov et
    al. (Eq. 8) normalize by :math:`M_\mathrm{NFW}` at a fixed concentration
    :math:`c_\mathrm{NFW} = 15` instead, which is

    .. math::

        m = \frac{M_\mathrm{NFW}}
                 {\ln(1 + c_\mathrm{NFW}) - c_\mathrm{NFW} / (1 + c_\mathrm{NFW})}
          = M_\mathrm{NFW} / 1.8351 \quad (c_\mathrm{NFW} = 15).

    **Axis.** ``axis`` is Koposov et al.'s :math:`(x_\mathrm{NFW},
    y_\mathrm{NFW}, z_\mathrm{NFW})`, which they map to the angles of their
    Eqs. 10-11; it need not be normalized, and ``axis`` and ``-axis`` give the
    same halo. Their rotation (Eq. 12) and the one used here both take the axis
    to :math:`z'` and differ only by a rotation about it, which leaves an
    axisymmetric halo unchanged.

    **Potential.** There is no closed form. `from_params` expands the density
    in the halo's frame with `MultipoleProfilePotential.from_density` (even
    :math:`l` and :math:`m = 0` only, since the halo is axisymmetric and
    reflection-symmetric there), and the expansion is evaluated at
    :math:`R\,\mathbf{x}`. Koposov et al. also evaluate their halo as an
    expansion, with GalPot (Dehnen & Binney 1998). ``density`` is the
    expansion's reconstructed density, which is consistent with its
    potential.

    Accuracy is set mostly by ``l_max``. For a strongly flattened, tilted
    halo (``q_in = 0.55``, ``q_out = 0.9``, ``axis = (0.25, 0.45, 0.85)``,
    ``gamma = 1.27``, ``beta = 2.97``, ``r_s = 16.1`` kpc) on the default
    grid, the largest relative error at five positions 0.37 to 180 kpc from
    the centre, against an ``l_max = 32``, ``n_r = 256`` build for the
    potential and force and against the closed-form density for Poisson's
    equation; with the build time after compilation (float64, one CPU):

    ======= ========= ========= ================================ ========
    l_max   potential force     :math:`\nabla^2\Phi / 4\pi G\rho - 1` build
    ======= ========= ========= ================================ ========
    4       1.4e-4    4.8e-3    2.1e-2                           25 ms
    8       3.0e-6    2.1e-4    9.8e-4                           0.2 s
    12      1.2e-7    1.2e-5    8.0e-5                           0.35 s
    16      5.6e-9    6.7e-7    5.7e-6                           0.8 s
    20      1.8e-10   4.1e-8    3.2e-7                           1.2 s
    ======= ========= ========= ================================ ========

    The default ``l_max = 12`` keeps forces to ~1e-5. ``n_r`` matters much
    less: in the spherical limit ``n_r = 64`` already gives forces to 2e-8.

    **Parameters and derivatives.** The instance holds the expansion and the
    axis, not the physical parameters. Every argument of `from_params`
    except the grid shape may be traced, so build the halo inside the
    function that `jax.jit` compiles or `jax.grad` differentiates; the
    expansion is rebuilt from the parameters there. ``axis`` enters only
    through :math:`R` and is a parameter of the instance, so it can also be
    changed or differentiated after the build.

    References
    ----------
    - Koposov, S. E., Erkal, D., Li, T. S., et al. 2023, MNRAS, 521, 4936,
      `arXiv:2211.04495 <https://arxiv.org/abs/2211.04495>`_.
    - Vasiliev, E., Belokurov, V., & Erkal, D. 2021, MNRAS, 501, 2279,
      `arXiv:2009.10726 <https://arxiv.org/abs/2009.10726>`_.

    Examples
    --------
    >>> import unxt as u
    >>> import galax.potential as gp

    >>> halo = gp.KoposovEtAl2023HaloPotential.from_params(
    ...     u.Q(5.4e11, "Msun"), u.Q(15.0, "kpc"),
    ...     q_in=0.6, q_out=0.9, axis=[0.2, 0.4, 0.9], units="galactic")
    >>> x = u.Q([8.0, 0.0, 1.0], "kpc")
    >>> halo.potential(x, u.Q(0.0, "Gyr")).round(5)
    Q(-0.09724, 'kpc2 / Myr2')

    The spherical, uncut limit is the NFW potential:

    >>> sph = gp.KoposovEtAl2023HaloPotential.from_params(
    ...     u.Q(5.4e11, "Msun"), u.Q(15.0, "kpc"), r_cut=u.Q(1e8, "kpc"),
    ...     units="galactic")
    >>> nfw = gp.NFWPotential(m=u.Q(5.4e11, "Msun"), r_s=u.Q(15.0, "kpc"),
    ...                       units="galactic")
    >>> bool(u.ustrip("", sph.potential(x, 0) / nfw.potential(x, 0) - 1) < 1e-6)
    True
    """

    expansion: MultipoleProfilePotential
    """The potential in the halo's frame, axisymmetric about its z-axis."""

    axis: AbstractParameter = ParameterField(  # type: ignore[assignment]
        default=u.Q([0.0, 0.0, 1.0], ""),
        dimensions="dimensionless",
        doc="Symmetry axis, shape ``(3,)``; need not be normalized.",
    )

    @classmethod
    def from_params(
        cls,
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
        axis: Any = (0.0, 0.0, 1.0),
        l_max: int = 12,
        n_r: int = 128,
        r_min: Any = DEFAULT_R_MIN,
        r_max: Any = DEFAULT_R_MAX,
        units: Any,
        constants: Any = default_constants,
    ) -> "KoposovEtAl2023HaloPotential":
        r"""Build the halo from its physical parameters.

        Parameters
        ----------
        m : Quantity['mass']
            Characteristic mass :math:`4 \pi \rho_0 r_s^3`; not the total
            mass (see the class docstring for Koposov et al.'s
            :math:`M_\mathrm{NFW}`).
        r_s : Quantity['length']
            Scale radius.
        gamma, beta : float, optional
            Inner and outer slopes (defaults 1 and 3, the NFW values).
        q_in, q_out : float, optional
            Axis ratio along ``axis`` well inside and well outside ``r_q``
            (default 1, spherical).
        r_q : Quantity['length'], optional
            Radius where the axis ratio changes over (default 30 kpc).
        r_cut : Quantity['length'], optional
            Gaussian cutoff radius (default 500 kpc, as Koposov et al.).
        axis : array-like, optional
            Symmetry axis, shape ``(3,)``, not necessarily normalized
            (default :math:`\hat{z}`).
        l_max : int, optional
            Maximum multipole order (default 12); only even orders are
            nonzero. See the class docstring for the accuracy it buys.
        n_r : int, optional
            Number of radial knots (default 128).
        r_min, r_max : Quantity['length'], optional
            Radial range of the expansion (default 0.01 to 2000 kpc). Outside
            it the potential continues as a fitted power law per mode; see
            `MultipoleProfilePotential`.
        units : AbstractUnitSystem
            Unit system.
        constants : Mapping, optional
            Physical constants (default from `default_constants`).

        Every argument except ``l_max``, ``n_r``, ``units`` and
        ``constants`` may be a `jax.jit` argument or be differentiated
        through. ``l_max`` and ``n_r`` set array shapes.
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
        expansion = _build(
            p,
            val(r_min, "length"),
            val(r_max, "length"),
            ImmutableMap(constants),
            n_r=n_r,
            l_max=l_max,
            units=usys,
        )
        return cls(
            expansion=expansion,
            axis=u.Q(val(axis, "dimensionless"), ""),
            units=usys,
            constants=constants,
        )

    @ft.partial(jax.jit)
    def _potential(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        rot = rotation_to_z(self.params(t)["axis"])
        return self.expansion._potential(xyz @ rot.T, t)  # type: ignore[no-any-return]  # noqa: SLF001

    @vectorize_method(signature="(3),()->(3)")
    @ft.partial(jax.jit)
    def _gradient(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz3:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        rot = rotation_to_z(self.params(t)["axis"])
        out: gt.BBtSz3 = rot.T @ self.expansion._gradient(rot @ xyz, t)  # noqa: SLF001
        return out

    @ft.partial(jax.jit)
    def _density(self, xyz: gt.BBtQorVSz3, t: gt.BBtQorVSz0, /) -> gt.BBtSz0:
        xyz = u.ustrip(AllowValue, self.units["length"], xyz)
        rot = rotation_to_z(self.params(t)["axis"])
        return self.expansion._density(xyz @ rot.T, t)  # type: ignore[no-any-return]  # noqa: SLF001


# ===================================================================


def rotation_to_z(axis: Array, /) -> Array:
    r"""Rotation matrix taking ``axis`` to :math:`\hat{z}`.

    Rodrigues' formula for the smallest such rotation. ``axis`` and ``-axis``
    describe the same halo, so the axis is first flipped into the upper
    hemisphere, which keeps the formula away from its pole at
    :math:`-\hat{z}`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> R = rotation_to_z(jnp.array([1.0, 2.0, -2.0]))
    >>> n = jnp.array([-1.0, -2.0, 2.0]) / 3
    >>> bool(jnp.allclose(R @ n, jnp.array([0.0, 0.0, 1.0])))
    True
    """
    n = axis / jnp.linalg.norm(axis)
    n = jnp.where(n[2] < 0, -n, n)
    v = jnp.array([n[1], -n[0], 0.0])  # n x z
    k = jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    out: Array = jnp.eye(3) + k + k @ k / (1.0 + n[2])
    return out


def log_density(p: dict[str, Any], xyz: gt.BBtSz3, t: gt.BBtSz0, /) -> gt.BBtSz0:
    r"""Log of the density at ``xyz`` in the halo's frame.

    Written as a sum of logs, as `MultipoleProfilePotential.from_density`
    recommends for derivatives of the build, with the flattened radius as
    :math:`\ln m = \ln r + \tfrac12 \ln[1 + (q^{-2} - 1) z^2 / r^2]`.
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


@ft.partial(jax.jit, static_argnames=("n_r", "l_max", "units"))
def _build(
    p: dict[str, Any],
    r_min: Any,
    r_max: Any,
    constants: Any,
    /,
    *,
    n_r: int,
    l_max: int,
    units: u.AbstractUnitSystem,
) -> MultipoleProfilePotential:
    """Expand the halo-frame density; jitted so re-construction is cached."""
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
