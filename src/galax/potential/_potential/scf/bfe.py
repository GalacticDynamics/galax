"""Self-Consistent Field Potential."""

__all__ = ["SCFPotential", "STnlmSnapshotParameter"]

from collections.abc import Callable
from dataclasses import KW_ONLY
from typing import Any

import astropy.units as u
import equinox as eqx
import jax.numpy as xp
from jaxtyping import Array, Float
from typing_extensions import override

from galax.potential._potential.core import AbstractPotential
from galax.potential._potential.param import AbstractParameter, ParameterField
from galax.typing import (
    ArrayAnyShape,
    FloatLike,
    FloatOrIntScalar,
    FloatScalar,
    Vec3,
    VecN,
)
from galax.utils import partial_jit

from .bfe_helper import phi_nl_vec
from .bfe_helper import rho_nl as calculate_rho_nl
from .coeffs import compute_coeffs_discrete
from .gegenbauer import GegenbauerCalculator
from .utils import cartesian_to_spherical, real_Ylm

##############################################################################


class SCFPotential(AbstractPotential):
    r"""Self-Consistent Field (SCF) potential.

    A gravitational potential represented as a basis function expansion.  This
    uses the self-consistent field (SCF) method of Hernquist & Ostriker (1992)
    and Lowing et al. (2011), and represents all coefficients as real
    quantities.

    Parameters
    ----------
    m : numeric
        Scale mass.
    r_s : numeric
        Scale length.
    Snlm : Array[float, (nmax+1, lmax+1, lmax+1)] | Callable
        Array of coefficients for the cos() terms of the expansion.  This should
        be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`, where `nmax` is the
        number of radial expansion terms and `lmax` is the number of spherical
        harmonic `l` terms.  If a callable is provided, it should accept a
        single argument `t` and return the array of coefficients for that time.
    Tnlm : Array[float, (nmax+1, lmax+1, lmax+1)] | Callable
        Array of coefficients for the sin() terms of the expansion.  This should
        be a 3D array with shape `(nmax+1, lmax+1, lmax+1)`, where `nmax` is the
        number of radial expansion terms and `lmax` is the number of spherical
        harmonic `l` terms.  If a callable is provided, it should accept a
        single argument `t` and return the array of coefficients for that time.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the length,
        mass, time, and angle units.
    """

    m: AbstractParameter = ParameterField(dimensions="mass")  # type: ignore[assignment]
    r_s: AbstractParameter = ParameterField(dimensions="length")  # type: ignore[assignment]
    Snlm: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]
    Tnlm: AbstractParameter = ParameterField(dimensions="dimensionless")  # type: ignore[assignment]

    nmax: int = eqx.field(init=False, static=True, repr=False)
    lmax: int = eqx.field(init=False, static=True, repr=False)
    _ultra_sph: GegenbauerCalculator = eqx.field(init=False, static=True, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # shape parameters
        shape = self.Snlm(0).shape
        object.__setattr__(self, "nmax", shape[0] - 1)
        object.__setattr__(self, "lmax", shape[1] - 1)

        # gegenbauer calculator
        object.__setattr__(self, "_ultra_sph", GegenbauerCalculator(self.nmax))

    # ==========================================================================

    @partial_jit()
    # @eqx.filter_vmap(in_axes=(None, 1, None))  # type: ignore[misc]  # on `q` axis 1
    def _potential_energy(
        self, xyz: Float[Array, "N 3"] | Vec3, /, t: FloatOrIntScalar
    ) -> VecN | FloatScalar:
        r, theta, phi = cartesian_to_spherical(xyz).T
        r_s = self.r_s(t)
        s = xp.atleast_1d(r / r_s)  # ([n],[l],[m],[N])
        theta = xp.atleast_1d(theta)[None, None, None]  # ([n],[l],[m],[N])
        phi = xp.atleast_1d(phi)[None, None, None]  # ([n],[l],[m],[N])

        ns = xp.arange(self.nmax + 1)[:, None, None]  # (n, [l], [m])
        ls = xp.arange(self.lmax + 1)[None, :, None]  # ([n], l, [m])
        phi_nl = phi_nl_vec(s, ns, ls, self._ultra_sph)  # (n, l, [m], N)

        li, mi = xp.tril_indices(self.lmax + 1)  # (l*(l+1)//2,)
        shape = (1, self.lmax + 1, self.lmax + 1, 1)  # ([n], l, m, [N])
        midx = xp.zeros(shape, dtype=int).at[:, li, mi, 0].set(mi)  # ([n], l, m, [N])

        Ylm = xp.zeros(shape[:-1] + (len(s),))
        Ylm = Ylm.at[0, li, mi, :].set(
            real_Ylm(theta[:, 0, 0, :], li[..., None], mi[..., None])
        )

        Snlm = self.Snlm(t, r_s=r_s)[..., None]
        Tnlm = self.Tnlm(t, r_s=r_s)[..., None]

        out = (self._G * self.m(t) / r_s) * xp.sum(
            Ylm * phi_nl * (Snlm * xp.cos(midx * phi) + Tnlm * xp.sin(midx * phi)),
            axis=(0, 1, 2),
        )
        return out[0] if len(xyz.shape) == 1 else out

    # @partial_jit()
    # # @vectorize_method(signature="(3),()->()")
    # def _potential_energy(self, xyz: Vec3, /, t: FloatOrIntScalar) -> FloatScalar:
    #     """Compute the potential energy at the given position(s)."""
    #     out = self._potential_energy_helper(xp.atleast_2d(xyz), t)
    #     return out[0] if len(xyz.shape) == 1 else out

    # ==========================================================================

    # @partial_jit()
    # @eqx.filter_vmap(in_axes=(None, 1, None))  # type: ignore[misc]  # on `q` axis 1
    # def _gradient(self, q: Float[Array, "3"], /, t: jt.Array) -> jt.Array:
    #     """Compute the gradient."""
    #     r, theta, phi = cartesian_to_spherical(q)
    #     r_s = self.r_s(t)
    #     s = xp.atleast_1d(r / r_s)[:, None, None, None]
    #     theta = xp.atleast_1d(theta)[:, None, None, None]
    #     phi = xp.atleast_1d(phi)[:, None, None, None]

    #     ns = xp.arange(self.nmax + 1)[None, :, None, None]  # ([N], n, [l], [m])
    #     ls = xp.arange(self.lmax + 1)[None, None, :, None]  # ([N], [n], l, [m])
    #     phi_nl = calculate_phi_nl(s, ns, ls, gegenbauer=self._ultra_sph)
    #     dphi_nl_dr = phi_nl_grad(s, ns, ls, self._ultra_sph)

    #     li, mi = xp.tril_indices(self.lmax + 1)  # (l*(l+1)//2,)
    #     shape = (1, 1, self.lmax + 1, self.lmax + 1)
    #     lidx = xp.zeros(shape, dtype=int).at[:, :, li, mi].set(li)
    #     midx = xp.zeros(shape, dtype=int).at[:, :, li, mi].set(mi)
    #     mvalid = xp.zeros(shape).at[:, :, li, mi].set(1)  # m <= l
    #     Ylm = real_Ylm(lidx, midx, theta)
    #     dYlm_dtheta = calculate_dYlm_dtheta(lidx, midx, theta)

    #     Snlm = self.Snlm(t, r_s=r_s)[None]
    #     Tnlm = self.Tnlm(t, r_s=r_s)[None]

    #     grad_r = xp.sum(
    #         (mvalid * Ylm)
    #         * dphi_nl_dr
    #         * (Snlm * xp.cos(midx * phi) + Tnlm * xp.sin(midx * phi)),
    #         axis=(1, 2, 3),
    #     )
    #     grad_theta = (1 / s[:, 0, 0, 0]) * xp.sum(
    #         (mvalid * dYlm_dtheta)
    #         * phi_nl
    #         * (Snlm * xp.cos(midx * phi) + Tnlm * xp.sin(midx * phi)),
    #         axis=(1, 2, 3),
    #     )
    #     grad_phi = (1 / s[:, 0, 0, 0]) * xp.sum(
    #         (mvalid * Ylm / xp.sin(theta))
    #         * phi_nl
    #         * (Tnlm * xp.cos(midx * phi) - Snlm * xp.sin(midx * phi)),
    #         axis=(1, 2, 3),
    #     )
    #     return (self._G * self.m(t) / r_s) * xp.stack([grad_r, grad_theta, grad_phi],
    #             axis=-1)

    # @partial_jit()
    # def gradient(self, q: jt.Array, /, t: jt.Array) -> jt.Array:
    #     """Compute the potential energy at the given position(s)."""
    #     out = self._gradient(expand_dim1(q), t)
    #     return out[0, 0] if len(q.shape) == 1 else out[:, 0]  # TODO: fix this

    @partial_jit()
    @eqx.filter_vmap(in_axes=(None, 1, None))  # type: ignore[misc]  # on `q` axis 1
    def density(
        self, q: Float[Array, "3 N"], /, t: Float[Array, "1"]
    ) -> Float[Array, "N"]:  # type: ignore[name-defined]
        """Compute the density at the given position(s)."""
        r, theta, phi = cartesian_to_spherical(q)
        r_s = self.r_s(t)
        s = xp.atleast_1d(r / r_s)[:, None, None, None]
        theta = xp.atleast_1d(theta)[:, None, None, None]
        phi = xp.atleast_1d(phi)[:, None, None, None]

        ns = xp.arange(self.nmax + 1)[:, None, None]  # (n, [l], [m])
        ls = xp.arange(self.lmax + 1)[None, :, None]  # ([n], l, [m])

        phi_nl = calculate_rho_nl(s, ns[None], ls[None], gegenbauer=self._ultra_sph)

        li, mi = xp.tril_indices(self.lmax + 1)  # (l*(l+1)//2,)
        shape = (1, 1, self.lmax + 1, self.lmax + 1)
        midx = xp.zeros(shape, dtype=int).at[:, :, li, mi].set(mi)
        Ylm = xp.zeros((len(theta), 1, self.lmax + 1, self.lmax + 1))
        Ylm = Ylm.at[:, li, mi, :].set(real_Ylm(li[None], mi[None], theta[:, :, 0, 0]))

        Snlm = self.Snlm(t, r_s=r_s)[None]
        Tnlm = self.Tnlm(t, r_s=r_s)[None]

        out = (self._G * self.m(t) / r_s) * xp.sum(
            Ylm * phi_nl * (Snlm * xp.cos(midx * phi) + Tnlm * xp.sin(midx * phi)),
            axis=(1, 2, 3),
        )
        return out[0] if len(q.shape) == 1 else out


# =============================================================================


class STnlmSnapshotParameter(AbstractParameter):
    """Parameter for the STnlm coefficients."""

    snapshot: Callable[  # type: ignore[name-defined]
        [Float[Array, "N"]],
        tuple[Float[Array, "3 N"], Float[Array, "N"]],
    ]
    """Cartesian coordinates of the snapshot.

    This should be a callable that accepts a single argument `t` and returns
    the cartesian coordinates and the masses of the snapshot at that time.
    """

    nmax: int = eqx.field(static=True, converter=int)
    """Radial expansion term."""

    lmax: int = eqx.field(static=True, converter=int)
    """Spherical harmonic term."""

    _: KW_ONLY
    unit: u.Unit = eqx.field(default=u.one, static=True, converter=u.Unit)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.unit != u.one:
            msg = "unit must be dimensionless"
            raise ValueError(msg)

    @override
    def __call__(  # type: ignore[override]
        self, t: FloatLike, *, r_s: float, **kwargs: Any
    ) -> tuple[ArrayAnyShape, ArrayAnyShape]:
        """Return the coefficients at the given time(s).

        Parameters
        ----------
        t : float | Array[float, ()]
            Time at which to evaluate the coefficients.
        r_s : float | Array[float, ()]
            Scale length of the potential at the given time(s.
        **kwargs : Any
            Additional keyword arguments are ignored.

        Returns
        -------
        Snlm : Array[float, (nmax+1, lmax+1, lmax+1)]
            The value of the cosine expansion coefficient.
        Tnlm : Array[float, (nmax+1, lmax+1, lmax+1)]
            The value of the sine expansion coefficient.
        """
        xyz, m = self.snapshot(t)
        return compute_coeffs_discrete(xyz, m, nmax=self.nmax, lmax=self.lmax, r_s=r_s)
