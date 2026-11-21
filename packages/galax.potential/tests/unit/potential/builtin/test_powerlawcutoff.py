import functools as ft

from typing import Any

import jax
import mpmath as mp
import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..io.test_gala import parametrize_test_method_gala
from ..optional_deps import GSL_ENABLED, OptDeps
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMTotMixin
from galax.potential._src.builtin.powerlawcutoff import potential


class AlphaParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["dimensionless"]:
        return u.Q(0.9, "")

    # =====================================================

    def test_alpha_constant(self, pot_cls, fields):
        """Test the `alpha` parameter."""
        fields["alpha"] = u.Q(1.0, "")
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Q(0, "Myr")) == u.Q(1.0, "")

    def test_alpha_userfunc(self, pot_cls, fields):
        """Test the `alpha` parameter."""

        def cos_alpha(t: u.Quantity["time"]) -> ParametricQuantity[""]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "")

        fields["alpha"] = cos_alpha
        pot = pot_cls(**fields)
        assert pot.alpha(t=u.Q(0, "Myr")) == u.Q(10, "")


class RCParameterMixin(ParameterFieldMixin):
    """Test the shape parameter."""

    @pytest.fixture(scope="class")
    def field_r_c(self) -> u.Quantity["length"]:
        return u.Q(1.0, "kpc")

    # =====================================================

    def test_r_c_constant(self, pot_cls, fields):
        """Test the `r_c` parameter."""
        fields["r_c"] = u.Q(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(1.0, "kpc")

    def test_r_c_userfunc(self, pot_cls, fields):
        """Test the `r_c` parameter."""

        def cos_r_c(t: u.Quantity["time"]) -> ParametricQuantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_c"] = cos_r_c
        pot = pot_cls(**fields)
        assert pot.r_c(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


class TestPowerLawCutoffPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    AlphaParameterMixin,
    RCParameterMixin,
):
    """Test the `galax.potential.PowerLawCutoffPotential` class."""

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.PowerLawCutoffPotential]:
        return gp.PowerLawCutoffPotential

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_alpha: u.Quantity,
        field_r_c: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "alpha": field_alpha,
            "r_c": field_r_c,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.PowerLawCutoffPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-1.20227523, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/powerlawcutoff", atol=1e-8
    )
    def test_gradient(self, pot: gp.PowerLawCutoffPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.PowerLawCutoffPotential, x: gt.QuSz3) -> None:
        expect = u.Q(41457.38551946, "solMass / kpc3")
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/powerlawcutoff", atol=1e-8
    )
    def test_hessian(self, pot: gp.PowerLawCutoffPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/powerlawcutoff", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Interoperability

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotential, x: gt.QuSz3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.PowerLawCutoffPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)


# ==============================================================================
# The potential kernel over the advertised range 0 <= alpha < 3, against mpmath.
#
# The reference is Phi = -G M(<r)/r - 4 pi G int_r^inf rho(s) s ds, with both
# integrals by `mpmath.quad` of the density, so nothing of the closed form
# (incomplete gamma functions) is shared with the code under test.

mp.mp.dps = 30

RC, M_TOT, G = 1.7, 3.0, 1.3  # generic values, so a stray factor shows up
ALPHAS = (0.0, 0.5, 1.0, 1.5, 1.9, 1.999, 2.0, 2.001, 2.5, 2.9)
NEAR_TWO = (2 - 1e-9, 2 + 1e-9)  # where the incomplete gamma terms cancel
RADII = (1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0)  # in r_c
EXTREME_RADII = (1e-8, 20.0)


def _params(alpha: float, dtype: Any = np.float64) -> dict[str, Any]:
    return {
        "alpha": dtype(alpha),
        "r_c": dtype(RC),
        "m_tot": dtype(M_TOT),
        "G": dtype(G),
    }


@ft.cache
def _reference(alpha: float, s: float) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """``(Phi, G M(<r)/r^2, Phi'')`` at ``r = s r_c``."""
    al, rc, m, g = (mp.mpf(v) for v in (alpha, RC, M_TOT, G))
    r = mp.mpf(s) * rc

    def rho(x: mp.mpf) -> mp.mpf:
        norm = m / (2 * mp.pi * mp.gamma(mp.mpf(1.5) - al / 2) * rc**3)
        return norm * (rc / x) ** al * mp.exp(-((x / rc) ** 2))

    # s' = r t^k removes the s'^(2 - alpha) endpoint singularity at 0.
    k = 1 / (3 - al)
    mass = mp.quad(
        lambda t: 4 * mp.pi * rho(r * t**k) * (r * t**k) ** 2 * r * k * t ** (k - 1),
        [0, 1],
    )
    edges = [r, *(rc * x for x in (1, 3, 10, 40) if rc * x > r)]
    if r >= rc:
        edges = [r, r + 5 * rc, r + 20 * rc]
    tail = mp.quad(lambda x: 4 * mp.pi * rho(x) * x, [*edges, mp.inf])
    return (
        -g * mass / r - g * tail,
        g * mass / r**2,
        g * (4 * mp.pi * rho(r) - 2 * mass / r**3),
    )


def _rel_err(value: Any, ref: mp.mpf) -> float:
    return float(abs(mp.mpf(float(value)) - ref) / abs(ref))


@pytest.mark.parametrize("alpha", [*ALPHAS, *NEAR_TWO])
@pytest.mark.parametrize("s", [*RADII, *EXTREME_RADII])
def test_kernel_potential_matches_mpmath(alpha: float, s: float) -> None:
    """The potential is finite and correct to 1e-10 for every 0 <= alpha < 3."""
    phi = jax.jit(potential)(_params(alpha), np.float64(s * RC))
    assert np.isfinite(phi)
    assert _rel_err(phi, _reference(alpha, s)[0]) < 1e-10


@pytest.mark.parametrize("alpha", [*ALPHAS, *NEAR_TWO])
@pytest.mark.parametrize("s", list(RADII))
def test_kernel_gradient_is_shell_theorem(alpha: float, s: float) -> None:
    """``dPhi/dr = G M(<r) / r^2``, and the Hessian is finite and correct."""
    p, r = _params(alpha), np.float64(s * RC)
    _, dphi, d2phi = _reference(alpha, s)

    grad = jax.jit(jax.grad(potential, argnums=1))(p, r)
    assert _rel_err(grad, dphi) < 1e-10

    hess = jax.jit(jax.hessian(potential, argnums=1))(p, r)
    # Phi'' = 4 pi G rho - 2 G M / r^3 cancels (alpha = 1 at small r), so the
    # error is held to the size of the terms.
    assert abs(float(hess) - float(d2phi)) < 1e-8 * float(G * M_TOT / (s * RC) ** 3)


@pytest.mark.parametrize("s", [1e-3, 0.5, 3.0])
def test_kernel_potential_is_continuous_across_alpha_2(s: float) -> None:
    """No jump between alpha = 2 - 1e-6, 2 and 2 + 1e-6."""
    r = np.float64(s * RC)
    lo, mid, hi = (float(potential(_params(a), r)) for a in (2 - 1e-6, 2.0, 2 + 1e-6))
    assert np.isfinite([lo, mid, hi]).all()
    # Phi changes by O(1e-6 |Phi| log(1/s)) over this step.
    assert abs(lo - mid) < 1e-4 * abs(mid)
    assert abs(hi - mid) < 1e-4 * abs(mid)


@pytest.mark.parametrize("alpha", [0.5, 1.9, 2.0, 2.001, 2.9])
@pytest.mark.parametrize("s", [1e-3, 0.1, 1.0, 3.0, 10.0])
def test_kernel_float32(alpha: float, s: float) -> None:
    """float32 value, gradient and Hessian are finite and good to 1e-4."""
    with jax.enable_x64(False):  # noqa: FBT003
        p, r = _params(alpha, np.float32), np.float32(s * RC)
        phi = jax.jit(potential)(p, r)
        grad = jax.jit(jax.grad(potential, argnums=1))(p, r)
        hess = jax.jit(jax.hessian(potential, argnums=1))(p, r)
    assert phi.dtype == np.float32
    assert np.isfinite([phi, grad, hess]).all()
    ref, dphi, d2phi = _reference(alpha, s)
    assert _rel_err(phi, ref) < 1e-4
    assert _rel_err(grad, dphi) < 1e-4
    assert abs(float(hess) - float(d2phi)) < 1e-4 * float(G * M_TOT / (s * RC) ** 3)


def test_kernel_far_field_is_point_mass() -> None:
    """At 20 r_c the cutoff has removed everything: Phi = -G M_tot / r."""
    r = 20 * RC
    for alpha in ALPHAS:
        phi = float(potential(_params(alpha), np.float64(r)))
        assert phi == pytest.approx(-G * M_TOT / r, rel=1e-12)


def test_kernel_vmap_over_alpha_and_radius() -> None:
    """One call over a grid of (alpha, r), through 2, has no NaN."""
    alphas = np.linspace(0.0, 2.95, 60)
    r = np.geomspace(1e-8, 30, 50) * RC
    p = _params(0.0)
    grid = jax.vmap(lambda al: jax.vmap(lambda x: potential({**p, "alpha": al}, x))(r))(
        alphas
    )
    assert np.isfinite(np.asarray(grid)).all()
    assert (np.asarray(grid) < 0).all()
