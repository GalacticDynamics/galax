from typing import Any, ClassVar, override

import mpmath as mp
import numpy as np
import pytest
from unxts.parametric import ParametricQuantity

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
import galax.potential.params as gpp
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin
from galax.potential import hardcutoffnfw


class ParameterRTMixin(ParameterFieldMixin):
    """Test the mass parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_r_t(self) -> u.Quantity["length"]:
        return u.Q(67.0, "kpc")

    # =====================================================

    def test_r_t_units(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ) -> None:
        """Test the mass parameter."""
        fields["r_t"] = 1.0 * u.unit(10 * u.unit("kpc"))
        fields["units"] = u.unitsystems.galactic
        pot = pot_cls(**fields)
        assert isinstance(pot.r_t, gpp.ConstantParameter)
        assert jnp.isclose(pot.r_t(0), u.Q(10, "kpc"), atol=u.Q(1e-8, "kpc"))

    def test_r_t_constant(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the mass parameter."""
        fields["r_t"] = u.Q(1.0, "kpc")
        pot = pot_cls(**fields)
        assert pot.r_t(t=u.Q(0, "Myr")) == u.Q(1.0, "kpc")

    def test_r_t_userfunc(
        self, pot_cls: type[gp.AbstractSinglePotential], fields: dict[str, Any]
    ):
        """Test the scale radius parameter."""

        def cos_scalelength(t: u.Quantity["time"]) -> ParametricQuantity["length"]:
            return u.Q(10 * jnp.cos(t.ustrip("Myr")), "kpc")

        fields["r_t"] = cos_scalelength
        pot = pot_cls(**fields)
        assert pot.r_t(t=u.Q(0, "Myr")) == u.Q(10, "kpc")


###############################################################################


@pytest.mark.parametrize("r_t", [20.0, 3.0], ids=["r_t>r_s", "r_t<r_s"])
def test_potential_matches_mpmath(r_t: float) -> None:
    """``potential`` and ``density`` against mpmath quadrature of the density.

    ``Phi(r) = -G [M(<r) / r + 4 pi int_r^r_t rho s ds]`` with
    ``rho = m / (4 pi r_s^3 x (1 + x)^2)`` inside r_t and 0 outside, so
    ``Phi(inf) = 0`` and the outer shell sets the inner offset ``G m / (r_s +
    r_t)``. Measured error <= 4.4e-16, a margin of 200 at rtol 1e-13.
    """
    pot = gp.HardCutoffNFWPotential(
        m=u.Q(1e12, "Msun"), r_s=u.Q(5.0, "kpc"), r_t=u.Q(r_t, "kpc"), units="galactic"
    )
    p = pot.params(0.0)
    mp.mp.dps = 30
    g, m, r_s, rt = (mp.mpf(float(p[k])) for k in ("G", "m", "r_s", "r_t"))

    def rho(s: mp.mpf) -> mp.mpf:
        x = s / r_s
        return m / (4 * mp.pi * r_s**3 * x * (1 + x) ** 2)

    for f in (0.05, 0.3, 0.7, 0.999, 1.001, 1.5, 4.0):
        rr = f * rt
        mass = mp.quad(lambda s: 4 * mp.pi * s**2 * rho(s), [0, min(rr, rt)])
        shell = mp.quad(lambda s: 4 * mp.pi * s * rho(s), [rr, rt]) if rr < rt else 0
        want = float(-g * (mass / rr + shell))
        r = float(rr)
        np.testing.assert_allclose(
            hardcutoffnfw.potential(p, r), want, rtol=1e-13, err_msg=f"{f}"
        )
        got = pot.potential(jnp.array([r, 0.0, 0.0]), 0.0)
        np.testing.assert_allclose(got, want, rtol=1e-13, err_msg=f"{f}")

    # The cut is at r_t exactly: zero density 5e-4 r_t above it, NFW just below.
    r_out, r_in = float(rt) * (1 + 5e-4), float(rt) * (1 - 5e-4)
    assert hardcutoffnfw.density(p, r_out) == 0.0
    np.testing.assert_allclose(
        hardcutoffnfw.density(p, r_in), float(rho(mp.mpf(r_in))), rtol=1e-13
    )


class TestHardCutoffNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    ParameterRTMixin,
):
    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.HardCutoffNFWPotential]:
        return gp.HardCutoffNFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_r_t: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {"m": field_m, "r_s": field_r_s, "r_t": field_r_t, "units": field_units}

    # ==========================================================================

    def test_potential(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        exp = u.Q(-1.80505084, pot.units["specific energy"])
        got = pot.potential(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/cutoffnfw", atol=1e-8
    )
    def test_gradient(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(9.45944763e08, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/cutoffnfw", atol=1e-8
    )
    def test_hessian(self, pot: gp.HardCutoffNFWPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/cutoffnfw", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
