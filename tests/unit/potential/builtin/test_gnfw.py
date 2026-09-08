from typing import Any, override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..param.test_field import ParameterFieldMixin
from ..test_core import AbstractSinglePotential_Test
from .test_common import ParameterMMixin, ParameterRSMixin

###############################################################################


class GammaParameterMixin(ParameterFieldMixin):
    """Test the inner-slope parameter."""

    @pytest.fixture(scope="class")
    def field_gamma(self) -> u.Quantity["dimensionless"]:
        return u.Q(1.0, "")

    # =====================================================

    def test_gamma_constant(self, pot_cls, fields):
        fields["gamma"] = u.Q(0.5, "")
        pot = pot_cls(**fields)
        assert pot.gamma(t=u.Q(0, "Myr")) == u.Q(0.5, "")

    def test_gamma_userfunc(self, pot_cls, fields):
        def cos_gamma(t: u.Quantity["time"]) -> u.Quantity[""]:
            return u.Q(0.5 * jnp.cos(t.ustrip("Myr")), "")

        fields["gamma"] = cos_gamma
        pot = pot_cls(**fields)
        assert pot.gamma(t=u.Q(0, "Myr")) == u.Q(0.5, "")


class TestGNFWPotential(
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMMixin,
    ParameterRSMixin,
    GammaParameterMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.gNFWPotential]:
        return gp.gNFWPotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m: u.Quantity,
        field_r_s: u.Quantity,
        field_gamma: u.Quantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m": field_m,
            "r_s": field_r_s,
            "gamma": field_gamma,
            "units": field_units,
        }

    # ==========================================================================
    # gamma=1 reduces to the NFW potential (see class docstring), so these
    # expected values match `tests/unit/potential/builtin/test_nfw.py`.

    def test_potential(self, pot: gp.gNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-1.87120528, pot.units["specific energy"])
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_gradient(self, pot: gp.gNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q([0.06589185, 0.1317837, 0.19767556], pot.units["acceleration"])
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.gNFWPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(9.45944763e08, pot.units["mass density"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    def test_hessian(self, pot: gp.gNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.05559175, -0.02060021, -0.03090031],
                [-0.02060021, 0.02469144, -0.06180062],
                [-0.03090031, -0.06180062, -0.02680908],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(pot.hessian(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_tidal_tensor(self, pot: gp.gNFWPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.03776704, -0.02060021, -0.03090031],
                [-0.02060021, 0.00686674, -0.06180062],
                [-0.03090031, -0.06180062, -0.04463378],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )

    # ---------------------------------
    # Regression test: GalacticDynamics/galax#817, fixed by #818.
    # `Bz_from_hyp2f1` used to blow up to nan at large r/r_s because the
    # underlying `_2F_1` implementation was only valid for small arguments.

    @pytest.mark.parametrize("r", [1.0, 8.0, 50.0, 100.0, 1000.0])
    def test_large_radius_is_finite(
        self, pot_cls: type[gp.gNFWPotential], fields: dict[str, Any], r: float
    ) -> None:
        fields["gamma"] = u.Q(0.5, "")
        pot = pot_cls(**fields)
        x = u.Q(jnp.asarray([r, 0.0, 0.0]), "kpc")

        assert jnp.isfinite(pot.potential(x, t=0))
        assert jnp.all(jnp.isfinite(pot.gradient(x, t=0).ustrip(pot.units)))
        assert jnp.isfinite(pot.density(x, t=0))
        assert jnp.all(jnp.isfinite(pot.hessian(x, t=0)))
