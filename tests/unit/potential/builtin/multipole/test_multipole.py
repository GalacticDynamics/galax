"""Test the `MultipolePotential` class."""

from typing import Any

import astropy.units as u
import pytest
from jaxtyping import Array, Shaped
from plum import convert
from typing_extensions import override

import quaxed.numpy as jnp
from unxt import Quantity
from unxt.unitsystems import AbstractUnitSystem

import galax.potential as gp
import galax.typing as gt
from ...io.test_gala import parametrize_test_method_gala
from ...test_core import AbstractPotential_Test
from ..test_common import ParameterMTotMixin, ParameterScaleRadiusMixin
from .test_abstractmultipole import ParameterAngularCoefficientsMixin

###############################################################################


class ParameterISlmMixin(ParameterAngularCoefficientsMixin):
    """Test the ISlm parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_ISlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """ISlm parameter."""
        ISlm = jnp.zeros((field_l_max + 1, field_l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)
        return ISlm  # noqa: RET504

    # =====================================================

    def test_ISlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, :].set(5.0)

        fields["ISlm"] = Quantity(ISlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.ISlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ISlm.value, Quantity(ISlm, ""))

    def test_ISlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = ISlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=Quantity(0, "Myr")), ISlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ISlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = lambda t: ISlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=Quantity(0, "Myr")), ISlm)


class ParameterITlmMixin(ParameterAngularCoefficientsMixin):
    """Test the ITlm parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_ITlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """ITlm parameter."""
        return jnp.zeros((field_l_max + 1, field_l_max + 1))

    # =====================================================

    def test_ITlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, :].set(5.0)

        fields["ITlm"] = Quantity(ITlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.ITlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ITlm.value, Quantity(ITlm, ""))

    def test_ITlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, 0].set(5.0)

        fields["ITlm"] = ITlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=Quantity(0, "Myr")), ITlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ITlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, :].set(5.0)

        fields["ITlm"] = lambda t: ITlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=Quantity(0, "Myr")), ITlm)


class ParameterOSlmMixin(ParameterAngularCoefficientsMixin):
    """Test the OSlm parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_OSlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """OSlm parameter."""
        OSlm = jnp.zeros((field_l_max + 1, field_l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)
        return OSlm  # noqa: RET504

    # =====================================================

    def test_OSlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, :].set(5.0)

        fields["OSlm"] = Quantity(OSlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.OSlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OSlm.value, Quantity(OSlm, ""))

    def test_OSlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = OSlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=Quantity(0, "Myr")), OSlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OSlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = lambda t: OSlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=Quantity(0, "Myr")), OSlm)


class ParameterOTlmMixin(ParameterAngularCoefficientsMixin):
    """Test the OTlm parameter."""

    pot_cls: type[gp.AbstractPotential]

    @pytest.fixture(scope="class")
    def field_OTlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """OTlm parameter."""
        return jnp.zeros((field_l_max + 1, field_l_max + 1))

    # =====================================================

    def test_OTlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, :].set(5.0)

        fields["OTlm"] = Quantity(OTlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.OTlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OTlm.value, Quantity(OTlm, ""))

    def test_OTlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, 0].set(5.0)

        fields["OTlm"] = OTlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=Quantity(0, "Myr")), OTlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OTlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, :].set(5.0)

        fields["OTlm"] = lambda t: OTlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=Quantity(0, "Myr")), OTlm)


###############################################################################


class TestMultipolePotential(
    AbstractPotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterScaleRadiusMixin,
    ParameterISlmMixin,
    ParameterITlmMixin,
    ParameterOSlmMixin,
    ParameterOTlmMixin,
):
    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.MultipolePotential]:
        return gp.MultipolePotential

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_l_max: int,
        field_ISlm: Shaped[Array, "3 3"],
        field_ITlm: Shaped[Array, "3 3"],
        field_OSlm: Shaped[Array, "3 3"],
        field_OTlm: Shaped[Array, "3 3"],
        field_units: AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "l_max": field_l_max,
            "ISlm": field_ISlm,
            "ITlm": field_ITlm,
            "OSlm": field_OSlm,
            "OTlm": field_OTlm,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.MultipolePotential, x: gt.QVec3) -> None:
        expect = Quantity(33.59908611, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.MultipolePotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [-0.13487022, -0.26974043, 10.79508472], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), Quantity)
        assert jnp.allclose(got, expect, atol=Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.MultipolePotential, x: gt.QVec3) -> None:
        expect = Quantity(4.73805126e-05, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.MultipolePotential, x: gt.QVec3) -> None:
        expect = Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotentialBase, x: gt.QVec3) -> None:
        """Test the `AbstractPotentialBase.tidal_tensor` method."""
        expect = Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.xfail()
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotentialBase, x: gt.QVec3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.xfail()
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.MultipolePotential,
        method0: str,
        method1: str,
        x: gt.QVec3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
