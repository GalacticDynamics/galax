"""Test the `MultipolePotential` class."""

import re
from typing import Any
from typing_extensions import override

import pytest
from jaxtyping import Array, Shaped
from plum import convert

import quaxed.numpy as jnp
import unxt as u

import galax._custom_types as gt
import galax.potential as gp
from ...io.test_gala import parametrize_test_method_gala
from ...test_core import AbstractSinglePotential_Test
from ..test_common import ParameterMTotMixin, ParameterScaleRadiusMixin
from .test_abstractmultipole import ParameterAngularCoefficientsMixin

###############################################################################


class ParameterISlmMixin(ParameterAngularCoefficientsMixin):
    """Test the ISlm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_ISlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """ISlm parameter."""
        ISlm = jnp.zeros((field_l_max + 1, field_l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)
        return ISlm

    # =====================================================

    def test_ISlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, :].set(5.0)

        fields["ISlm"] = u.Quantity(ISlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.ISlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ISlm.value, u.Quantity(ISlm, ""))

    def test_ISlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = ISlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=u.Quantity(0, "Myr")), ISlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ISlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = lambda t: ISlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=u.Quantity(0, "Myr")), ISlm)


class ParameterITlmMixin(ParameterAngularCoefficientsMixin):
    """Test the ITlm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

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

        fields["ITlm"] = u.Quantity(ITlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.ITlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ITlm.value, u.Quantity(ITlm, ""))

    def test_ITlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, 0].set(5.0)

        fields["ITlm"] = ITlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=u.Quantity(0, "Myr")), ITlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ITlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, :].set(5.0)

        fields["ITlm"] = lambda t: ITlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=u.Quantity(0, "Myr")), ITlm)


class ParameterOSlmMixin(ParameterAngularCoefficientsMixin):
    """Test the OSlm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

    @pytest.fixture(scope="class")
    def field_OSlm(self, field_l_max) -> Shaped[Array, "3 3"]:
        """OSlm parameter."""
        OSlm = jnp.zeros((field_l_max + 1, field_l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)
        return OSlm

    # =====================================================

    def test_OSlm_units(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, :].set(5.0)

        fields["OSlm"] = u.Quantity(OSlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.OSlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OSlm.value, u.Quantity(OSlm, ""))

    def test_OSlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = OSlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=u.Quantity(0, "Myr")), OSlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OSlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = lambda t: OSlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=u.Quantity(0, "Myr")), OSlm)


class ParameterOTlmMixin(ParameterAngularCoefficientsMixin):
    """Test the OTlm parameter."""

    pot_cls: type[gp.AbstractSinglePotential]

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

        fields["OTlm"] = u.Quantity(OTlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.OTlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OTlm.value, u.Quantity(OTlm, ""))

    def test_OTlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, 0].set(5.0)

        fields["OTlm"] = OTlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=u.Quantity(0, "Myr")), OTlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OTlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, :].set(5.0)

        fields["OTlm"] = lambda t: OTlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=u.Quantity(0, "Myr")), OTlm)


###############################################################################


class TestMultipolePotential(
    AbstractSinglePotential_Test,
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
        field_units: u.AbstractUnitSystem,
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

    def test_check_init(
        self, pot_cls: type[gp.MultipoleInnerPotential], fields_: dict[str, Any]
    ) -> None:
        """Test the `MultipoleInnerPotential.__check_init__` method."""
        fields_["ISlm"] = fields_["ISlm"][::2]  # make it the wrong shape
        match = re.escape("I/OSlm and I/OTlm must have the shape")
        with pytest.raises(ValueError, match=match):
            pot_cls(**fields_)

    # ==========================================================================

    def test_potential(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(33.59908611, unit="kpc2 / Myr2")
        assert jnp.isclose(
            pot.potential(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_gradient(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [-0.13487022, -0.26974043, 10.79508472], pot.units["acceleration"]
        )
        got = convert(pot.gradient(x, t=0), u.Quantity)
        assert jnp.allclose(got, expect, atol=u.Quantity(1e-8, expect.unit))

    def test_density(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(4.73805126e-05, pot.units["mass density"])
        assert jnp.isclose(
            pot.density(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    def test_hessian(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        expect = u.Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.hessian(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Quantity(
            [
                [-0.08670228, 0.09633587, 0.09954706],
                [0.09633587, 0.05780152, 0.19909413],
                [0.09954706, 0.19909413, 0.02890076],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Quantity(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.xfail
    def test_galax_to_gala_to_galax_roundtrip(
        self, pot: gp.AbstractPotential, x: gt.QuSz3
    ) -> None:
        super().test_galax_to_gala_to_galax_roundtrip(pot, x)

    @pytest.mark.xfail
    @parametrize_test_method_gala
    def test_method_gala(
        self,
        pot: gp.MultipolePotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        super().test_method_gala(pot, method0, method1, x, atol)
