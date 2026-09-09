"""Test the `MultipolePotential` class."""

import re

from jaxtyping import Array, Shaped
from typing import Any, override

import equinox as eqx
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..io.test_gala import parametrize_test_method_gala
from ..test_core import AbstractSinglePotential_Test
from .test_abstractmultipole import (
    MultipoleTestMixin,
    ParameterAngularCoefficientsMixin,
)
from .test_common import ParameterMTotMixin, ParameterRSMixin

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

        fields["ISlm"] = u.Q(ISlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.ISlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ISlm.value, u.Q(ISlm, ""))

    def test_ISlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = ISlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=u.Q(0, "Myr")), ISlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ISlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ISlm = jnp.zeros((l_max + 1, l_max + 1))
        ISlm = ISlm.at[1, 0].set(5.0)

        fields["ISlm"] = lambda t: ISlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ISlm(t=u.Q(0, "Myr")), ISlm)


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

        fields["ITlm"] = u.Q(ITlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.ITlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.ITlm.value, u.Q(ITlm, ""))

    def test_ITlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, 0].set(5.0)

        fields["ITlm"] = ITlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=u.Q(0, "Myr")), ITlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_ITlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        ITlm = jnp.zeros((l_max + 1, l_max + 1))
        ITlm = ITlm.at[1, :].set(5.0)

        fields["ITlm"] = lambda t: ITlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.ITlm(t=u.Q(0, "Myr")), ITlm)


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

        fields["OSlm"] = u.Q(OSlm, "")
        pot = pot_cls(**fields)
        assert isinstance(pot.OSlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OSlm.value, u.Q(OSlm, ""))

    def test_OSlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = OSlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=u.Q(0, "Myr")), OSlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OSlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OSlm = jnp.zeros((l_max + 1, l_max + 1))
        OSlm = OSlm.at[1, 0].set(5.0)

        fields["OSlm"] = lambda t: OSlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OSlm(t=u.Q(0, "Myr")), OSlm)


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

        fields["OTlm"] = u.Q(OTlm, "")
        fields["l_max"] = l_max
        pot = pot_cls(**fields)
        assert isinstance(pot.OTlm, gp.params.ConstantParameter)
        assert jnp.allclose(pot.OTlm.value, u.Q(OTlm, ""))

    def test_OTlm_constant(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, 0].set(5.0)

        fields["OTlm"] = OTlm
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=u.Q(0, "Myr")), OTlm)

    @pytest.mark.xfail(reason="TODO: user function doesn't have units")
    def test_OTlm_userfunc(self, pot_cls, fields):
        """Test the mass parameter."""
        l_max = fields["l_max"]
        OTlm = jnp.zeros((l_max + 1, l_max + 1))
        OTlm = OTlm.at[1, :].set(5.0)

        fields["OTlm"] = lambda t: OTlm * jnp.exp(-jnp.abs(t))
        pot = pot_cls(**fields)
        assert jnp.allclose(pot.OTlm(t=u.Q(0, "Myr")), OTlm)


###############################################################################


class TestMultipolePotential(
    MultipoleTestMixin,
    AbstractSinglePotential_Test,
    # Parameters
    ParameterMTotMixin,
    ParameterRSMixin,
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
        with pytest.raises(eqx.EquinoxTracetimeError, match=match):
            pot_cls(**fields_)

    # ==========================================================================

    def test_potential(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        expect = u.Q(33.59908611, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/multipole", atol=1e-8
    )
    def test_gradient(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        # Exactly zero: both r^l Y_lm and r^{-(l+1)} Y_lm are solid harmonics
        # (source-free) for every l, m. See `AbstractMultipolePotential._density`.
        expect = u.Q(0, pot.units["mass density"])
        assert jnp.isclose(pot.density(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/multipole", atol=1e-8
    )
    def test_hessian(self, pot: gp.MultipolePotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/multipole", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")

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


###############################################################################
# Regression: batched evaluation must match per-position evaluation.
#
# `compute_Ylm` used to pass length-1 `l`/`m` arrays to `sph_harm_y` against a
# length-N `theta`, which silently returned wrong values at every batch index
# but 0 for any `l > 0` term. Every other Multipole fixture here evaluates a
# single position, so nothing caught it.


def _lm_coeffs(
    l_max: int,
) -> tuple[Shaped[Array, "{l_max}+1 {l_max}+1"], Shaped[Array, "{l_max}+1 {l_max}+1"]]:
    """Build ``Slm``/``Tlm`` with non-zero entries at ``m >= 1``.

    Requires ``l_max >= 2``: the coefficients set below reach ``(2, 2)``, which
    is what puts a non-zero ``m >= 1`` term into the expansion.
    """
    Slm = jnp.zeros((l_max + 1, l_max + 1))
    Slm = Slm.at[1, 0].set(0.4).at[1, 1].set(0.3).at[2, 2].set(0.15)
    Tlm = jnp.zeros((l_max + 1, l_max + 1))
    Tlm = Tlm.at[1, 1].set(0.25).at[2, 1].set(-0.2)
    return Slm, Tlm


_BATCH_XYZ = u.Q(
    [[1.3, -2.1, 0.7], [4.0, 3.0, -5.0], [-1.5, 2.5, 3.5], [0.2, 0.6, -0.1]], "kpc"
)


@pytest.mark.parametrize(
    "pot",
    [
        gp.MultipoleInnerPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            Slm=_lm_coeffs(2)[0],
            Tlm=_lm_coeffs(2)[1],
            l_max=2,
            units="galactic",
        ),
        gp.MultipoleOuterPotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            Slm=_lm_coeffs(2)[0],
            Tlm=_lm_coeffs(2)[1],
            l_max=2,
            units="galactic",
        ),
        gp.MultipolePotential(
            m_tot=u.Q(1e12, "Msun"),
            r_s=u.Q(10.0, "kpc"),
            ISlm=_lm_coeffs(2)[0],
            ITlm=_lm_coeffs(2)[1],
            OSlm=_lm_coeffs(2)[0],
            OTlm=_lm_coeffs(2)[1],
            l_max=2,
            units="galactic",
        ),
    ],
    ids=["inner", "outer", "both"],
)
def test_batched_matches_per_position(pot: gp.AbstractPotential) -> None:
    """Evaluate a batch of positions and each position alone; require equality."""
    t = u.Q(0.0, "Gyr")
    batched = pot.potential(_BATCH_XYZ, t)
    one_at_a_time = jnp.stack([pot.potential(xyz, t) for xyz in _BATCH_XYZ])

    # The two paths agree exactly (max relative difference 0.0) in float64, but
    # requiring bit-identity would over-specify the contract: XLA may fuse the
    # batched and scalar paths differently, and more so on an accelerator. The
    # regression this guards against is ~1e-1 relative, so 1e-8 keeps seven
    # orders of detection margin while staying robust. Do not tighten.
    assert jnp.allclose(
        batched, one_at_a_time, rtol=1e-8, atol=u.Q(1e-10, batched.unit)
    )
