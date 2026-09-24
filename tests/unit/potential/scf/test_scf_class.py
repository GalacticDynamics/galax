"""Test `SCFPotential` against the standard potential test suite."""

from jaxtyping import Array, Shaped
from typing import Any, override

import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..builtin.test_common import ParameterMTotMixin, ParameterRSMixin
from ..test_core import AbstractSinglePotential_Test
from galax.interop.optional_deps import GSL_ENABLED, OptDeps


class TestSCFPotential(
    AbstractSinglePotential_Test,
    ParameterMTotMixin,
    ParameterRSMixin,
):
    HAS_GALA_COUNTERPART = True

    @pytest.fixture(scope="class")
    @override
    def pot_cls(self) -> type[gp.SCFPotential]:
        return gp.SCFPotential

    @pytest.fixture(scope="class")
    def field_Snlm(self) -> Shaped[Array, "2 2 2"]:
        return jnp.zeros((2, 2, 2)).at[0, 0, 0].set(1.0).at[0, 1, 1].set(0.1)

    @pytest.fixture(scope="class")
    def field_Tnlm(self) -> Shaped[Array, "2 2 2"]:
        return jnp.zeros((2, 2, 2)).at[0, 1, 1].set(0.05)

    @pytest.fixture(scope="class")
    @override
    def fields_(
        self,
        field_m_tot: u.Quantity,
        field_r_s: u.Quantity,
        field_Snlm: Shaped[Array, "2 2 2"],
        field_Tnlm: Shaped[Array, "2 2 2"],
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "m_tot": field_m_tot,
            "r_s": field_r_s,
            "Snlm": field_Snlm,
            "Tnlm": field_Tnlm,
            "units": field_units,
        }

    # ==========================================================================

    def test_potential(self, pot: gp.SCFPotential, x: gt.QuSz3) -> None:
        expect = u.Q(-0.93838334, unit="kpc2 / Myr2")
        assert jnp.isclose(pot.potential(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_gradient(self, pot: gp.SCFPotential, x: gt.QuSz3) -> None:
        expect = u.Q([0.05689436, 0.10603672, 0.15517907], pot.units["acceleration"])
        got = pot.gradient(x, t=0)
        assert jnp.allclose(got, expect, atol=u.Q(1e-8, expect.unit))

    def test_density(self, pot: gp.SCFPotential, x: gt.QuSz3) -> None:
        exp = u.Q(3.72911825e08, unit="solMass / kpc3")
        got = pot.density(x, t=0)
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    def test_hessian(self, pot: gp.SCFPotential, x: gt.QuSz3) -> None:
        expect = u.Q(
            [
                [0.04064982, -0.02084225, -0.03060797],
                [-0.02084225, 0.01266348, -0.05728349],
                [-0.03060797, -0.05728349, -0.03223266],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(pot.hessian(x, t=0), expect, atol=u.Q(1e-8, expect.unit))

    def test_hessian_matches_finite_difference_of_gradient(
        self, pot: gp.SCFPotential, x: gt.QuSz3
    ) -> None:
        """Cross-check the hessian against a finite difference of the gradient.

        Gala cannot serve as the oracle for the hessian: its SCF backend has
        no analytic hessian in the C code, so `gala`'s ``hessian()`` always
        returns zero (that is why the "hessian" case is commented out of
        ``test_method_gala`` below rather than compared). Gala *does*
        independently verify galax's `gradient` (see ``test_method_gala``),
        so this test numerically differentiates that gala-verified gradient
        and checks it against the analytic hessian instead. Do not delete
        this as "redundant" with `test_hessian`: that test only pins the
        hessian to a hardcoded value (a regression guard against changes),
        while this is the only check that the hessian is independently
        correct -- e.g. it would catch a future hand-written `_hessian`
        override that drifted from the autodiff-derived gradient, which a
        uniform rescaling of `_potential` would not (both derivatives scale
        together). Measured agreement at ``h=1e-5`` is ~4e-12.
        """
        h = 1e-5
        x0, unit = x.value, x.unit

        def grad(xv: gt.Sz3) -> gt.Sz3:
            return pot.gradient(u.Q(xv, unit), t=0).value

        fd = jnp.stack(
            [
                (grad(x0.at[i].add(h)) - grad(x0.at[i].add(-h))) / (2 * h)
                for i in range(3)
            ],
            axis=0,
        )
        got = pot.hessian(x, t=0).value
        assert jnp.allclose(got, fd, atol=1e-9)

    # ---------------------------------
    # Convenience methods

    def test_tidal_tensor(self, pot: gp.SCFPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        expect = u.Q(
            [
                [0.03362294, -0.02084225, -0.03060797],
                [-0.02084225, 0.0056366, -0.05728349],
                [-0.03060797, -0.05728349, -0.03925954],
            ],
            "1/Myr2",
        )
        assert jnp.allclose(
            pot.tidal_tensor(x, t=0), expect, atol=u.Q(1e-8, expect.unit)
        )

    # ==========================================================================
    # Interoperability

    @pytest.mark.skipif(
        not OptDeps.GALA.installed or not GSL_ENABLED, reason="requires gala + GSL"
    )
    @pytest.mark.parametrize(
        ("method0", "method1", "atol"),
        [
            ("potential", "energy", 1e-8),
            ("gradient", "gradient", 1e-8),
            ("density", "density", 1e-8),
            # ("hessian", "hessian", 1e-8),  # gala's SCFPotential has no
            # analytic hessian in its C backend and always returns 0.
        ],
    )
    def test_method_gala(
        self,
        pot: gp.AbstractPotential,
        method0: str,
        method1: str,
        x: gt.QuSz3,
        atol: float,
    ) -> None:
        """Test the equivalence of methods between gala and galax.

        This test only runs if the potential can be mapped to gala.
        """
        super().test_method_gala(pot, method0, method1, x, atol)
