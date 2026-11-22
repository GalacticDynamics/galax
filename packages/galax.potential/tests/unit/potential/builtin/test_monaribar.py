from typing import Any, ClassVar

import mpmath as mp
import numpy as np
import pytest

import quaxed.numpy as jnp
import unxt as u

import galax.potential as gp
import galax.potential.custom_types as gt
from ..test_core import AbstractSinglePotential_Test
from galax.potential import monarietal2016bar as monari


def _mp_potential(p: dict[str, float], t: float, x: Any, y: Any, z: Any) -> Any:
    """Monari et al. (2016) eq. 1-2 with Dehnen (2000) eq. 3 for ``U``."""
    a, v0, r0, rb, pb, om = (
        mp.mpf(p[k]) for k in ("alpha", "v0", "R0", "Rb", "phi_b", "Omega")
    )
    r = mp.sqrt(x**2 + y**2 + z**2)
    s = r / rb
    u_ = s**3 - 2 if s < 1 else -(s**-3)
    gamma = 2 * (mp.atan2(y, x) - pb - om * mp.mpf(t))
    return a * v0**2 / 3 * (r0 / rb) ** 3 * u_ * (x**2 + y**2) / r**2 * mp.cos(gamma)


@pytest.mark.parametrize(
    "xyz",
    [
        (0.3, 0.5, 0.2),
        (-0.9, 0.4, -0.7),
        (1.0, -0.6, 0.5),  # r = 1.3 < R_b
        (-0.2, -1.5, 0.9),  # r = 1.76 > R_b, where the density is 0
        (2.0, 1.5, 0.8),
    ],
)
def test_density_and_laplacian_match_mpmath(xyz: tuple[float, float, float]) -> None:
    """``4 pi G rho = laplacian(Phi)`` with Phi transcribed independently.

    The reference is ``mpmath.diff`` of the potential at 40 digits, so it does
    not use the radial ``L(r)`` of the code. Measured error < 1e-13.
    """
    p = {
        "G": 1.3,
        "alpha": 0.3,
        "v0": 1.1,
        "R0": 2.0,
        "Rb": 1.7,
        "phi_b": 0.4,
        "Omega": 0.6,
    }
    t = 0.7
    with mp.workdps(40):
        pt = tuple(map(mp.mpf, xyz))
        lap = sum(
            mp.diff(lambda x, y, z: _mp_potential(p, t, x, y, z), pt, order)
            for order in ((2, 0, 0), (0, 2, 0), (0, 0, 2))
        )
        rho = float(lap / (4 * mp.pi * p["G"]))
    q = np.asarray(xyz, dtype=np.float64)
    np.testing.assert_allclose(monari.density(p, q, t), rho, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(
        monari.laplacian(p, q, t), float(lap), rtol=1e-11, atol=1e-13
    )


class TestMonariEtAl2016BarPotential(
    AbstractSinglePotential_Test,
):
    """Test the `galax.potential.MonariEtAl2016BarPotential` class."""

    HAS_GALA_COUNTERPART: ClassVar[bool] = False

    @pytest.fixture(scope="class")
    def pot_cls(self) -> type[gp.MonariEtAl2016BarPotential]:
        return gp.MonariEtAl2016BarPotential

    @pytest.fixture(scope="class")
    def field_alpha(self) -> u.Quantity["dimensionless"]:
        return u.Q(0.01, "")

    @pytest.fixture(scope="class")
    def field_R0(self) -> u.Quantity["length"]:
        return u.Q(8.0, "kpc")

    @pytest.fixture(scope="class")
    def field_v0(self) -> u.Quantity["speed"]:
        return u.Q(220.0, "km/s")

    @pytest.fixture(scope="class")
    def field_Rb(self) -> u.Quantity["length"]:
        return u.Q(3.5, "kpc")

    @pytest.fixture(scope="class")
    def field_phi_b(self) -> u.Quantity["angle"]:
        return u.Q(25, "deg")

    @pytest.fixture(scope="class")
    def field_Omega(self) -> u.Quantity["frequency"]:
        return u.Q(52.2, "km/(s kpc)")

    @pytest.fixture(scope="class")
    def fields_(
        self,
        field_alpha: u.AbstractQuantity,
        field_R0: u.AbstractQuantity,
        field_v0: u.AbstractQuantity,
        field_Rb: u.AbstractQuantity,
        field_phi_b: u.AbstractQuantity,
        field_Omega: u.AbstractQuantity,
        field_units: u.AbstractUnitSystem,
    ) -> dict[str, Any]:
        return {
            "alpha": field_alpha,
            "R0": field_R0,
            "v0": field_v0,
            "Rb": field_Rb,
            "phi_b": field_phi_b,
            "Omega": field_Omega,
            "units": field_units,
        }

    # ==========================================================================

    def test_defaults(
        self,
        pot_cls: type[gp.MonariEtAl2016BarPotential],
        field_R0: u.AbstractQuantity,
        field_v0: u.AbstractQuantity,
        field_units: u.AbstractUnitSystem,
    ) -> None:
        """The defaults are weakly typed, in their own unit."""
        pot = pot_cls(R0=field_R0, v0=field_v0, units=field_units)
        for name, want in [
            ("alpha", u.Q(0.01, "")),
            ("Omega", u.Q(52.2, "km/(s kpc)")),
            ("Rb", u.Q(3.5, "kpc")),
            ("phi_b", u.Q(25, "deg")),
        ]:
            got = getattr(pot, name).value
            assert got.unit == want.unit, name
            assert got.value.weak_type, name
            assert got.value == want.value, name

    def test_potential(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        got = pot.potential(x, t=0)
        exp = u.Q(-0.00013381, pot.units["specific energy"])
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/monaribar", atol=1e-8
    )
    def test_gradient(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        return pot.gradient(x, t=0).ustrip(pot.units["acceleration"])

    def test_density(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        got = pot.density(x, t=0)
        exp = u.Q(4.79482196e-10, "Msun / kpc3")
        assert jnp.isclose(got, exp, atol=u.Q(1e-8, exp.unit))

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/monaribar", atol=1e-8
    )
    def test_hessian(self, pot: gp.MonariEtAl2016BarPotential, x: gt.QuSz3) -> None:
        return pot.hessian(x, t=0).ustrip("1/Myr2")

    # ---------------------------------
    # Convenience methods

    @pytest.mark.array_compare(
        file_format="text", reference_dir="reference/monaribar", atol=1e-8
    )
    def test_tidal_tensor(self, pot: gp.AbstractPotential, x: gt.QuSz3) -> None:
        """Test the `AbstractPotential.tidal_tensor` method."""
        return pot.tidal_tensor(x, t=0).ustrip("1/Myr2")
