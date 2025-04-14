"""Tests for the `galax.potential.` class."""

from typing import Any

import pytest

import unxt as u

import galax.potential as gp
from ..test_composite import AbstractCompositePotential_Test
from galax.potential._src.builtin.milkyway import AbstractPreCompositedPotential


class AbstractSpecialCompositePotential_Test(AbstractCompositePotential_Test):
    """Test the `galax.potential.AbstractCompositePotential` class."""

    def test_init_too_many_args(
        self,
        pot_cls: type[AbstractPreCompositedPotential],
        pot_map: dict[str, Any],
    ) -> None:
        """Test that the potential raises an error when given too many arguments."""
        with pytest.raises(ValueError, match="invalid keys"):
            _ = pot_cls(
                **pot_map,
                not_expected=gp.KeplerPotential(
                    m_tot=u.Quantity(1, "solMass"), units="galactic"
                ),
            )
