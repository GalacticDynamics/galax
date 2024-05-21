"""Test :class:`~galax.coordinates._pspt`."""

import pytest

from .test_base_psp import AbstractPhaseSpacePosition_Test
from galax.coordinates import PhaseSpacePosition


class TestPhaseSpacePosition(AbstractPhaseSpacePosition_Test[PhaseSpacePosition]):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[PhaseSpacePosition]:
        """Return the class of a phase-space position."""
        return PhaseSpacePosition
