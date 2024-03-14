"""Test :class:`~galax.coordinates._pspt`."""

import pytest

from .test_base import AbstractPhaseSpaceTimePosition_Test
from galax.coordinates import PhaseSpaceTimePosition


class TestPhaseSpaceTimePosition(
    AbstractPhaseSpaceTimePosition_Test[PhaseSpaceTimePosition]
):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[PhaseSpaceTimePosition]:
        """Return the class of a phase-space position."""
        return PhaseSpaceTimePosition
