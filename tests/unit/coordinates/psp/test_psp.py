"""Test :class:`~galax.coordinates._src.psps.core`."""

import pytest

import galax.coordinates as gc
from .test_base_psp import AbstractOnePhaseSpacePosition_Test


class TestPhaseSpacePosition(AbstractOnePhaseSpacePosition_Test[gc.PhaseSpacePosition]):
    """Test :class:`~galax.coordinates.PhaseSpacePosition`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[gc.PhaseSpacePosition]:
        """Return the class of a phase-space position."""
        return gc.PhaseSpacePosition
