"""Test `galax.coordinates.AbstractPhaseSpaceCoordinate`."""

from abc import ABCMeta

import pytest

import galax.coordinates as gc
from .test_base_single import AbstractBasicPhaseSpaceCoordinate_Test


class Test_PhaseSpaceCoordinate(
    AbstractBasicPhaseSpaceCoordinate_Test[gc.PhaseSpaceCoordinate], metaclass=ABCMeta
):
    """Test :class:`~galax.coordinates.PhaseSpaceCoordinate`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[gc.PhaseSpaceCoordinate]:
        """Return the class of a phase-space position."""
        return gc.PhaseSpaceCoordinate
