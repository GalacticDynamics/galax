"""Test `galax.coordinates.AbstractPhaseSpaceCoordinate`."""

from abc import ABCMeta
from typing import TypeVar

import pytest

import galax.coordinates as gc
from .test_base_single import AbstractBasicPhaseSpaceCoordinate_Test

WT = TypeVar("WT", bound=gc.PhaseSpaceCoordinate)


class Test_PhaseSpaceCoordinate(
    AbstractBasicPhaseSpaceCoordinate_Test[gc.PhaseSpaceCoordinate], metaclass=ABCMeta
):
    """Test :class:`~galax.coordinates.PhaseSpaceCoordinate`."""

    @pytest.fixture(scope="class")
    def w_cls(self) -> type[gc.PhaseSpaceCoordinate]:
        """Return the class of a phase-space position."""
        return gc.PhaseSpaceCoordinate
