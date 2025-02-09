"""Test `galax.coordinates.AbstractPhaseSpaceCoordinate`."""

from abc import ABCMeta
from typing import TypeVar

import galax.coordinates as gc
from .test_base import AbstractPhaseSpaceCoordinate_Test

WT = TypeVar("WT", bound=gc.AbstractBasicPhaseSpaceCoordinate)


class AbstractBasicPhaseSpaceCoordinate_Test(
    AbstractPhaseSpaceCoordinate_Test[WT], metaclass=ABCMeta
):
    """Test :class:`~galax.coordinates.AbstractBasicPhaseSpaceCoordinate`."""
