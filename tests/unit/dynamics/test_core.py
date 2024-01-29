"""Test :mod:`~galax.dynamics.core`."""

import numpy as np

from galax.dynamics import PhaseSpacePosition


class TestPhaseSpacePosition:
    """Test :class:`~galax.dynamics.PhaseSpacePosition`."""

    def test_len(self) -> None:
        """Test length."""
        # Simple
        q = np.random.random(size=(10, 3))
        p = np.random.random(size=(10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 10

        # Complex shape
        q = np.random.random(size=(4, 10, 3))
        p = np.random.random(size=(4, 10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 4
