"""Test :mod:`~galax.dynamics.core`."""

import jax.experimental.array_api as xp
import jax.numpy as jnp
import pytest
from jax import random

from galax.dynamics import PhaseSpacePosition
from galax.units import galactic


class TestPhaseSpacePosition:
    """Test :class:`~galax.dynamics.PhaseSpacePosition`."""

    def test_len(self) -> None:
        """Test length."""
        _, *subkeys = random.split(random.PRNGKey(0), num=5)

        # Simple
        q = random.uniform(subkeys[0], shape=(10, 3))
        p = random.uniform(subkeys[1], shape=(10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 10

        # Complex shape
        q = random.uniform(subkeys[2], shape=(4, 10, 3))
        p = random.uniform(subkeys[3], shape=(4, 10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 4

    # ------------------------------------------------------------------------

    def test_w(self) -> None:
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.w`."""
        _, *subkeys = random.split(random.PRNGKey(0), num=3)

        q = random.uniform(subkeys[0], shape=(10, 3))
        p = random.uniform(subkeys[1], shape=(10, 3))
        psp = PhaseSpacePosition(q, p)

        # units = None
        w = psp.w()
        assert jnp.all(w == xp.concat((q, p), axis=1))

        # units != None
        with pytest.raises(NotImplementedError):
            _ = psp.w(units=galactic)

    # ------------------------------------------------------------------------
    # `wt()`

    def test_wt_notime(self) -> None:
        """Test :attr:`~galax.dynamics.core.PhaseSpacePosition.wt`."""
        _, *subkeys = random.split(random.PRNGKey(0), num=3)

        q = random.uniform(subkeys[0], shape=(10, 3))
        p = random.uniform(subkeys[1], shape=(10, 3))
        psp = PhaseSpacePosition(q, p)

        # units = None
        wt = psp.wt()
        assert jnp.array_equal(wt[..., :-1], xp.concat((q, p), axis=-1))
        assert jnp.array_equal(wt[..., -1], xp.zeros(len(wt)))

        # units != None
        with pytest.raises(NotImplementedError):
            _ = psp.wt(units=galactic)

    def test_wt_time(self) -> None:
        """Test :attr:`~galax.dynamics.core.AbstractPhaseSpacePositionBase.wt`."""
        _, *subkeys = random.split(random.PRNGKey(0), num=4)

        q = random.uniform(subkeys[0], shape=(10, 3))
        p = random.uniform(subkeys[1], shape=(10, 3))
        t = random.uniform(subkeys[2], shape=(10, 1))
        psp = PhaseSpacePosition(q, p, t=t)

        # units = None
        wt = psp.wt()
        assert jnp.array_equal(wt[..., :-1], xp.concat((q, p), axis=-1))
        assert jnp.array_equal(wt[..., -1:], t)

        # units != None
        with pytest.raises(NotImplementedError):
            _ = psp.wt(units=galactic)
