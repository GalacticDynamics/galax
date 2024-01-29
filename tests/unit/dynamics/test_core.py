"""Test :mod:`~galax.dynamics.core`."""

import jax.experimental.array_api as xp
import jax.numpy as jnp
import pytest
from jax import random

from galax.dynamics import PhaseSpacePosition
from galax.units import galactic


class TestPhaseSpacePosition:
    """Test :class:`~galax.dynamics.PhaseSpacePosition`."""

    def test_slice(self) -> None:
        """Test slicing."""
        _, *_subkeys = random.split(random.PRNGKey(0), num=9)
        subkeys = iter(_subkeys)

        # Simple
        x = random.uniform(next(subkeys), shape=(10, 3))
        v = random.uniform(next(subkeys), shape=(10, 3))
        o = PhaseSpacePosition(x, v)
        new_o = o[:5]
        assert new_o.shape == (5,)

        # 1d slice on 3d
        x = random.uniform(next(subkeys), shape=(10, 8, 3))
        v = random.uniform(next(subkeys), shape=(10, 8, 3))
        o = PhaseSpacePosition(x, v)
        new_o = o[:5]
        assert new_o.shape == (5, 8)

        # 3d slice on 3d
        o = PhaseSpacePosition(x, v)
        new_o = o[:5, :4]
        assert new_o.shape == (5, 4)

        # Boolean array
        x = random.uniform(next(subkeys), shape=(10, 3))
        v = random.uniform(next(subkeys), shape=(10, 3))
        o = PhaseSpacePosition(x, v)
        ix = xp.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).astype(bool)
        new_o = o[ix]
        assert new_o.shape == (sum(ix),)

        # Integer array
        x = random.uniform(next(subkeys), shape=(10, 3))
        v = random.uniform(next(subkeys), shape=(10, 3))
        o = PhaseSpacePosition(x, v)
        ix = xp.asarray([0, 3, 5])
        new_o = o[ix]
        assert new_o.shape == (len(ix),)

    # ------------------------------------------------------------------------

    def test_len(self) -> None:
        """Test length."""
        _, *_subkeys = random.split(random.PRNGKey(0), num=5)
        subkeys = iter(_subkeys)

        # Simple
        q = random.uniform(next(subkeys), shape=(10, 3))
        p = random.uniform(next(subkeys), shape=(10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 10

        # Complex shape
        q = random.uniform(next(subkeys), shape=(4, 10, 3))
        p = random.uniform(next(subkeys), shape=(4, 10, 3))
        psp = PhaseSpacePosition(q, p)
        assert len(psp) == 4

    # ------------------------------------------------------------------------

    def test_w(self) -> None:
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.w`."""
        _, *_subkeys = random.split(random.PRNGKey(0), num=3)
        subkeys = iter(_subkeys)

        q = random.uniform(next(subkeys), shape=(10, 3))
        p = random.uniform(next(subkeys), shape=(10, 3))
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
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.wt`."""
        _, *_subkeys = random.split(random.PRNGKey(0), num=3)
        subkeys = iter(_subkeys)

        q = random.uniform(next(subkeys), shape=(10, 3))
        p = random.uniform(next(subkeys), shape=(10, 3))
        psp = PhaseSpacePosition(q, p)

        # units = None
        wt = psp.wt()
        assert jnp.array_equal(wt[..., :-1], xp.concat((q, p), axis=-1))
        assert jnp.array_equal(wt[..., -1], xp.zeros(len(wt)))

        # units != None
        with pytest.raises(NotImplementedError):
            _ = psp.wt(units=galactic)

    def test_wt_time(self) -> None:
        """Test :attr:`~galax.dynamics.PhaseSpacePosition.wt`."""
        _, *_subkeys = random.split(random.PRNGKey(0), num=4)
        subkeys = iter(_subkeys)

        q = random.uniform(next(subkeys), shape=(10, 3))
        p = random.uniform(next(subkeys), shape=(10, 3))
        t = random.uniform(next(subkeys), shape=(10, 1))
        psp = PhaseSpacePosition(q, p, t=t)

        # units = None
        wt = psp.wt()
        assert jnp.array_equal(wt[..., :-1], xp.concat((q, p), axis=-1))
        assert jnp.array_equal(wt[..., -1:], t)

        # units != None
        with pytest.raises(NotImplementedError):
            _ = psp.wt(units=galactic)
