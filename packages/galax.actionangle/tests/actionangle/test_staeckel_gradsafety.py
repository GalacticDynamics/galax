"""Gradient safety of the Staeckel-fudge limits and Taylor coefficients."""

import pytest

from ._helpers import check_grad_safety

# The composites (MW2022 and a disk + NFW halo), the slowest compiles, have
# their own file (test_staeckel_gradsafety_composite) for parallel runs. The F
# closures' limit sites are checked in test_staeckel_prolate.


# MN is smooth like KK and makes every kill KK did (memory: KK compiled
# the largest transient of the suite, +2.3 GB).
@pytest.mark.parametrize("name", ["mn", "nfw"])
def test_grad_safety(name):
    """See `check_grad_safety`; on MN also `check_circular_fwd_rev`."""
    check_grad_safety(name, circular=name == "mn")
