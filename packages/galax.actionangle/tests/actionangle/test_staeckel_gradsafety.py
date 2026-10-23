"""Gradient safety of the Staeckel-fudge limits and Taylor coefficients."""

import pytest

from ._helpers import check_grad_safety

# MW2022, the slowest compile, has its own file for parallel runs. The F
# closures' limit sites are checked in test_staeckel_prolate.


@pytest.mark.parametrize("name", ["kk", "mn", "nfw"])
def test_grad_safety(name):
    """See `check_grad_safety`; on MN also `check_circular_fwd_rev`."""
    check_grad_safety(name, circular=name == "mn")
