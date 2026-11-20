"""Gradient safety on composites (its own file: the slowest compiles)."""

import pytest

from ._helpers import check_grad_safety


@pytest.mark.parametrize(
    ("name", "part"),
    [("mw2022", "limits"), ("composite", "taylor_l"), ("composite", "taylor_n")],
)
def test_grad_safety_composite(name, part):
    """See `check_grad_safety` and `check_circular_fwd_rev`.

    MW2022's limits; the Taylor coefficients, one nest per program, on a
    smaller disk + NFW-halo composite: MW2022's lambda nest differentiated
    in jacfwd or jacrev takes ~2.5 GB to compile.
    """
    check_grad_safety(name, circular=True, part=part)
