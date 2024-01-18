import re

import astropy.units as u
import jax.experimental.array_api as xp
import pytest

from galax.integrate import parse_time_specification
from galax.units import dimensionless


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (  # Case 1
            {"t": None, "t1": None, "t2": None, "n_steps": None, "dt": None},
            (ValueError, "must specify some combination of t, t1, t2, n_steps, dt"),
        ),
        # (  # Case 2a
        #     {
        #         "t": xp.linspace(0, 1, 10),
        #         "t1": None,
        #         "t2": None,
        #         "n_steps": None,
        #         "dt": None,
        #     },
        #     xp.linspace(0, 1, 10),
        # ),
        # (  # Case 2b
        #     {
        #         "t": xp.linspace(0, 1, 10) * u.dimensionless_unscaled,
        #         "t1": None,
        #         "t2": None,
        #         "n_steps": None,
        #         "dt": None,
        #     },
        #     xp.linspace(0, 1, 10),
        # ),
        # (  # Case 3
        #     {"t": None, "t1": 0, "t2": 1, "n_steps": 10, "dt": None},
        #     xp.linspace(0, 1, 10),
        # ),
        # (  # Case 4
        #     {"t": None, "t1": 0, "t2": None, "n_steps": 10, "dt": 0.1},
        #     xp.linspace(0, 1, 10),
        # ),
        # (  # Case 5
        #     {"t": None, "t1": 0, "t2": 1, "dt": 0.1},
        #     xp.linspace(0, 1, 10),
        # ),
        # (  # Case 6
        #     {"t": None, "t1": 0, "dt": xp.asarray([0.1] * 10)},
        #     xp.linspace(0, 1, 10),
        # ),
    ],
)
def test_parse_time_specification(kwargs, expected):
    if isinstance(expected, tuple):
        with pytest.raises(expected[0], match=re.escape(expected[1])):
            parse_time_specification(units=dimensionless, **kwargs)
        return

    assert parse_time_specification(units=dimensionless, **kwargs) == expected
