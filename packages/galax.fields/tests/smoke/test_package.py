"""Testing :mod:`galax.fields` module."""

import galax.fields as gf
import galax.fields.utils as gfu


def test_all() -> None:
    """Test the `galax.fields` API."""
    assert set(gf.__all__) == {
        "AbstractField",
        "AbstractSolver",
        "SolveState",
        "integrate_field",
        "parse_time_specification",
    }
    for name in gf.__all__:
        assert hasattr(gf, name)


def test_utils_all() -> None:
    """Test the `galax.fields.utils` API."""
    assert set(gfu.__all__) == {"parse_saveat", "parse_to_t_y"}
    for name in gfu.__all__:
        assert hasattr(gfu, name)
