"""Agama comparisons are opt-in: skipped unless ``--run-agama`` is passed.

The root conftest owns the opt-in; whether agama is installed is each
portion's own `optional_deps` probe. The hook keys on the flag, not on `-m`,
so CI's ``-m"not slow"`` cannot switch these tests on by replacing a default
marker expression.
"""

import importlib.util
from pathlib import Path

from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]

# Load the root conftest by location, as `test_conftest_shim.py` does: putting
# the repo root on `sys.path` would make `docs`, `packages`, ... importable.
_spec = importlib.util.spec_from_file_location(
    "galax_root_conftest_agama", ROOT / "conftest.py"
)
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)


@pytest.mark.parametrize(
    ("keyword", "run", "skipped"),
    [("agama", False, True), ("agama", True, False), ("slow", False, False)],
)
def test_only_agama_tests_skip_and_only_without_the_flag(
    keyword: str, run: bool, skipped: bool
) -> None:
    added: list[pytest.MarkDecorator] = []
    config = SimpleNamespace(getoption=lambda _: run)
    item = SimpleNamespace(keywords=dict.fromkeys([keyword]), add_marker=added.append)
    conftest.pytest_collection_modifyitems(config, [item])
    assert [m.name for m in added] == (["skip"] if skipped else [])
    if skipped:
        assert added[0].kwargs["reason"] == "compares against agama; pass --run-agama"
