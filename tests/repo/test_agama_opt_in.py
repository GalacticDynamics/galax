"""Agama comparisons are opt-in: skipped unless ``--run-agama`` is passed.

The root conftest owns the opt-in; whether agama is installed is each
portion's own `optional_deps` probe. The hook keys on the flag, not on `-m`,
so CI's ``-m"not slow"`` cannot switch these tests on by replacing a default
marker expression.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

# Load the root conftest by location, as `test_conftest_shim.py` does: putting
# the repo root on `sys.path` would make `docs`, `packages`, ... importable.
_spec = importlib.util.spec_from_file_location(
    "galax_root_conftest_agama", ROOT / "conftest.py"
)
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)


class _Config:
    def __init__(self, *, run_agama: bool) -> None:
        self._run = run_agama

    def getoption(self, name: str) -> bool:
        assert name == "--run-agama"
        return self._run


class _Item:
    def __init__(self, *keywords: str) -> None:
        self.keywords = dict.fromkeys(keywords)
        self.added: list[pytest.MarkDecorator] = []

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)


def test_agama_tests_skip_without_the_flag() -> None:
    item = _Item("agama")
    conftest.pytest_collection_modifyitems(_Config(run_agama=False), [item])
    assert [m.name for m in item.added] == ["skip"]
    assert item.added[0].kwargs["reason"] == "compares against agama; pass --run-agama"


def test_agama_tests_run_with_the_flag() -> None:
    item = _Item("agama")
    conftest.pytest_collection_modifyitems(_Config(run_agama=True), [item])
    assert item.added == []


def test_unmarked_tests_are_never_touched() -> None:
    item = _Item("slow")
    conftest.pytest_collection_modifyitems(_Config(run_agama=False), [item])
    assert item.added == []
