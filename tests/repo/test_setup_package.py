"""Tests for the per-portion `setup_package` helpers.

Each portion ships its own copy (see the design doc), so these run against all
of them.
"""

import importlib

from typing import Any, NoReturn

import pytest

PORTIONS = ["coordinates", "potential", "dynamics", "actionangle"]


def fake_entry_point(missing: str) -> Any:
    """Build an entry point whose `load()` fails on a missing module."""

    class EntryPoint:
        def load(self) -> NoReturn:
            msg = f"No module named {missing!r}"
            raise ModuleNotFoundError(msg, name=missing)

    return EntryPoint()


@pytest.mark.parametrize("portion", PORTIONS)
def test_missing_third_party_is_skipped(portion: str, monkeypatch: Any) -> None:
    """A plugin whose third-party target isn't installed is skipped quietly."""
    mod = importlib.import_module(f"galax.{portion}.setup_package")
    monkeypatch.setattr(mod, "entry_points", lambda **_: [fake_entry_point("gala")])

    mod.load_interop_plugins(f"galax.{portion}.interop")  # must not raise


@pytest.mark.parametrize("portion", PORTIONS)
def test_missing_galax_module_propagates(portion: str, monkeypatch: Any) -> None:
    """A missing `galax` module is a real bug, not an absent optional dep."""
    mod = importlib.import_module(f"galax.{portion}.setup_package")
    monkeypatch.setattr(
        mod, "entry_points", lambda **_: [fake_entry_point("galax.interop.nope")]
    )

    with pytest.raises(ModuleNotFoundError, match="galax.interop.nope"):
        mod.load_interop_plugins(f"galax.{portion}.interop")


@pytest.mark.parametrize(
    ("env", "expected"),
    [("False", False), ("None", None), ("beartype.beartype", "beartype.beartype")],
)
@pytest.mark.parametrize("portion", PORTIONS)
def test_runtime_typechecker_from_env(
    portion: str, env: str, expected: object, monkeypatch: Any
) -> None:
    """`GALAX_ENABLE_RUNTIME_TYPECHECKING` is parsed into `RUNTIME_TYPECHECKER`."""
    mod = importlib.import_module(f"galax.{portion}.setup_package")
    monkeypatch.setenv("GALAX_ENABLE_RUNTIME_TYPECHECKING", env)
    try:
        assert expected == importlib.reload(mod).RUNTIME_TYPECHECKER
    finally:
        monkeypatch.undo()
        importlib.reload(mod)
