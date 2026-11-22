"""Tests for `galax.setup`."""

import contextlib
import importlib

from typing import Any, NoReturn

import pytest

import galax.setup as gs


def fake_entry_point(missing: str) -> Any:
    """Build an entry point whose `load()` fails on a missing module."""

    class EntryPoint:
        def load(self) -> NoReturn:
            msg = f"No module named {missing!r}"
            raise ModuleNotFoundError(msg, name=missing)

    return EntryPoint()


def test_missing_third_party_is_skipped(monkeypatch: Any) -> None:
    """A plugin whose third-party target isn't installed is skipped quietly."""
    monkeypatch.setattr(gs, "entry_points", lambda **_: [fake_entry_point("gala")])

    gs.load_interop_plugins("galax.potential.interop")  # must not raise


def test_missing_galax_module_propagates(monkeypatch: Any) -> None:
    """A missing `galax` module is a real bug, not an absent optional dep."""
    monkeypatch.setattr(
        gs, "entry_points", lambda **_: [fake_entry_point("galax.interop.nope")]
    )

    with pytest.raises(ModuleNotFoundError, match="galax.interop.nope"):
        gs.load_interop_plugins("galax.potential.interop")


@pytest.mark.parametrize(
    ("env", "expected"),
    [("False", False), ("None", None), ("beartype.beartype", "beartype.beartype")],
)
def test_runtime_typechecker_from_env(
    env: str, expected: object, monkeypatch: Any
) -> None:
    """`GALAX_ENABLE_RUNTIME_TYPECHECKING` is parsed into `RUNTIME_TYPECHECKER`."""
    monkeypatch.setenv("GALAX_ENABLE_RUNTIME_TYPECHECKING", env)
    try:
        assert expected == importlib.reload(gs).RUNTIME_TYPECHECKER
    finally:
        monkeypatch.undo()
        importlib.reload(gs)


def test_hook_is_a_null_context_when_disabled(monkeypatch: Any) -> None:
    """With typechecking off the hook does nothing and costs nothing."""
    monkeypatch.setattr(gs, "RUNTIME_TYPECHECKER", False)

    assert isinstance(gs.install_import_hook("galax.setup"), contextlib.nullcontext)
