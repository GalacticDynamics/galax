"""Tests for the per-portion `load_interop_plugins` helper.

Each portion that loads interop plugins ships its own copy in `setup_package`,
so these run against all of them.
"""

import importlib

from typing import Any, NoReturn

import pytest

# The portions with a `*.interop` entry-point group, the only ones that load plugins.
LOADING_PORTIONS = ["coordinates", "potential", "dynamics"]


def fake_entry_point(missing: str) -> Any:
    """Build an entry point whose `load()` fails on a missing module."""

    class EntryPoint:
        def load(self) -> NoReturn:
            msg = f"No module named {missing!r}"
            raise ModuleNotFoundError(msg, name=missing)

    return EntryPoint()


@pytest.mark.parametrize("portion", LOADING_PORTIONS)
def test_missing_third_party_is_skipped(portion: str, monkeypatch: Any) -> None:
    """A plugin whose third-party target isn't installed is skipped quietly."""
    mod = importlib.import_module(f"galax.{portion}.setup_package")
    monkeypatch.setattr(mod, "entry_points", lambda **_: [fake_entry_point("gala")])

    mod.load_interop_plugins(f"galax.{portion}.interop")  # must not raise


@pytest.mark.parametrize("portion", LOADING_PORTIONS)
def test_missing_galax_module_propagates(portion: str, monkeypatch: Any) -> None:
    """A missing `galax` module is a real bug, not an absent optional dep."""
    mod = importlib.import_module(f"galax.{portion}.setup_package")
    monkeypatch.setattr(
        mod, "entry_points", lambda **_: [fake_entry_point("galax.interop.nope")]
    )

    with pytest.raises(ModuleNotFoundError, match="galax.interop.nope"):
        mod.load_interop_plugins(f"galax.{portion}.interop")
