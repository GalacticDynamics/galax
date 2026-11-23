"""galax has no built-in runtime-typechecking hook.

Runtime type checking is switched on from outside galax, with jaxtyping's import
hook (see `docs/testing.rst`). A portion that wraps its own imports in a hook, or
reads `GALAX_ENABLE_RUNTIME_TYPECHECKING`, would be a second, undocumented way to
do the same thing.
"""

import pathlib

ROOT = pathlib.Path(__file__).parents[2]
SOURCES = sorted(ROOT.glob("packages/*/src/**/*.py"))
REMOVED = ("install_import_hook", "RUNTIME_TYPECHECKER", "RUNTIME_TYPECHECKING")


def test_the_scan_finds_sources() -> None:
    """Guard the guard: a glob that matches nothing would prove nothing."""
    assert len(SOURCES) > 100


def test_no_test_sets_the_removed_variable() -> None:
    """Nothing sets `GALAX_ENABLE_RUNTIME_TYPECHECKING`: it would check nothing now.

    A test that sets it to turn type checking on keeps passing after the variable
    stopped being read, and stops testing what it says it tests.
    """
    tests = [
        *ROOT.glob("packages/*/tests/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(tests)
        if path != pathlib.Path(__file__)
        and "GALAX_ENABLE_RUNTIME_TYPECHECKING" in path.read_text(encoding="utf-8")
    ]
    assert (
        not offenders
    ), f"these tests set a variable galax no longer reads: {offenders}"


def test_no_source_uses_the_builtin_hook() -> None:
    """No module under `packages/*/src` mentions the removed hook or variable."""
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in SOURCES
        for name in REMOVED
        if name in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"the built-in typechecking hook is back:\n{offenders}"
