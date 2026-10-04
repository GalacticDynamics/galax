"""Shared namespace directories must stay empty of owned modules.

A PEP 420 namespace directory is contributed to by several distributions. If one
of them ships an `__init__.py` there, the directory becomes a regular package and
every other distribution's contribution disappears. If one ships a loose module
there, two distributions can claim the same path and the installed result depends
on install order.

This is the rule that retired `galax/_version.py` and forced the per-library
`optional_deps` split. It had no test.
"""

import pathlib

import pytest

# Directories contributed to by more than one distribution. Everything else is
# owned outright by exactly one, and may hold whatever it likes.
SHARED_NAMESPACE_DIRS = ("galax", "galax/interop")

SRC_ROOTS = [
    p
    for p in [
        pathlib.Path(__file__).parents[2] / "src",
        *sorted((pathlib.Path(__file__).parents[2] / "packages").glob("*/src")),
    ]
    if p.is_dir()
]


@pytest.mark.parametrize("shared", SHARED_NAMESPACE_DIRS)
def test_no_owned_module_in_shared_namespace_dir(shared: str) -> None:
    """No `__init__.py` and no loose `*.py` directly in a shared directory."""
    offenders = [
        str(f.relative_to(root.parent))
        for root in SRC_ROOTS
        if (d := root / shared).is_dir()
        for f in sorted(d.glob("*.py"))
    ]
    assert not offenders, (
        f"{shared}/ is shared by several distributions and must hold no module "
        f"of its own; found: {offenders}"
    )
