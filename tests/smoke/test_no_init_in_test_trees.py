"""Test trees must not be packages.

With no `__init__.py`, `--import-mode=importlib` names a test module after its
whole path from the repo root -- `test_base.py` in `galax.potential`'s tree
becomes `packages.galax_potential.tests.potential.test_base`. Add one and the
naming root moves to the first directory that has it, so the tree's
subdirectories become top-level modules under their bare names.

That is how a test directory displaces a library. `galax.potential`'s tree
contains `potential/io/`, and with `tests/potential/__init__.py` present the
root was `potential`, leaving `potential.io`; one more `__init__.py` higher up
and it would have been `io`, registered as the standard library's module and
replacing it for the whole session -- `io.StringIO` then vanishes in files that
have nothing to do with these tests. Measured, not hypothetical: a probe laid
out that way makes `hasattr(io, "StringIO")` return `False` elsewhere.

Collection does not need these files: dropping all 23 left the count at 5236
exactly. So the invariant is cheap, and this guard replaces the three
assertions that used to sit inside `galax.potential`'s tree checking the
consequence (`test_stdlib_not_shadowed.py`) rather than the cause. Checking the
cause also works from here, where the earlier version could not -- it could
only fail in a session that had already imported the offending tree.

`matplotlib`'s integration directory kept its documentation when its
`__init__.py` went: see that tree's `README.md`.
"""

import pathlib

ROOT = pathlib.Path(__file__).parents[2]

# `packages/*/tests` does not exist until each portion owns its own tests; the
# glob simply finds nothing then.
TEST_TREES = (
    "tests",
    *(f"packages/{p.name}/tests" for p in sorted((ROOT / "packages").glob("*"))),
)


def test_no_test_tree_is_a_package() -> None:
    """An `__init__.py` anywhere under a test tree shortens the naming root."""
    offenders = sorted(
        p.relative_to(ROOT).as_posix()
        for tree in TEST_TREES
        for p in (ROOT / tree).rglob("__init__.py")
    )
    assert not offenders, (
        "test trees must not be packages -- these shorten the module naming "
        f"root and can displace a stdlib module: {offenders}"
    )
