"""The test tree must not displace a standard-library module.

This tree contains `potential/io/`, and a test package named `io` would be
registered as the module `io` and replace the standard library's for the whole
session -- `io.StringIO` then vanishes in files that have nothing to do with
these tests. Measured, not hypothetical: a probe laid out as `tests/io/` makes
`hasattr(io, "StringIO")` return `False` in an unrelated module.

What prevents it is the inner package. `packages/galax.potential/tests/` has no
`__init__.py` and `tests/potential/` does, so the naming root is `potential`
and this tree's subpackages are `potential.io`, `potential.builtin` and so on.

This guard lives in the tree rather than in `tests/smoke/` because it can only
fail where `potential/io/` has actually been imported, and the smoke job runs
on its own.
"""

import io
import sys

STDLIB_AT_RISK = ("io", "param")


def test_this_tree_is_namespaced_under_potential() -> None:
    """The naming root is `potential`, which is what keeps `io` out of the way."""
    assert __name__.startswith("potential."), __name__


def test_stdlib_io_is_the_real_one() -> None:
    """`potential/io/` must not have become the module `io`."""
    assert hasattr(io, "StringIO"), io.__file__
    assert io.StringIO("x").read() == "x"


def test_no_subpackage_displaced_a_stdlib_module() -> None:
    """Nothing in this tree may own a top-level stdlib name.

    `param` is listed beside `io` because `potential/param/` is the other
    subdirectory whose bare name could plausibly collide one day.
    """
    hijacked = [
        name
        for name in STDLIB_AT_RISK
        if (mod := sys.modules.get(name)) is not None
        and "galax" in str(getattr(mod, "__file__", "") or "")
    ]
    assert not hijacked, f"a test package displaced a stdlib module: {hijacked}"
