"""Runtime type checking is switched on from outside galax.

`docs/testing.rst` documents wrapping the first galax import in
`jaxtyping.install_import_hook("galax", "beartype.beartype")`. A hint beartype
cannot handle makes the import itself fail, and the failure depends on import
order, so every portion is imported in a fresh interpreter.
"""

import subprocess
import sys

import pytest

PORTIONS = (
    "coordinates",
    "potential",
    "fields",
    "orbits",
    "nbody",
    "dynamics",
    "actionangle",
)


@pytest.mark.slow
def test_every_portion_imports_under_the_hook() -> None:
    """The documented recipe imports all seven portions without error."""
    code = (
        "from jaxtyping import install_import_hook\n"
        "with install_import_hook('galax', 'beartype.beartype'):\n"
        + "".join(f"    import galax.{portion}\n" for portion in PORTIONS)
        # The import succeeding is not enough: the hook must have wrapped galax.
        + "import galax.fields\n"
        + "assert hasattr(galax.fields.parse_time_specification, '__wrapped__')\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
