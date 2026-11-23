"""Runtime type checking is switched on from outside galax.

`docs/testing.rst` documents wrapping the first galax import in
`jaxtyping.install_import_hook("galax", "beartype.beartype")`. A hint beartype
cannot handle makes the import itself fail, and the failure depends on import
order, so the seven portions are imported together in a fresh interpreter, with
`coordinates` first.
"""

import os
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
    # Galax's own hook reads this variable, and would wrap the modules by itself.
    env = {
        k: v for k, v in os.environ.items() if k != "GALAX_ENABLE_RUNTIME_TYPECHECKING"
    }
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
