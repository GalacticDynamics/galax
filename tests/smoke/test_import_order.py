"""Portions must be usable without importing the others first.

`galax` is a PEP 420 namespace package, so `import galax.coordinates` no longer
drags in `galax.potential` and `galax.dynamics` the way the old
`galax/__init__.py` did. That makes lazy, out-of-order portion imports ordinary
user code, and it must stay safe.

These run in a subprocess: the import order *is* the thing under test, and the
pytest session has already imported everything.
"""

import subprocess
import sys
import textwrap

import pytest


def run(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("portion", ["coordinates", "potential", "dynamics"])
def test_portion_imports_alone(portion: str) -> None:
    """Each portion imports on its own, with no sibling imported first."""
    proc = run(f"import galax.{portion}")
    assert proc.returncode == 0, proc.stderr


# `coordinates` -> `potential` -> `dynamics`: each portion may import the ones
# below it, never the ones above.
ABOVE = {
    "coordinates": ["potential", "dynamics"],
    "potential": ["dynamics"],
}


@pytest.mark.parametrize(
    ("portion", "higher"), [(p, h) for p, hs in ABOVE.items() for h in hs]
)
def test_portion_does_not_import_upward(portion: str, higher: str) -> None:
    """Importing a portion must not drag in one above it in the hierarchy.

    An upward import is a hard dependency in the wrong direction, which blocks
    splitting the portions into separate distributions. It is also how the
    tracer-leak regression above became possible: the edge only has to exist
    for a deferred import to fire at an arbitrary moment.
    """
    proc = run(f"""
        import sys
        import galax.{portion}
        assert "galax.{higher}" not in sys.modules, (
            "galax.{portion} imported galax.{higher}"
        )
    """)
    assert proc.returncode == 0, proc.stderr


def test_jit_then_late_potential_import() -> None:
    """A jitted call before `galax.potential` is imported must not leak a tracer.

    Regression: `AbstractPhaseSpaceObject.angular_momentum` was `jax.jit`-ed and
    deferred `from galax.dynamics import specific_angular_momentum` in its body,
    so on first call that import ran *inside a live trace*. It pulled in
    `galax.potential`, whose module-level `default_constants` was then built from
    tracers -- poisoning every potential constructed afterwards with
    `UnexpectedTracerError`.
    """
    proc = run("""
        import coordinax as cx
        import unxt as u

        import galax.coordinates as gc

        q = cx.CartesianPos3D(x=u.Q(1, "kpc"), y=u.Q([1.0, 2], "kpc"), z=u.Q(2, "kpc"))
        p = cx.CartesianVel3D(
            x=u.Q(0, "km/s"), y=u.Q([1.0, 2], "km/s"), z=u.Q(0, "km/s")
        )
        w = gc.PhaseSpaceCoordinate(q, p, t=u.Q(0, "Myr"))

        w.angular_momentum()  # traces, and pulls in galax.dynamics

        import galax.potential as gp

        w.potential_energy(gp.MilkyWayPotential())
    """)
    assert proc.returncode == 0, proc.stderr
    assert "UnexpectedTracerError" not in proc.stderr
