"""Portions must be usable without importing the others first.

`galax` is a PEP 420 namespace package, so `import galax.coordinates` no longer
drags in `galax.potential` and `galax.dynamics` the way the old
`galax/__init__.py` did. That makes lazy, out-of-order portion imports ordinary
user code, and it must stay safe.

These run in a subprocess: the import order *is* the thing under test, and the
pytest session has already imported everything.
"""

import ast
import pathlib
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


def _imported_modules(node: ast.AST, /) -> list[str]:
    """Return the module names an `import` / `from ... import` pulls in."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def _within(name: str, packages: tuple[str, ...], /) -> bool:
    """Whether `name` is one of `packages` or a submodule of one.

    Not `str.startswith`, which would also match a sibling like
    `galax.dynamicsfoo`.
    """
    return any(name == pkg or name.startswith(f"{pkg}.") for pkg in packages)


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

    That method is gone, so this uses `kinetic_energy`, which is still
    `jax.jit`-ed and stays inside `galax.coordinates`. The guard is deliberately
    outcome-level: whatever a traced `galax.coordinates` call does internally,
    importing `galax.potential` afterwards must yield usable constants. It is the
    black-box counterpart to `test_no_deferred_upward_imports`, which forbids the
    mechanism structurally.
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

        w.kinetic_energy()  # traces, with galax.potential not yet imported

        import galax.potential as gp

        pot = gp.MilkyWayPotential()

        # `G` is the value the original bug poisoned, via module-level
        # `default_constants` being built under the trace.
        g = pot.constants["G"]
        assert type(g.value).__name__ != "DynamicJaxprTracer", type(g.value)

        w.potential_energy(pot)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "UnexpectedTracerError" not in proc.stderr


def test_no_deferred_upward_imports() -> None:
    """No function body in a portion may import a portion above it.

    A module-scope upward import is impossible (it would be a cycle), so an
    upward edge can only ever appear deferred inside a function. That is exactly
    the shape that fired mid-trace and poisoned `galax.potential`'s module state,
    and exactly what blocks splitting the portions into distributions.

    This is the structural check. `sys.modules` cannot supply it: a deferred
    import that has not fired yet leaves no trace there, so
    `test_portion_does_not_import_upward` passes on an edge that still exists.
    """
    src = pathlib.Path(__file__).parents[2] / "src" / "galax"
    offenders: list[str] = []

    for portion, higher in ABOVE.items():
        banned = tuple(f"galax.{h}" for h in higher)
        for path in sorted((src / portion).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            offenders += [
                f"{path.relative_to(src)}:{node.lineno} "
                f"in {func.name}(): imports {name}"
                for func in ast.walk(tree)
                if isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef)
                for node in ast.walk(func)
                for name in _imported_modules(node)
                if _within(name, banned)
            ]

    assert not offenders, "deferred upward imports:\n" + "\n".join(offenders)
