"""Portions must be usable without importing the others first.

`galax` is a PEP 420 namespace package, so `import galax.coordinates` does not
drag in `galax.potential` or `galax.dynamics`. Lazy, out-of-order portion
imports are therefore ordinary user code, and must stay safe.

These run in a subprocess: the import order *is* the thing under test, and the
pytest session has already imported everything.
"""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest


def run(
    code: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter, with `env` (default: this one's)."""
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
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


@pytest.mark.parametrize(
    "portion",
    [
        "coordinates",
        "potential",
        "fields",
        "orbits",
        "nbody",
        "dynamics",
        "actionangle",
    ],
)
def test_portion_imports_alone(portion: str) -> None:
    """Each portion imports on its own, with no sibling imported first."""
    proc = run(f"import galax.{portion}")
    assert proc.returncode == 0, proc.stderr


# `fields` and `coordinates` are at the bottom and import no other galax
# portion. `potential` sits above `coordinates`, and `orbits` above `fields`,
# `coordinates` and `potential`. `nbody` is above `orbits`, `dynamics` above
# `nbody`, and `actionangle` above `dynamics`. A portion may import the ones
# below it, never the ones above.
ABOVE = {
    "coordinates": [
        "potential",
        "fields",
        "orbits",
        "nbody",
        "dynamics",
        "actionangle",
    ],
    "potential": ["fields", "orbits", "nbody", "dynamics", "actionangle"],
    "fields": [
        "coordinates",
        "potential",
        "orbits",
        "nbody",
        "dynamics",
        "actionangle",
    ],
    "orbits": ["nbody", "dynamics", "actionangle"],
    "nbody": ["dynamics", "actionangle"],
    "dynamics": ["actionangle"],
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

        gp.potential_energy(pot, w)
    """)
    assert proc.returncode == 0, proc.stderr
    assert "UnexpectedTracerError" not in proc.stderr


def test_x64_enabled_after_import() -> None:
    """Turning on x64 after importing galax still gives float64 values.

    Regression: ``G`` (and other constants and defaults) were built as JAX
    arrays when galax was imported, so importing galax before
    ``jax_enable_x64`` was turned on left them float32 for the whole process.
    ``G`` was then 4.4985023e-12 instead of 4.49850215e-12 kpc3 / (Msun Myr2).
    This checks ``G`` in a potential, in each default component of
    `galax.potential.MilkyWayPotential` (whose float32 and traced builds come
    first) and in `galax.nbody.NBodyField`, and some parameter defaults.

    pytest sets ``JAX_ENABLE_X64`` for the whole session, so the subprocess
    runs with it removed: x64 is off at import, as in a plain script.
    """
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    proc = run(
        """
        import jax
        import unxt as u

        import galax.nbody as gn
        import galax.potential as gp

        assert not jax.config.jax_enable_x64
        gp.MilkyWayPotential()  # caches the float32 default components
        jax.config.update("jax_enable_x64", True)
        # The first float64 build of the default components happens in a trace.
        jax.jit(lambda: gp.MilkyWayPotential()["disk"].constants["G"].value)()

        unit = "kpc3 / (solMass Myr2)"
        G = u.Q(4.498502151469554e-12, unit)  # astropy CODATA 2018

        def check(name, g):
            g = u.uconvert(unit, g)
            assert g.value.dtype == "float64", (name, g.value.dtype)
            assert abs(g.value / G.value - 1) < 1e-14, (name, g.value.item())

        check("NFW", gp.NFWPotential(m=1e12, r_s=20, units="galactic").constants["G"])
        for k, pot in gp.MilkyWayPotential().items():
            assert "Tracer" not in type(pot.constants["G"].value).__name__, k
            check(f"MilkyWayPotential.{k}", pot.constants["G"])
        field = gn.NBodyField(u.Q([1.0], "Msun"), eps=u.Q(0.1, "kpc"))
        check("NBodyField", u.Q(field._G, unit))

        # Parameter defaults are built per instance too, and stay weakly typed.
        bar = gp.MonariEtAl2016BarPotential(
            R0=u.Q(8.0, "kpc"), v0=u.Q(220.0, "km/s"), units="galactic"
        )
        tri = gp.TriaxialHernquistPotential(
            m_tot=u.Q(1e10, "Msun"), r_s=u.Q(1.0, "kpc"), units="galactic"
        )
        for pot, name, expect in [
            (bar, "alpha", 0.01),
            (bar, "Omega", 52.2),
            (bar, "Rb", 3.5),
            (tri, "q1", 1.0),
        ]:
            q = getattr(pot, name).value
            assert q.value.dtype == "float64", (name, q.value.dtype)
            assert q.value.weak_type, name
            assert q.value.item() == expect, (name, q.value.item())
        """,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


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
    root = pathlib.Path(__file__).parents[2]
    # Every tree that contributes to the `galax` namespace; `packages/*/src` is
    # globbed so portions moving out of `src/` stay covered.
    src_roots = [root / "src", *sorted(root.glob("packages/*/src"))]
    offenders: list[str] = []

    for portion, higher in ABOVE.items():
        banned = tuple(f"galax.{h}" for h in higher)
        # `rglob` on a missing directory yields nothing and raises nothing, so
        # a portion that is not found must fail here rather than scan 0 files.
        files = sorted(
            path
            for src in src_roots
            for path in (src / "galax" / portion).rglob("*.py")
        )
        assert files, f"portion {portion!r} not found under any of {src_roots}"
        for path in files:
            tree = ast.parse(path.read_text(), filename=str(path))
            offenders += [
                f"{path.relative_to(root)}:{node.lineno} "
                f"in {func.name}(): imports {name}"
                for func in ast.walk(tree)
                if isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef)
                for node in ast.walk(func)
                for name in _imported_modules(node)
                if _within(name, banned)
            ]

    assert not offenders, "deferred upward imports:\n" + "\n".join(offenders)
