"""Testing :mod:`galax.nbody` module."""

import ast
import re
import sys
import tomllib
from pathlib import Path

import galax.nbody as gn


def test_all() -> None:
    """Test the `galax.nbody` API."""
    assert set(gn.__all__) == {
        "NBodyField",
        "Component",
        "System",
        "Particles",
        "DirectSum",
        "PlummerKernel",
        "AbstractSofteningKernel",
        "AbstractForceBackend",
        "Sources",
        "Targets",
        "AbstractGradPolicy",
        "Autodiff",
        "StopGrad",
        "NoGrad",
        "total_energy",
        "evolve",
        "NBodyResult",
        "check_timestep",
    }
    for name in gn.__all__:
        assert hasattr(gn, name)


def test_imports_are_declared_dependencies() -> None:
    """Every package the source imports is in ``[project].dependencies``."""
    root = Path(__file__).parents[2]
    with (root / "pyproject.toml").open("rb") as f:
        project = tomllib.load(f)
    declared = {
        re.match(r"[A-Za-z0-9._-]+", d)[0] for d in project["project"]["dependencies"]
    }
    # `ast` finds imports anywhere, including indented ones in functions and
    # `with` blocks. Relative imports (level > 0) stay inside the package.
    mods = []
    for path in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mods.append(node.module)
    imported = set()
    for mod in mods:
        parts = mod.split(".")
        name = ".".join(parts[:2]) if parts[0] == "galax" else parts[0]
        if name not in sys.stdlib_module_names and name not in {
            "galax.nbody",
            "__future__",
        }:
            imported.add(name)
    assert imported - declared == set()
    sources = project["tool"]["uv"]["sources"]
    assert {d for d in declared if d.startswith("galax.")} <= set(sources)
