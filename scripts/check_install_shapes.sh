#!/usr/bin/env bash
# Verify the two install shapes the split must get right:
#   1. a bare `galax` keeps working exactly as before (criterion 5)
#   2. a single interop package installs on its own (criterion 8, driver 1)
#
# Both need a clean environment: the dev venv has every extra installed, so it
# cannot tell a required dependency from an optional one.
set -euo pipefail

WHEELS=$(mktemp -d)
trap 'rm -rf "$WHEELS"' EXIT

# Build every distribution as 999.0.0, a final release above anything published.
# This does three jobs at once: a published PyPI version can never satisfy an
# install by name (so we cannot test the wrong artifact), the `>0.0.3` pins between
# distributions resolve from the local build, and no `--prerelease=allow` is
# needed -- which would also let the resolver pick pre-release *third-party*
# versions (it picked beartype 0.23.0rc2, which breaks `import galax`).
export SETUPTOOLS_SCM_PRETEND_VERSION=999.0.0

echo "== building wheels and sdists =="
# Both, because an sdist can be broken in ways a wheel is not: hatchling's
# default sdist include differs from the wheel's `packages = [...]`, so a
# package can build a correct wheel and an sdist that omits its own source.
uv build -o "$WHEELS" .
for p in packages/*/; do
  uv build -o "$WHEELS" "$p"
done

echo "== every distribution produced both artifacts =="
for p in . packages/*/; do
  name=$(basename "$(cd "$p" && pwd)")
  [[ "$p" == "." ]] && name="galax"
  norm="${name//./_}"
  ls "$WHEELS/${norm}"-*.whl    >/dev/null || { echo "no wheel for $name"; exit 1; }
  ls "$WHEELS/${norm}"-*.tar.gz >/dev/null || { echo "no sdist for $name"; exit 1; }
done

echo "== shape 1: bare \`pip install galax\` =="
# Install the freshly built root wheel *by path*, so a published PyPI version
# cannot satisfy it; --find-links resolves its inter-distribution pins from the
# same directory.
uv run --isolated --no-project \
  --find-links "$WHEELS" \
  --with "$(echo "$WHEELS"/galax-[0-9]*.whl)" \
  python - <<'PY'
import importlib.util
from importlib.metadata import entry_points

import galax.coordinates, galax.dynamics, galax.potential  # noqa: F401

# astropy interop is a REQUIRED dependency, so it must register with no extras.
eps = {ep.name for ep in entry_points(group="galax.potential.interop")}
assert "astropy" in eps, f"astropy interop missing from a bare install: {eps}"

# Registered is not the same as working. Evaluating a potential at an astropy
# Quantity dispatches only if the plugin really loaded; without it plum raises.
import astropy.units as apyu
import numpy as np

import galax.potential as gp

pot = gp.KeplerPotential(m_tot=1e11, units="galactic")
pot.potential(np.array([8.0, 0.0, 0.0]) * apyu.kpc, 0 * apyu.Myr)

# and the heavy ones must NOT have come along
for absent in ("gala", "galpy", "matplotlib"):
    assert absent not in eps, f"{absent} should not be installed by a bare galax"
    assert importlib.util.find_spec(absent) is None, f"{absent} is importable"

print("bare install OK")
PY

echo "== shape 2: \`galax.interop.gala\` alone =="
uv run --isolated --no-project \
  --find-links "$WHEELS" \
  --with "$(echo "$WHEELS"/galax_interop_gala-*.whl)" \
  python - <<'PY'
from importlib.metadata import entry_points

import galax.interop.gala.optional_deps as od

eps = {ep.name for ep in entry_points(group="galax.potential.interop")}
assert "gala" in eps, f"gala interop did not register: {eps}"
assert "galpy" not in eps, "galpy must not be pulled in by the gala package"
assert isinstance(od.GSL_ENABLED, bool)

# Registered is not the same as working: do a real gala -> galax conversion.
import gala.potential as galap
from gala.units import galactic

import galax.potential as gp

converted = gp.io.convert_potential(
    gp.io.GalaxLibrary, galap.NFWPotential(m=1e12, r_s=20, units=galactic)
)
assert isinstance(converted, gp.NFWPotential), type(converted)
print("subset install OK")
PY

echo "all install shapes OK"
