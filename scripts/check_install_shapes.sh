#!/usr/bin/env bash
# Verify the install shapes that matter:
#   1. a bare `pip install galax` gives a working core plus the astropy interop
#   2. a single interop package installs on its own
#   3. `galax.coordinates` installs on its own, without the heavier portions
#
# Both need a clean environment: the dev venv has every extra installed, so it
# cannot tell a required dependency from an optional one.
set -euo pipefail

WHEELS=$(mktemp -d)

# hatchling's version-file hook writes `_version.py` into each package's source
# tree, and those gitignored files back the editable dev install. Building as
# 999.0.0 (below) would leave `galax.interop.<lib>.__version__` reading 999.0.0
# after every run, so put them back on exit.
VERSION_FILES=()
for p in packages/*/; do
  # galax.interop.gala -> galax/interop/gala; galax.coordinates -> galax/coordinates
  leaf=$(basename "$p")
  VERSION_FILES+=("${p}src/${leaf//.//}/_version.py")
done
BACKUP=$(mktemp -d)
for f in "${VERSION_FILES[@]}"; do
  if [[ -e "$f" ]]; then cp -p "$f" "$BACKUP/$(echo "$f" | tr / _)"; fi
done
restore_version_files() {
  for f in "${VERSION_FILES[@]}"; do
    saved="$BACKUP/$(echo "$f" | tr / _)"
    if [[ -e "$saved" ]]; then cp -p "$saved" "$f"; else rm -f "$f"; fi
  done
  rm -rf "$WHEELS" "$BACKUP"
}
trap restore_version_files EXIT

# Build every distribution as 999.0.0, a final release above anything published.
# This does three jobs at once: a published PyPI version can never satisfy an
# install by name (so we cannot test the wrong artifact), the `>0.0.3` pins between
# distributions resolve from the local build, and no `--prerelease=allow` is
# needed -- which would also let the resolver pick pre-release *third-party*
# versions, and a pre-release dependency can break `import galax` outright.
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

echo "== every wheel actually contains its source =="
# `uv build` builds each wheel *from* its sdist, so an sdist that omits its source
# gives an empty wheel. Shapes 1 and 2 only import galax, astropy and gala; this
# covers galpy and matplotlib too.
for whl in "$WHEELS"/*.whl; do
  unzip -Z1 "$whl" | grep -E '^galax/.*\.py$' | grep -v '/_version\.py$' >/dev/null \
    || { echo "$(basename "$whl") contains no galax source (only _version.py, or nothing)"; exit 1; }
done

echo "== the root wheel ships no galax/interop/ =="
# Every `galax>0.0.3` floor in packages/ assumes the first post-split root
# release has no interop tree. A root wheel that still shipped it would let
# `galax` and `galax.interop.*` write the same files, which is what the strict
# floor exists to prevent.
if unzip -Z1 "$WHEELS"/galax-[0-9]*.whl | grep -E '^galax/interop/' >/dev/null; then
  echo "the root galax wheel still contains galax/interop/ files"; exit 1
fi

echo "== the root wheel ships no galax/coordinates/ =="
# `galax.coordinates` owns that leaf now. A root wheel still carrying it would
# let two distributions write the same files.
if unzip -Z1 "$WHEELS"/galax-[0-9]*.whl | grep -E '^galax/coordinates/' >/dev/null; then
  echo "the root galax wheel still contains galax/coordinates/ files"; exit 1
fi

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

import astropy.units as apyu
import numpy as np

# The three core portions must all import on a bare install; `galax.coordinates`
# and `galax.dynamics` are imported for that alone.
import galax.coordinates
import galax.dynamics
import galax.potential as gp

# astropy interop is a REQUIRED dependency, so it must register with no extras.
eps = {ep.name for ep in entry_points(group="galax.potential.interop")}
assert "astropy" in eps, f"astropy interop missing from a bare install: {eps}"

# Registered is not the same as working. Evaluating a potential at an astropy
# Quantity dispatches only if the plugin really loaded; without it plum raises.
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

import gala.potential as galap
from gala.units import galactic

import galax.interop.gala.optional_deps as od
import galax.potential as gp

eps = {ep.name for ep in entry_points(group="galax.potential.interop")}
assert "gala" in eps, f"gala interop did not register: {eps}"
assert "galpy" not in eps, "galpy must not be pulled in by the gala package"
assert isinstance(od.GSL_ENABLED, bool)

# Registered is not the same as working: do a real gala -> galax conversion.
converted = gp.io.convert_potential(
    gp.io.GalaxLibrary, galap.NFWPotential(m=1e12, r_s=20, units=galactic)
)
assert isinstance(converted, gp.NFWPotential), type(converted)
print("subset install OK")
PY

echo "== shape 3: \`galax.coordinates\` alone =="
# The leaf must install without dragging in the potential or dynamics stacks.
# Installed by path so a published version cannot satisfy it.
uv run --isolated --no-project \
  --find-links "$WHEELS" \
  --with "$(echo "$WHEELS"/galax_coordinates-*.whl)" \
  python - <<'PY'
import importlib.util

import unxt as u

import galax.coordinates as gc

# Registered is not the same as working: build a coordinate and read it back.
w = gc.PhaseSpacePosition(q=u.Q([8.0, 0, 0], "kpc"), p=u.Q([0.0, 220, 0], "km/s"))
assert w.q.shape == (), w.q.shape

# and the heavier portions must NOT have come along
for absent in ("galax.potential", "galax.dynamics", "diffrax", "optimistix"):
    assert importlib.util.find_spec(absent) is None, f"{absent} is importable"

print("coordinates-only install OK")
PY

echo "all install shapes OK"
