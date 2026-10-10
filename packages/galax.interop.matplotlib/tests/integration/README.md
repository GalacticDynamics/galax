# matplotlib integration tests

These compare generated figures against committed baselines with the
`pytest-mpl` plugin, which also registers the `mpl_image_compare` marker the
tests are decorated with -- `--strict-markers` rejects it otherwise, so
collection needs the plugin whether or not `--mpl` is passed.

## `hashes.json` keys follow the import path

`pytest-mpl` keys the hash library by the test's module path plus its name, so
an entry reads
`packages.galax_interop_matplotlib.tests.integration.test_orbit.test_orbit_plot`.
Moving or renaming this tree changes those keys and every lookup misses --
`Hash for test '...' not found` -- even though the figures are unchanged. Fix
that by renaming the keys, not by regenerating: the hashes are produced on Linux
in CI and differ from a local macOS render, so regenerating elsewhere replaces
correct values with ones CI will reject.

The baseline _images_ are named after the test function alone, so they are not
affected.

## Regenerating the baselines

```console
uv run pytest packages/galax.interop.matplotlib/tests/integration \
  --mpl-generate-path=packages/galax.interop.matplotlib/tests/integration/baseline
uv run pytest packages/galax.interop.matplotlib/tests/integration \
  --mpl-generate-hash-library=packages/galax.interop.matplotlib/tests/integration/hashes.json
```
