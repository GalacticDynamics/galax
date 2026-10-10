# matplotlib integration tests

These compare generated figures against committed baselines with the
`pytest-mpl` plugin, which also registers the `mpl_image_compare` marker the
tests are decorated with -- `--strict-markers` rejects it otherwise, so
collection needs the plugin whether or not `--mpl` is passed.

## Regenerating the baselines

```console
uv run pytest packages/galax.interop.matplotlib/tests/integration \
  --mpl-generate-path=packages/galax.interop.matplotlib/tests/integration/baseline
uv run pytest packages/galax.interop.matplotlib/tests/integration \
  --mpl-generate-hash-library=packages/galax.interop.matplotlib/tests/integration/hashes.json
```
