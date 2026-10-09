# Arraydiff reference data

`test_gradient`/`test_hessian`/`test_tidal_tensor` in the `test_*.py` files in
the parent directory are decorated with `@pytest.mark.array_compare` (from
`pytest-arraydiff`) instead of hardcoding an expected literal + `jnp.allclose`
assertion. Each test returns the computed (unit-stripped) array, and the plugin
diffs it against `<slug>/<test name>.txt` here, where `<slug>` is the
potential's name (its class name, for the couple of files with more than one
test class).

Values round-trip through these files at 6 significant figures (the default `%g`
text format), which is coarser than the `atol=1e-8` these replaced -- fine for a
regression check, not for asserting exact physics. `atol=1e-8` is passed
explicitly on the marker (rather than the plugin's default of `0`) so potentials
with near-zero elements (e.g. `test_null.py`) still compare correctly.

## Regenerating a reference file

Run the target test(s) with `--arraydiff-generate-path` pointed at the resolved
directory (matching the test's `reference_dir=` kwarg), e.g.:

```console
uv run pytest tests/unit/potential/builtin/test_gnfw.py \
  -k "test_gradient or test_hessian or test_tidal_tensor" \
  --arraydiff-generate-path=tests/unit/potential/builtin/reference/gnfw \
  -o filterwarnings=
```

`-o filterwarnings=` clears this repo's `filterwarnings = ["error"]` for the one
invocation: `pytest-arraydiff` calls `pytest.skip()` from inside an old-style
hookwrapper to abort the test after writing the file, which pytest turns into a
`PluggyTeardownRaisedWarning` -- normally harmless, but fatal here since
warnings are errors. Regular test runs (`--arraydiff`, no generate-path) aren't
affected.

Note that `--arraydiff-generate-path` writes every generated file as
`<path>/<item.name>.txt`, ignoring `reference_dir=` entirely -- so generating
more than one potential's files in a single invocation makes same-named tests
(`test_gradient` in every file) collide. Generate one potential (one
`reference_dir`) at a time.
