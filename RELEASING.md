# Releasing galax

galax ships as several distributions from one repository. Each has its own
version, its own git tag and its own CD workflow, so any one can be released
alone — but the usual case is a coordinated release of all of them.

## Distributions

| distribution               | tag prefix                   | directory                           |
| -------------------------- | ---------------------------- | ----------------------------------- |
| `galax`                    | `v`                          | repository root                     |
| `galax.interop.astropy`    | `galax-interop-astropy-v`    | `packages/galax.interop.astropy`    |
| `galax.interop.gala`       | `galax-interop-gala-v`       | `packages/galax.interop.gala`       |
| `galax.interop.galpy`      | `galax-interop-galpy-v`      | `packages/galax.interop.galpy`      |
| `galax.interop.matplotlib` | `galax-interop-matplotlib-v` | `packages/galax.interop.matplotlib` |

## Coordinated release

1. Check `main` is green.
2. Push the coordinator tag: `git tag v0.1.0 && git push origin v0.1.0`.
3. `create-package-tags.yml` creates each `<prefix>-v0.1.0` tag and pushes it.
4. Each package's CD workflow builds and publishes on its own tag.
5. Publish the GitHub release for the coordinator tag.

## Single-package release

Push only that package's tag:

```sh
git tag galax-interop-gala-v0.1.1
git push origin galax-interop-gala-v0.1.1
```

`validate_tag.py` refuses a tag that does not belong to the package being built,
so a mistyped prefix fails the build rather than publishing a wrong version.

## What does not work yet

- **The per-package CD workflows build; they do not publish.** Publishing needs
  ten trusted publishers (five distributions, each on PyPI and TestPyPI), which
  only the repository owner can create. That is deliberately out of scope for
  the split itself.
- **The coordinator fan-out's dispatch step is untested.** Tags pushed with
  `GITHUB_TOKEN` do not trigger `on: push: tags:` workflows, so
  `create-package-tags.yml` dispatches each package's CD workflow explicitly
  with `gh workflow run`. That cannot be exercised without pushing a real tag.
  Do the first coordinated release with one package first and verify its CD run
  actually starts before relying on the fan-out for all four.
- **A dispatched run skips `validate_tag.py`.** The validation step is gated on
  `github.event_name == 'push'`, so only a direct per-package tag push is
  validated; the fan-out path is not.
- **The dispatch has no idempotency check**, so re-running it re-dispatches
  every package. Harmless while the workflows only build; add one when
  publishing is enabled.

## Trusted publishers

**Each distribution needs its own trusted publisher on both PyPI and TestPyPI,
keyed to its own workflow filename.** Ten configurations. A publisher registered
against the wrong filename fails with `invalid-publisher` at publish time, after
the tag exists — so verify one end to end on TestPyPI before tagging the rest.

| distribution               | workflow filename                 |
| -------------------------- | --------------------------------- |
| `galax`                    | `cd.yml`                          |
| `galax.interop.astropy`    | `cd-galax-interop-astropy.yml`    |
| `galax.interop.gala`       | `cd-galax-interop-gala.yml`       |
| `galax.interop.galpy`      | `cd-galax-interop-galpy.yml`      |
| `galax.interop.matplotlib` | `cd-galax-interop-matplotlib.yml` |

## Migration notes for users

Changes that need to appear in the release notes, because nothing in the code
can signpost them:

- **`galax.interop.optional_deps` was removed.** Each interop distribution now
  declares its own check: `galax.interop.astropy.optional_deps`,
  `galax.interop.gala.optional_deps` (also exports `GSL_ENABLED`),
  `galax.interop.galpy.optional_deps`, `galax.interop.matplotlib.optional_deps`.
  The `OptDeps` name and its member names are unchanged, so only the import
  moves. A signposting stub was deliberately not left behind: `galax/interop/`
  is a namespace directory shared by five distributions and may hold no module
  of its own.
- **`galax[interop-astropy]` is now redundant** — `galax.interop.astropy` is a
  required dependency of `galax`. The extra is kept as a no-op alias.

## Version floors

Distributions pin each other with floors only (`galax>0.0.3`), no upper bounds.

The floor is deliberately the latest _released_ galax, not the version being
prepared: a floor naming an unreleased version cannot be resolved locally, so
the install-shape checks could not run before the release they are meant to
gate. **Raise every floor to the new version as part of the first coordinated
release**, and bump again when a package starts relying on something newly
added.

**These floors are only correct while the first post-split root release
genuinely ships _without_ `galax/interop/`.** The floor `galax>0.0.3` rests on
the claim that every release above 0.0.3 has dropped the in-tree interop code;
if a root release still carried `galax/interop/`, it would overlap files with
the interop distributions and the floor would permit a broken install. Verify
this at release time: build the root wheel and confirm it contains no
`galax/interop/` entries before tagging.

## Adding a distribution

Tag globs are not checked against each other automatically. The root
distribution versions from tags matching `v[0-9]*` (see
`version.raw-options.scm.git.describe_command` in `pyproject.toml`), and the
coordinator workflow fans out on the same form. Per-package tags start with
their own prefix (`galax-interop-gala-v`), so today none can match the root's
form, and nothing verifies that stays true. **When adding a distribution, check
that its tag glob cannot match the root's `v[0-9]*` form**, and that no existing
package's glob is a prefix of the new one. Add it to `validate_tag.py`'s
callers, to the tables above, and give it its own trusted publisher and CD
workflow.
