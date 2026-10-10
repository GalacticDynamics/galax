# Releasing galax

galax ships as several distributions from one repository. Each has its own
version, its own git tag and its own CD workflow, so any one can be released
alone — but the usual case is a coordinated release of all of them.

## Distributions

| distribution               | tag prefix                   | directory                           |
| -------------------------- | ---------------------------- | ----------------------------------- |
| `galax`                    | `v`                          | repository root                     |
| `galax.coordinates`        | `galax-coordinates-v`        | `packages/galax.coordinates`        |
| `galax.dynamics`           | `galax-dynamics-v`           | `packages/galax.dynamics`           |
| `galax.interop.astropy`    | `galax-interop-astropy-v`    | `packages/galax.interop.astropy`    |
| `galax.interop.gala`       | `galax-interop-gala-v`       | `packages/galax.interop.gala`       |
| `galax.interop.galpy`      | `galax-interop-galpy-v`      | `packages/galax.interop.galpy`      |
| `galax.interop.matplotlib` | `galax-interop-matplotlib-v` | `packages/galax.interop.matplotlib` |
| `galax.potential`          | `galax-potential-v`          | `packages/galax.potential`          |

## Release lines

Two branches release independently, and they do **not** share machinery:

| line     | branch            | distributions | how CD fires                  |
| -------- | ----------------- | ------------- | ----------------------------- |
| `v0.1.x` | `versions/v0.1.x` | one (`galax`) | publish a GitHub release      |
| `v0.2.x` | `main`            | eight         | the coordinated fan-out below |

`versions/v0.1.x` predates the namespace split: it carries only `cd.yml` and
`ci.yml`, with no per-package CD workflows and no `create-package-tags.yml`, and
this file does not exist on it. Everything below describes `main` only.

That branch's `cd.yml` triggers on `workflow_dispatch`, `pull_request`, `push`
to `main`, and `release: published` -- **not** on tags, and its `main` filter
never matches a commit on it. So a `v0.1.x` release is made by tagging and then
publishing a GitHub release, which fires `release: published` regardless of
which branch the tag points at. Pushing the tag alone does nothing.

## Coordinated release

1. Check `main` is green.
2. Push the coordinator tag: `git tag v0.2.0 && git push upstream v0.2.0`.
3. `create-package-tags.yml` creates each `<prefix>-v0.2.0` tag and pushes it,
   then dispatches each `cd-<prefix>.yml` with `gh workflow run`. Tags pushed
   with `GITHUB_TOKEN` do not trigger `on: push: tags:`, so the dispatch is what
   starts the builds.
4. Each dispatched run builds and inspects its package. Publishing is not yet
   enabled (see "What does not work yet").
5. Publish the GitHub release for the coordinator tag.

`workflow_dispatch` only works for a workflow file that exists on the default
branch, so a pre-merge dry run of the fan-out from a branch will not work.

## Single-package release

Push only that package's tag:

```sh
git tag galax-interop-gala-v0.2.1
git push upstream galax-interop-gala-v0.2.1
```

The remote must be the one pointing at `GalacticDynamics/galax`. In a direct
clone that is `origin`; in a fork-based checkout `origin` is the fork, and a
release tag pushed there builds nothing.

`validate_tag.py` refuses a tag that does not belong to the package being built,
so a mistyped prefix fails the build rather than publishing a wrong version.

## What does not work yet

- **The per-package CD workflows build; they do not publish.** Publishing needs
  sixteen trusted publishers (eight distributions, each on PyPI and TestPyPI),
  which only the repository owner can create. That is deliberately out of scope
  for the split itself.
- **The coordinator fan-out's dispatch step is untested.** Tags pushed with
  `GITHUB_TOKEN` do not trigger `on: push: tags:` workflows, so
  `create-package-tags.yml` dispatches each package's CD workflow explicitly
  with `gh workflow run`. That cannot be exercised without pushing a real tag.
  Do the first coordinated release with one package first and verify its CD run
  actually starts before relying on the fan-out for all of them.
- **Dispatched runs are validated only when dispatched on a tag ref.** The
  validation step is gated on `startsWith(github.ref, 'refs/tags/')`, so both a
  direct per-package tag push and the coordinator's
  `gh workflow run --ref <tag>` are validated; a manual dispatch on a branch ref
  is not.
- **The dispatch has no idempotency check**, so re-running it re-dispatches
  every package. Harmless while the workflows only build; add one when
  publishing is enabled.

## Trusted publishers

**Each distribution needs its own trusted publisher on both PyPI and TestPyPI,
keyed to its own workflow filename.** Sixteen configurations. A publisher
registered against the wrong filename fails with `invalid-publisher` at publish
time, after the tag exists — so verify one end to end on TestPyPI before tagging
the rest.

| distribution               | workflow filename                 |
| -------------------------- | --------------------------------- |
| `galax`                    | `cd.yml`                          |
| `galax.coordinates`        | `cd-galax-coordinates.yml`        |
| `galax.dynamics`           | `cd-galax-dynamics.yml`           |
| `galax.interop.astropy`    | `cd-galax-interop-astropy.yml`    |
| `galax.interop.gala`       | `cd-galax-interop-gala.yml`       |
| `galax.interop.galpy`      | `cd-galax-interop-galpy.yml`      |
| `galax.interop.matplotlib` | `cd-galax-interop-matplotlib.yml` |
| `galax.potential`          | `cd-galax-potential.yml`          |

## Migration notes for users

Changes that need to appear in the release notes, because nothing in the code
can signpost them:

- **A parameter function's return annotation must now record its dimension.**
  unxt v2 made `Quantity` non-parametric and moved the parametric class to the
  separate `unxts.parametric` distribution, so `u.Quantity["mass"]` no longer
  carries `"mass"` anywhere galax can read it. `ParameterField` used that
  annotation to check a user-supplied parameter function returns what the field
  declares, so:

  ```python
  # before
  def m_of_t(t: u.Quantity["time"]) -> u.Quantity["mass"]: ...


  # now
  from unxts.parametric import ParametricQuantity


  def m_of_t(t: ParametricQuantity["time"]) -> ParametricQuantity["mass"]: ...
  ```

  The old spelling raises `TypeError` naming the fix. This is deliberately a
  hard error rather than a deprecation: the only alternative was to accept the
  annotation and silently stop checking, which would let a parameter function
  return the wrong dimension unnoticed. A loud break at import of the first
  wrong annotation is cheaper than a quiet one at analysis time.

  Annotations that are not read for their dimension are unaffected -- ordinary
  return types like `-> u.Quantity["1/s^2"]` need no change, since nothing
  inspects them. Astropy-annotated parameter functions are also unaffected and
  still checked.

- **`galax.interop.optional_deps` was removed, and not replaced four times
  over.** Every interop distribution _requires_ the library it wraps, so an "is
  it installed" probe inside one is a constant `True`. Only
  `galax.interop.gala.optional_deps` survives, and it answers the two questions
  a dependency pin cannot:

  ```python
  from galax.interop.gala.optional_deps import GALA_VERSION, GSL_ENABLED
  ```

  `GSL_ENABLED` because gala builds optionally against GSL, and `GALA_VERSION`
  because some conversions are gated on gala 1.11. There is **no**
  `galax.interop.{astropy,galpy,matplotlib}.optional_deps` -- importing one
  raises `ModuleNotFoundError`. Code that only asked whether the library was
  installed can drop the check entirely.

  Note `OptDeps` is gone as a name: use `GALA_VERSION` directly rather than
  `OptDeps.GALA`.

  A signposting stub was deliberately not left at the old path either:
  `galax/interop/` is a namespace directory shared by five distributions and may
  hold no module of its own.

- **`galax[interop-astropy]` is now redundant** — `galax.interop.astropy` is a
  required dependency of `galax`. The extra is kept as a no-op alias.

- **`w.potential_energy(pot)` and `w.total_energy(pot)` moved to
  `galax.potential` as functions.** They take a coordinate _and_ a potential, so
  they belonged on neither object; and as methods on a `galax.coordinates` type
  they forced that distribution's documentation to import `galax.potential`,
  which the split makes a dependency inversion.

  ```python
  # before
  w.potential_energy(pot)
  w.total_energy(pot)

  # now
  gp.potential_energy(pot, w)
  gp.total_energy(pot, w)
  ```

  There is no method form: these are plum-dispatched functions, so a third-party
  coordinate type can extend them. `w.kinetic_energy()` is unchanged.

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
