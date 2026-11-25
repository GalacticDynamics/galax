# Agent guidance for galax

galax is a monorepo: eleven distributions under `packages/`, each with its own
`src/galax/...` and `tests/`. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for
how to set up a development environment, run the tests and build the docs.

## Code graph

This repository has a committed graphify knowledge graph at
`graphify-out/graph.json`: every symbol, the links between them, the
architectural hubs, and which subsystem each symbol belongs to. It is built from
the source by tree-sitter, so it is as current as the last commit.

Querying it is cheaper and more accurate than grepping. The rules:

- For any question about this codebase, query the graph **before** your first
  grep, glob or source-file read:

  - `graphify query "<question>"` — a scoped subgraph for a plain-language
    question
  - `graphify explain "<symbol>"` — one symbol and its neighbours
  - `graphify path "<A>" "<B>"` — how two things connect
  - `graphify-out/GRAPH_REPORT.md` — hubs and subsystems, for broad architecture
    review only

  Read source files for the lines a query has already located, not to find them.

- Put this rule in every subagent prompt you write that explores the code. A
  subagent that greps from scratch undoes the point of the graph.
- The graph covers the library source and nothing else. `.graphifyignore`
  excludes `tests/`, because it was 44% of the graph for the one part of the
  tree whose layout is already predictable (`packages/<dist>/tests` mirrors
  `packages/<dist>/src`); and `benchmarks/`, `scripts/`, `docs/`, `noxfile.py`,
  `conftest.py` and every `*.toml`, because none of them is library code. Grep
  or glob for those; query the graph for everything under `packages/*/src`.
- After changing code, refresh it: `PYTHONHASHSEED=0 graphify update . --force`
  — tree-sitter only, no model call, no network. `--force` keeps your rebuild
  equal to the one the commit hook and CI produce; without it the rebuild is
  incremental and drifts.
- Ignore `built_at_commit` in the graph, and `GRAPH_REPORT.md`'s "built from
  commit … compare to check if the graph is stale". `update` does not rewrite
  that field, so it names an old commit for ever and is not a staleness signal.
  Do not rebuild on the strength of it.
- Use the pinned release, the one in `.pre-commit-config.yaml`, so your rebuild
  clusters the same way the hook does:
  `uvx --from graphifyy==0.9.119 graphify ...`.
- `graphify-out/graph.json` is generated. Do not read it directly, do not review
  it in a diff, and do not hand-resolve a conflict in it: rebuild it instead.
- A rebuild that rewrites the graph in this checkout is **expected** if the
  package has been installed or the tests have been run here: graphify's output
  depends on which files are on disk, and a generated `_version.py` is enough to
  move community numbering. Stage it and commit; it is not a bug and not a
  reason to skip the hook.
- Skip the graph only when the task is about the graph tooling itself being
  stale or wrong, or when asked to. An uncommitted or just-rebuilt graph file is
  normal and is not a reason to skip it.
