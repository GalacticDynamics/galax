.. include:: references.txt

*****************
How to contribute
*****************

We welcome contributions from anyone via pull requests on `GitHub
<https://github.com/GalacticDynamics/galax>`_. If you don't feel comfortable modifying or
adding functionality, we also welcome feature requests and bug reports as
`GitHub issues <https://github.com/GalacticDynamics/galax/issues>`_.

Developer documentation
=======================

.. toctree::
    :maxdepth: 1

    testing
    docs

The code graph
==============

``graphify-out/graph.json`` is a `graphify <https://github.com/Graphify-Labs/graphify>`_ map of
this repository: its symbols and the relationships between them. AI assistants query it instead of
grepping, and you can too. Extraction is local tree-sitter parsing -- no model call, and nothing
leaves your machine -- and every edge is marked ``EXTRACTED`` (explicit in the source) or
``INFERRED`` (resolved by graphify), so you can tell what was read from what was deduced.

``.graphifyignore`` fixes what is indexed: the ``src/`` trees of all eleven distributions, and
nothing else. Excluding ``tests/`` halved the file, for the one part of the tree whose layout is
already predictable -- ``packages/<dist>/tests`` mirrors ``packages/<dist>/src``. The benchmark
harness, the plot scripts, the Sphinx config, ``noxfile.py``, ``conftest.py`` and every
``pyproject.toml`` are out too: none is library code, and excluding the TOML also stops a
dependency bump from churning the graph. Grep those; query the graph for everything under
``packages/*/src``.

The graph is committed, and the commit checks rebuild it. Install the pinned release and register
the merge driver once per clone, so two branches that both changed the graph merge cleanly:

.. code-block:: bash

    uv tool install "graphifyy==0.9.119"
    PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
    git config merge.graphify.name "graphify graph.json union merge"
    git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"

Git refuses to take a merge-driver command from a tracked file, so this cannot be committed for
you -- it is one command per clone.

Then query it:

.. code-block:: bash

    graphify query "how are potential gradients dispatched"   # a scoped subgraph for a question
    graphify explain "AbstractPotential"                      # one symbol and its neighbours
    graphify path "Orbit" "AbstractPotential"                 # how two things connect
    open graphify-out/graph.html                              # the clickable graph

Day to day:

- The rebuild runs with your other commit checks. Like a formatter, it rebuilds and stops the
  commit; ``git add graphify-out/graph.json`` and commit again.
- Commit without rebuilding: ``SKIP=graphify-update git commit ...``.
- If a rebuild is ever refused because the graph would shrink -- you deleted code -- use
  ``GRAPHIFY_FORCE=1 git commit ...``. It usually just shrinks and says nothing.
- If the graph ever conflicts in a merge, rebuild rather than edit it:
  ``rm graphify-out/graph.json && graphify update . && git add graphify-out/graph.json``.
- A changed ``graph.json`` in a pull request is generated output. It is one line, collapsed by
  GitHub, and is not reviewed.
- The rebuild may rewrite the graph the first time you commit in a working environment, and the
  weekly job may rewrite it back. That is expected: graphify's output depends on which files are
  on disk, so a built checkout and a fresh clone disagree about community numbering even though
  the symbols are identical. ``galax`` is on hatch-vcs, so an installed checkout always has a
  generated ``_version.py`` and always disagrees.
