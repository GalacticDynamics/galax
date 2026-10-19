.. _galax-test:

=================
Running the tests
=================

The tests are written assuming they will be run with `tox
<https://tox.readthedocs.io/en/latest/>`_ or `pytest <http://doc.pytest.org/>`_.

To run the tests with tox, first make sure that tox is installed;
::

    pip install tox

then run the basic test suite with:
::

    tox -e test

or run the test suite with all optional dependencies with:
::

    tox -e test-alldeps

You can see a list of available test environments with:
::

    tox -l -v

which will also explain what each of them does.

You can also run the tests directly with pytest. To do this, make sure to
install the testing requirements (from the cloned :mod:`galax` repository
directory).
::

    pip install -e ".[test]"

Then you can run the tests with:
::

    pytest galax

.. _galax-comparison-libs:

Comparison libraries
====================

Parts of the suite cross-check :mod:`galax` against other dynamics packages.
``gala`` and ``galpy`` are declared optional dependencies, so they arrive with
the ``interop-all`` extra:
::

    uv sync --extra interop-all --group test-all

(``--extra all`` works too, and additionally pulls the plotting extras.)

Tests needing them are skipped when they are absent, so a bare checkout still
passes -- a green run does **not** mean they were exercised.

``agama`` is deliberately *not* in any dependency list and has to be built by
hand. It is useful for accuracy work on the multipole expansion, where it is
the natural reference. Four things make the usual ``pip install agama`` fail:

#. Its build imports ``numpy`` at setup time, which build isolation hides.
   Install ``numpy``, ``setuptools`` and ``wheel`` first and pass
   ``--no-build-isolation``.
#. It links against ``libpython``. Standalone interpreters (uv's, for
   instance) often ship no linkable library, and the build dies with
   ``symbol(s) not found`` naming ``_Py_Initialize``. Its own error message
   blames an x86_64 interpreter on Apple silicon, which is usually wrong --
   check ``platform.machine()`` before believing it. A Homebrew Python has a
   linkable ``libpython``.
#. It requires GSL. If GSL came from Homebrew, the compiler probe will not
   find it on its own; put the Homebrew prefix on ``CPATH`` and
   ``LIBRARY_PATH``.
#. Its ``setup.py`` asks interactive yes/no questions when a library is
   missing. ``pip`` gives it no stdin, so it fails with
   ``ValueError: No valid response (y/n) was provided`` rather than saying
   what it wanted. Build from the unpacked sdist instead, where answers can be
   piped in.

One more does not fail the build but aborts the test run: agama links
Homebrew's OpenMP runtime, and galpy's wheel bundles its own. With both
loaded the process dies with ``OMP: Error #15: Initializing libomp.dylib,
but found libomp.dylib already initialized``. Build agama without OpenMP --
the comparisons do not need threads -- by hiding ``brew`` from ``PATH`` so
its setup cannot find ``libomp``, and answer ``Y`` to "continue without
OpenMP".

Putting those together, from a virtual environment on a Python that has a
linkable ``libpython`` (on macOS, a Homebrew Python):
::

    pip install numpy setuptools wheel
    pip download --no-deps --no-binary :all: --no-build-isolation agama
    tar xzf agama-*.tar.gz && cd agama-*/
    printf 'Y\nN\nN\nN\n' | env PATH=/usr/bin:/bin \
        CPATH=<gsl-prefix>/include LIBRARY_PATH=<gsl-prefix>/lib \
        "$VIRTUAL_ENV/bin/python" setup.py build

That leaves an importable ``agama`` package under ``build/lib.*``; put that
directory on ``PYTHONPATH``. The ``Y`` continues without OpenMP and the
``N``\ s decline the optional downloads (Eigen, CVXOPT, UNSIO), which are not
needed for potential comparisons.

The agama module is linked against the libpython it was built with, so run
the tests from an environment on that same interpreter. With uv, a second
environment outside the checkout keeps the project's own untouched:
::

    UV_PROJECT_ENVIRONMENT=<agama venv> uv sync --locked \
        --python <that python> --extra interop-all --group test-all

Running the agama tests
-----------------------

The agama comparisons carry the ``agama`` marker and are skipped unless
``--run-agama`` is passed -- locally and in CI, whatever ``-m`` says:
::

    PYTHONPATH=<agama build dir> UV_PROJECT_ENVIRONMENT=<agama venv> \
        uv run --no-sync pytest --run-agama -m agama packages/galax.potential -ra

With the flag they still skip, with the reason ``requires agama``, when agama
is not importable, so check the ``-ra`` summary: a green run where they all
skipped compared nothing. The units and precision points below are handled
by the tests' shared fixture.

Three things to get right when comparing against agama:

- **Units.** ``agama.setUnits(mass=1, length=1, velocity=1)`` does *not* give
  :math:`G = 1`; it means one solar mass, one kiloparsec and one km/s, so
  :math:`G \approx 4.3 \times 10^{-6}`. Omit ``setUnits`` entirely for N-body
  units with :math:`G = 1`. Getting this wrong scales every potential by
  ~2e5, which looks like a catastrophic disagreement rather than a units bug.
- **Precision.** :mod:`galax` defaults to float32 while agama is float64, so a
  comparison that does not enable x64 measures the dtype, not the algorithm.
  For shallow cusps the float32 floor is ~1e-7, which is one ulp.
- **G.** agama's gravitational constant exceeds astropy's (CODATA 2018) by
  1.49e-10 relative, so closed forms agree to ~1.5e-10, not to machine
  precision. A constant 1.49e-10 offset is this, not a formula difference.

``agama.Potential(type='Dehnen', ...)`` restricts :math:`\gamma` to
:math:`[0, 2]`. For steeper inner slopes use ``type='Spheroid'`` with
``alpha=1, beta=4``, which is the same profile with :math:`\gamma`
unrestricted.
