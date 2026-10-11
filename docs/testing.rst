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
the interop extras:
::

    uv sync --extra all --group test-all

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

Putting those together, from a virtual environment on a Python that has a
linkable ``libpython``:
::

    pip install numpy setuptools wheel
    pip download --no-deps --no-binary :all: --no-build-isolation agama
    tar xzf agama-*.tar.gz && cd agama-*/
    export CPATH=<gsl-prefix>/include LIBRARY_PATH=<gsl-prefix>/lib
    yes N | python setup.py build

That leaves an importable ``agama.so`` in the build directory; put that
directory on ``PYTHONPATH``. Answering ``N`` declines the optional downloads
(Eigen and friends), which are not needed for potential comparisons.

Two things to get right when comparing against agama:

- **Units.** ``agama.setUnits(mass=1, length=1, velocity=1)`` does *not* give
  :math:`G = 1`; it means one solar mass, one kiloparsec and one km/s, so
  :math:`G \approx 4.3 \times 10^{-6}`. Omit ``setUnits`` entirely for N-body
  units with :math:`G = 1`. Getting this wrong scales every potential by
  ~2e5, which looks like a catastrophic disagreement rather than a units bug.
- **Precision.** :mod:`galax` defaults to float32 while agama is float64, so a
  comparison that does not enable x64 measures the dtype, not the algorithm.
  For shallow cusps the float32 floor is ~1e-7, which is one ulp.

``agama.Potential(type='Dehnen', ...)`` restricts :math:`\gamma` to
:math:`[0, 2]`. For steeper inner slopes use ``type='Spheroid'`` with
``alpha=1, beta=4``, which is the same profile with :math:`\gamma`
unrestricted.
