"""Release JAX's compiled-executable cache between modules here.

Every distinct combination of `build_expansion`'s static arguments -- ``n_r``,
``l_max``, ``n_theta``, ``n_phi``, the mode keys -- compiles its own
executable, and JAX keeps them for the life of the process. These modules
build expansions across many such combinations, so the cache grows
monotonically and is never reclaimed.

That is invisible under ``-n logical``, where xdist spreads the work over
four processes, and fatal without it: the ``Check Interoperability`` job runs
the whole suite serially in one process with every optional dependency
installed, and it died three times at 97-99% with
``The runner has received a shutdown signal`` and no failing test -- the
signature of the runner being reclaimed, which an out-of-memory kill also
produces. The same tests pass in the parallel jobs.

Measured over this package plus the functional multipole-profile
tests:

===================== ========== ==========
clearing               peak RSS   wall
===================== ========== ==========
none                   3.31 GB    130 s
per module             2.38 GB    142 s
per test               1.48 GB    184 s
===================== ========== ==========

Per-module is the trade taken: most of the reduction for 9% more wall time,
where per-test costs 41% and would lengthen the very job that is failing.
"""

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    yield
    jax.clear_caches()
