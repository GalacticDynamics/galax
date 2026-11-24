"""`TimeDependentTranslationParameter` returns what its hint says."""

import subprocess
import sys
import textwrap

import pytest


def test_translation_call_under_runtime_type_checking() -> None:
    """Both returns of ``__call__`` pass jaxtyping's import hook with beartype.

    With ``ustrip`` the value is a plain array, not a quantity, so the hint must
    allow both.
    """
    pytest.importorskip("beartype")
    code = textwrap.dedent("""
        from jaxtyping import install_import_hook
        with install_import_hook("galax", "beartype.beartype"):
            import galax.potential as gp
        import jax.numpy as jnp
        import unxt as u
        t = u.Q(jnp.linspace(0, 1, 50), "Gyr")
        x = u.Q(jnp.stack([jnp.linspace(0, 1, 50)] * 3, -1), "kpc")
        d = gp.params.TimeDependentTranslationParameter.from_(
            t, x, units=u.unitsystem("galactic")
        )
        try:  # a bare float is not a time array, so only the hook rejects it
            d(1.0)
        except TypeError:
            pass
        else:
            raise AssertionError("the hook did not check the call")
        assert isinstance(d(u.Q(100.0, "Myr")), u.AbstractQuantity)
        assert d(u.Q(100.0, "Myr"), ustrip=u.unit("kpc")).shape == (3,)
    """)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
