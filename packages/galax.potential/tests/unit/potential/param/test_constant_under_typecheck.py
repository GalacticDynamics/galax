"""`ConstantParameter` returns what its hint says, with and without ``ustrip``."""

import subprocess
import sys
import textwrap

import pytest


def test_constant_parameter_call_under_runtime_type_checking() -> None:
    """Both returns of ``__call__`` pass jaxtyping's import hook with beartype.

    With ``ustrip`` the value is a plain array, not a quantity, so the hint must
    allow both.
    """
    pytest.importorskip("beartype")
    code = textwrap.dedent("""
        from jaxtyping import install_import_hook
        with install_import_hook("galax", "beartype.beartype"):
            import galax.potential as gp
        import unxt as u
        p = gp.params.ConstantParameter(value=u.Q(1.0, "Msun"))
        try:  # a bare float is not a time quantity, so only the hook rejects it
            p(1.0)
        except TypeError:
            pass
        else:
            raise AssertionError("the hook did not check the call")
        assert isinstance(p(u.Q(0, "Gyr")), u.AbstractQuantity)
        assert float(p(u.Q(0, "Gyr"), ustrip=u.unit("Msun"))) == 1.0
    """)
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
