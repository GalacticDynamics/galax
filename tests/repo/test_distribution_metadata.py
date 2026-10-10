"""`pip install galax` installs the core portions and the astropy interop.

gala, galpy and matplotlib stay optional, behind extras.
"""

from importlib.metadata import PackageNotFoundError, metadata, requires

import pytest


def _required_names() -> set[str]:
    """Distribution names `galax` requires unconditionally."""
    return {
        r.split(";")[0].split(">")[0].split("[")[0].strip().lower().replace("_", "-")
        for r in (requires("galax") or [])
        if "extra ==" not in r
    }


def test_astropy_interop_is_required_not_an_extra() -> None:
    """Demoting it would silently stop astropy conversions registering.

    A missing entry point produces no error at all, so this must be pinned by
    metadata as well as by behaviour.
    """
    assert "galax-interop-astropy" in _required_names()


@pytest.mark.parametrize(
    "name", ["galax-interop-gala", "galax-interop-galpy", "galax-interop-matplotlib"]
)
def test_heavy_interop_stays_optional(name: str) -> None:
    """gala, galpy and matplotlib are extras, not required dependencies."""
    assert name not in _required_names()


@pytest.mark.parametrize(
    "extra",
    [
        "all",
        "interop-all",
        "interop-astropy",
        "interop-gala",
        "interop-galpy",
        "plot-all",
        "plot-matplotlib",
    ],
)
def test_every_published_extra_still_exists(extra: str) -> None:
    """Removing an extra breaks `pip install galax[...]` for existing users."""
    provided = metadata("galax").get_all("Provides-Extra") or []
    assert extra in provided


def test_coordinates_is_required_not_an_extra() -> None:
    """`pip install galax` must still bring the coordinates portion.

    Demoting it to an extra would leave `import galax.coordinates` failing on a
    default install, with nothing in the metadata to say why.
    """
    assert "galax-coordinates" in _required_names()


def test_dynamics_is_required_not_an_extra() -> None:
    """`pip install galax` must still bring the dynamics portion.

    Same reasoning as the two portions below: demoting it to an extra would
    leave `import galax.dynamics` failing on a default install, with nothing in
    the metadata to say why.
    """
    assert "galax-dynamics" in _required_names()


def test_no_distribution_depends_on_the_root() -> None:
    """Phase 2's end state, and the one claim nothing else checks.

    Every portion and interop must depend on the portions it uses, never on
    `galax` itself. A single `galax` requirement anywhere here recreates the
    `galax` <-> `galax.interop.astropy` cycle that phase 2 dissolved, and
    metadata is the only place that shows it -- the install shapes would still
    pass, because the root resolves fine.

    Skips a distribution that is not installed, so this holds on a bare install
    as well as under `--all-extras`.
    """
    offenders = {}
    for dist in (
        "galax.coordinates",
        "galax.potential",
        "galax.dynamics",
        "galax.interop.astropy",
        "galax.interop.gala",
        "galax.interop.galpy",
        "galax.interop.matplotlib",
    ):
        try:
            reqs = requires(dist) or []
        except PackageNotFoundError:
            continue
        named = {
            r.split(";")[0]
            .split(">")[0]
            .split("[")[0]
            .strip()
            .lower()
            .replace("_", "-")
            for r in reqs
        }
        if "galax" in named:
            offenders[dist] = sorted(n for n in named if n.startswith("galax"))
    assert not offenders, f"these depend on the root distribution: {offenders}"


def test_potential_is_required_not_an_extra() -> None:
    """`pip install galax` must still bring the potential portion.

    Same reasoning as the coordinates portion above: demoting it to an extra
    would leave `import galax.potential` failing on a default install, with
    nothing in the metadata to say why.
    """
    assert "galax-potential" in _required_names()
