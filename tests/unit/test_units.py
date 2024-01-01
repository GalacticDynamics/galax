# Standard library
import pickle

# Third party
import astropy.units as u
import numpy as np
import pytest

# This package
from galax.units import UnitSystem, dimensionless


def test_init():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

    with pytest.raises(
        ValueError, match="must specify a unit for the physical type .*mass"
    ):
        UnitSystem(u.kpc, u.Myr, u.radian)  # no mass

    with pytest.raises(
        ValueError, match="must specify a unit for the physical type .*angle"
    ):
        UnitSystem(u.kpc, u.Myr, u.Msun)

    with pytest.raises(
        ValueError, match="must specify a unit for the physical type .*time"
    ):
        UnitSystem(u.kpc, u.radian, u.Msun)

    with pytest.raises(
        ValueError, match="must specify a unit for the physical type .*length"
    ):
        UnitSystem(u.Myr, u.radian, u.Msun)

    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)
    usys = UnitSystem(usys)


def test_quantity_init():
    usys = UnitSystem(5 * u.kpc, 50 * u.Myr, 1e5 * u.Msun, u.rad)
    assert np.isclose((8 * u.Myr).decompose(usys).value, 8 / 50)


def test_preferred():
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.km / u.s)
    q = 15.0 * u.km / u.s
    assert usys.preferred("velocity") == u.km / u.s
    assert q.decompose(usys).unit == u.kpc / u.Myr
    assert usys.as_preferred(q).unit == u.km / u.s


def test_dimensionless():
    assert dimensionless["dimensionless"] == u.one
    assert dimensionless["length"] == u.one

    with pytest.raises(ValueError, match="can not be decomposed into"):
        (15 * u.kpc).decompose(dimensionless)

    with pytest.raises(ValueError, match="are not convertible"):
        dimensionless.as_preferred(15 * u.kpc)


def test_compare():
    usys1 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)
    usys1_clone = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.mas / u.yr)

    usys2 = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun, u.kiloarcsecond / u.yr)
    usys3 = UnitSystem(u.kpc, u.Myr, u.radian, u.kg, u.mas / u.yr)

    assert usys1 == usys1_clone
    assert usys1_clone == usys1

    assert usys1 != usys2
    assert usys2 != usys1

    assert usys1 != usys3
    assert usys3 != usys1


def test_pickle(tmpdir):
    usys = UnitSystem(u.kpc, u.Myr, u.radian, u.Msun)

    path = tmpdir / "test.pkl"
    with path.open(mode="wb") as f:
        pickle.dump(usys, f)

    with path.open(mode="rb") as f:
        usys2 = pickle.load(f)

    assert usys == usys2
