"""Test from_db function works."""

import os

import pytest

import fastsim


@pytest.mark.skipif(
    os.getenv("FASTSIM_RUN_NETWORK_TESTS") != "1",
    reason="makes live network request; set FASTSIM_RUN_NETWORK_TESTS=1 to run",
)
def test_from_db():
    """Assert from_db works for Vehicle."""
    vehicle_path = "v1/fastsim-3/conv/ford/fusion/2012/base/r1"
    fastsim.Vehicle.from_db(path=vehicle_path)
