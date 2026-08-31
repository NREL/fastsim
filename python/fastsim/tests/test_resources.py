"""Test all resources load successfully."""

import fastsim


def test_list_resources_for_cycle():
    """Assert list_resources works for Cycle and each resource can be loaded."""
    resources = fastsim.Cycle.list_resources()
    assert len(resources) > 0
    for name in resources:
        fastsim.Cycle.from_resource(name)


def test_list_resources_for_vehicle():
    """Assert list_resources works for Vehicle and each resource can be loaded."""
    resources = fastsim.Vehicle.list_resources()
    assert len(resources) > 0
    for name in resources:
        fastsim.Vehicle.from_resource(name)
