import pytest
import fastsim


def test_pass_unmodified():
    veh = fastsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    cyc = fastsim.Cycle.from_resource("udds.csv")

    sd = fastsim.SimDrive(veh, cyc)
    sd.walk()


def test_error():
    veh = fastsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
    cyc = fastsim.Cycle.from_resource("udds.csv")

    veh_dict = veh.to_pydict()
    veh_dict["mass_kilograms"] += 1e6
    veh = fastsim.Vehicle.from_pydict(veh_dict)

    sd = fastsim.SimDrive(veh, cyc)
    pytest.raises(RuntimeError, sd.walk)


# def test_allow():
#     params = fastsim.SimParams.default()

#     veh = fastsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
#     cyc = fastsim.Cycle.from_resource("udds.csv")

#     veh_dict = veh.to_pydict()
#     veh_dict["mass_kilograms"] += 1e6
#     veh = fastsim.Vehicle.from_pydict(veh_dict)

#     sd = fastsim.SimDrive(veh, cyc, params)
#     pytest.raises(RuntimeError, sd.walk)


# def test_allow_checked(): ...


# def test_correct(): ...


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
