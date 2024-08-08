import unittest
import fastsim as fsim

class TestVariablePathList(unittest.TestCase):
    def test_variable_path_list(self):
        # load 2012 Ford Fusion from file
        veh = fsim.Vehicle.from_file(
            str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
        )
        with open(fsim.resources_root() / "benchmark_variable_paths/vehicle_variable_paths.txt") as file:
            baseline_variable_paths = [line.strip() for line in file.readlines()]

        with open(fsim.resources_root() / "benchmark_variable_paths/vehicle_history_paths.txt") as file:
            baseline_history_variable_paths = [line.strip() for line in file.readlines()]
        
        assert(set(baseline_variable_paths) == set(veh.variable_path_list()))
        assert(set(baseline_history_variable_paths) == set(veh.history_path_list()))


if __name__ == '__main__':
    unittest.main()
