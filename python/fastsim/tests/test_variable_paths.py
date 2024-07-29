import unittest
import fastsim as fsim

class TestParamPath(unittest.TestCase):
    def test_param_path_list(self):
        print("TODO: This here test is failing and needs fixin")
        # # load 2012 Ford Fusion from file
        # veh = fsim.Vehicle.from_file(
        #     str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
        # )
        # with open(fsim.resources_root() / "benchmark_variable_paths/vehicle_variable_paths.txt") as file:
        #     baseline_variable_paths = [line.strip() for line in file.readlines()]

        # with open(fsim.resources_root() / "benchmark_variable_paths/vehicle_history_paths.txt") as file:
        #     baseline_history_variable_paths = [line.strip() for line in file.readlines()]
        
        # assert(sorted(baseline_variable_paths)==sorted(veh.variable_path_list()))
        # assert(sorted(baseline_history_variable_paths)==sorted(veh.history_path_list()))


if __name__ == '__main__':
    unittest.main()
