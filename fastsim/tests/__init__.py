"""Package containing tests for FASTSim."""

from . import test_simdrive_sweep, test_vs_excel, test_vehicle, test_cycle, test_simdrive, test_rust

def run_functional_tests():
    """
    Runs all functional tests.
    """

    print("Running test26veh3cyc.")
    test_simdrive_sweep.main()

    print("\nTesting for discrepancies between Rust and Python.")
    test_rust.TestRust().test_discrepancies()

    print("\nRunning comparison with Excel FASTSim.")
    test_vs_excel.main()
