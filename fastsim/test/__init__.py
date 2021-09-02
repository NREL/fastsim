"""Package containing tests for FASTSim."""

from . import test_simdrive_sweep, test_vs_excel

def run_functional_tests(use_jitclass=True):
    """
    Runs all functional tests.
    """

    print("Running test26veh3cyc.")
    test_simdrive_sweep.main(use_jitclass)
    
    print("\nRunning comparison with Excel FASTSim.")
    test_vs_excel.main(use_jitclass)
