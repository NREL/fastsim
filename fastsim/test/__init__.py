"""Package containing tests for FASTSim.
Run the tests either by importing the package or with 
`python -m unittest discover` from within fastsim."""

from . import test_simdrive_sweep, test_vs_excel
# from . import test_vs_excel # this test does not work yet and is therefore not imported for now
# from . import test_time_dilation # not setup to be used as module so not imported

def run_functional_tests(use_jitclass=True):
    """Runs all tests that are currently functioning as modules.
    Tests not run by this function are functioning as scripts --
    except test_vs_excel.py, which has known bugs."""

    print("Running test26veh3cyc.")
    test_simdrive_sweep.main(use_jitclass)
    
    print("\nRunning comparison with Excel FASTSim.")
    test_vs_excel.main(use_jitclass)
