"""Package containing tests for FASTSim."""

from . import accel_test, test26veh3cyc, test26veh3cyc_CPUtime
# from . import test_vs_excel # this test does not work yet and is therefore not imported for now
# from . import test_time_dilation # not setup to be used as module so not imported

def run_all_working_tests(use_jitclass=True):
    """Runs all tests that are currently functioning as modules.
    Tests not run by this function are functioning as scripts --
    except test_vs_excel.py, which has known bugs."""

    print("Running test26veh3cyc.")
    test26veh3cyc.main(use_jitclass)

    print("\nRunning test26veh3cyc_CPUtime.")
    test26veh3cyc_CPUtime.main(use_jitclass)
    
    print("\nRunning accel_test.")
    accel_test.main()
