"""Package containing tests for FASTSim."""


def run_all_working_tests(use_jitclass=True):
    """Runs all tests that are currently functioning as modules.
    Tests not run by this function are functioning as scripts --
    except test_vs_excel.py, which has known bugs."""
    from fastsim.tests import test26veh3cyc, test26veh3cyc_CPUtime, accel_test

    print("Running test26veh3cyc.")
    test26veh3cyc.main(use_jitclass)

    print("\nRunning test26veh3cyc_CPUtime.")
    test26veh3cyc_CPUtime.main(use_jitclass)
    
    print("\nRunning accel_test.")
    accel_test.main()
