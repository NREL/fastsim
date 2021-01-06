"""Package containing tests for FASTSim."""

def run_all_working_tests():
    """Runs all tests that are currently functioning as modules.
    Tests not run by this function are functioning as scripts --
    except test_vs_excel.py, which has known bugs."""
    from fastsim.tests import test26veh3cyc, test26veh3cyc_CPUtime, accel_test

    test26veh3cyc.run_test26veh3cyc()
    test26veh3cyc_CPUtime.run_test26veh3cyc_CPUtime()
    accel_test.main()