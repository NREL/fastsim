"""
Test fastsim/docs/cav_sweep.py for regressions
"""
import unittest
import os
import csv
from pathlib import Path

from fastsim.docs.cav_sweep import main, CSV_KEYS
import fastsim as fsim

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REGRESSION_DATA = Path(THIS_DIR) / 'test_cav_sweep.csv'
def env_to_bool(var):
    if var is None:
        return False
    return var.strip().lower() == 'true'
FASTSIM_TEST_EXTENSIVE = env_to_bool(os.getenv('FASTSIM_TEST_EXTENSIVE'))

class TestCavSweep(unittest.TestCase):
    def setUp(self):
        fsim.utils.disable_logging()
    
    def load_regression_data(self):
        if REGRESSION_DATA.exists():
            data = []
            with open(REGRESSION_DATA, newline='') as csvfile:
                reader = csv.reader(csvfile)
                keys = None
                for row_num, row in enumerate(reader):
                    if row_num == 0:
                        keys = row
                        continue
                    row_data = {}
                    for k, v in zip(keys, row):
                        try:
                            row_data[k] = float(v)
                        except:
                            row_data[k] = v
                    data.append(row_data)
            return data
        return None
    
    def save_regression_data(self, known_good_data):
        keys = CSV_KEYS
        with open(REGRESSION_DATA, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(keys)
            for item in known_good_data:
                writer.writerow([str(item[k]) for k in keys])

    def compare_one_case(self, known_good_case, current_case, use_rust):
        env = "Rust " if use_rust else "Py "
        known_good_key_set = set(known_good_case.keys())
        case_key_set = set(current_case.keys())
        self.assertTrue(
            known_good_key_set == case_key_set,
            msg=(env
                + f"Key sets differ! Expected {known_good_key_set}; "
                + f"got {case_key_set}; "
                + f"extra: {case_key_set - known_good_key_set}; "
                + f"missing: {known_good_key_set - case_key_set}"))
        msg = f"{env} Regression for {known_good_case['veh']}:{known_good_case['powertrain']}:{known_good_case['cycle']}"
        for k in CSV_KEYS:
            new_msg = msg + f" for key '{k}': known-good: {known_good_case[k]}; current: {current_case[k]}"
            if ':' in k:
                self.assertAlmostEqual(known_good_case[k], current_case[k], places=1, msg=new_msg)
            else:
                self.assertEqual(known_good_case[k], current_case[k], msg=new_msg)
    
    def _make_key(self, the_case):
        return f"{the_case['powertrain']}:{the_case['cycle']}:{the_case['veh']}"
    
    def _env_as_str(self, use_rust):
        return 'Rust' if use_rust else 'Py'

    def compare_for_regressions(self, known_good_data, outputs, use_rust):
        if FASTSIM_TEST_EXTENSIVE:
            self.assertTrue(len(known_good_data) == len(outputs), f"{self._env_as_str(use_rust)} Expected {len(known_good_data)} cases; got {len(outputs)}")
        else:
            self.assertTrue(len(outputs) <= len(known_good_data), f"{self._env_as_str(use_rust)} Expected {len(outputs)} to be <= {len(known_good_data)}")
        for out in outputs:
            out_key = self._make_key(out)
            found_key = False
            for kg in known_good_data:
                kg_key = self._make_key(kg)
                if out_key == kg_key:
                    found_key = True
                    self.compare_one_case(kg, out, use_rust)
            self.assertTrue(found_key, msg=f"{self._env_as_str(use_rust)} Could not find key {out_key} in known good data")

    def test_demo_for_regressions(self):
        known_good_data = self.load_regression_data()
        if known_good_data is None:
            self.save_regression_data(main(do_show=False, use_rust=False, verbose=False))
            known_good_data = self.load_regression_data()
        for use_rust in [False, True]:
            if FASTSIM_TEST_EXTENSIVE:
                outputs = main(do_show=False, use_rust=use_rust, verbose=False)
            else:
                outputs = main(
                    powertrain='hev',
                    cycle_name="TSDC_tripno_42648_cycle",
                    do_show=False,
                    use_rust=use_rust,
                    verbose=False)
            self.compare_for_regressions(known_good_data, outputs, use_rust)

if __name__ == '__main__':
    unittest.main()
