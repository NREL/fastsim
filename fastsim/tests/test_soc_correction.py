"""
Tests an HEV correction methodology versus other techniques
"""
import unittest

import numpy as np

import fastsim


DO_PLOTS = False
VERBOSE = False


class TestSocCorrection(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_that_soc_correction_method_works(self):
        "Test using an SOC equivalency method versus other techniques"
        veh = fastsim.vehicle.Vehicle(11) # Toyota Highlander Hybrid
        cyc_names = ['udds', 'us06', 'hwfet']
        results = {
            'cycle': [],
            'init_soc': [],
            'delta_soc': [],
            'fuel_kJ': [],
            'soc_corrected_fuel_kJ': [],
            'equivalent_fuel_kJ': [],
            'percent_error': [],
        }
        for cyc_name in cyc_names:
            cyc = fastsim.cycle.Cycle(cyc_name)

            sd0 = fastsim.simdrive.SimDriveClassic(cyc, veh)
            sd0.sim_drive() # with SOC correction
            for initSoc in np.linspace(veh.minSoc, veh.maxSoc, 10, endpoint=True):
                sd = fastsim.simdrive.SimDriveClassic(cyc, veh)
                sd.sim_drive(initSoc)
                equivalent_fuel_kJ = fastsim.simdrive.estimate_corrected_fuel_kJ(sd)
                delta_soc = sd.soc[-1] - sd.soc[0]
                results['cycle'].append(cyc_name)
                results['init_soc'].append(initSoc)
                results['delta_soc'].append(delta_soc)
                results['fuel_kJ'].append(sd.fuelKj)
                results['equivalent_fuel_kJ'].append(equivalent_fuel_kJ)
                results['soc_corrected_fuel_kJ'].append(sd0.fuelKj)
                err_percent = ((equivalent_fuel_kJ - sd0.fuelKj) * 100.0) / sd0.fuelKj
                results['percent_error'].append(err_percent)
                self.assertTrue(
                    np.abs(err_percent) < 2.0,
                    msg=f'Not within 2% for soc0={initSoc} & cycle={cyc_name}; {err_percent} %'
                )
        total_absolute_percent_error = sum([abs(x) for x in results['percent_error']])
        average_absolute_percent_error = total_absolute_percent_error / len(results['percent_error'])
        if VERBOSE:
            print(f"Total Absolute Percent Error: {total_absolute_percent_error}")
            print(f"Average Absolute Percent Error: {average_absolute_percent_error}")
        self.assertTrue(average_absolute_percent_error < 1.0)
        if DO_PLOTS:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            results = pd.DataFrame(results)
            sns.set()
            for cyc_name in cyc_names:
                mask = results['cycle'] == cyc_name
                fig, ax = plt.subplots()
                base_fuel_kJ = np.unique(results['soc_corrected_fuel_kJ'][mask])[0]
                ax.plot([min(results['delta_soc'][mask]), max(results['delta_soc'][mask])], [base_fuel_kJ] * 2, label='corrected')
                ax.plot(results['delta_soc'][mask], results['fuel_kJ'][mask] + results['equivalent_fuel_kJ'][mask], 'bo', label='estimate')
                ax.plot(results['delta_soc'][mask], results['fuel_kJ'][mask], 'k.', label='actual')
                ax.legend()
                ax.set_xlabel('ΔSOC')
                ax.set_ylabel('Fuel (kJ)')
                ax.set_title(f'Fuel Consumed by ΔSOC over {cyc_name}')
                fig.tight_layout()
                fig.savefig(f'test_that_soc_correction_method_works_{cyc_name}.png', dpi=300)
                plt.close()
                fig = None

                fig, ax = plt.subplots()
                ax.plot([min(results['delta_soc'][mask]), max(results['delta_soc'][mask])], [0.0] * 2, label='corrected')
                pct_err = ((results['fuel_kJ'][mask] + results['equivalent_fuel_kJ'][mask]) - base_fuel_kJ) * 100.0 / base_fuel_kJ
                ax.plot(results['delta_soc'][mask], pct_err, 'bo', label='estimate')
                pct_err = (results['fuel_kJ'][mask] - base_fuel_kJ) * 100.0 / base_fuel_kJ
                ax.plot(results['delta_soc'][mask], pct_err, 'k.', label='actual')
                ax.plot([min(results['delta_soc'][mask]), max(results['delta_soc'][mask])], [1.0]*2, 'r:', label='upper bound')
                ax.plot([min(results['delta_soc'][mask]), max(results['delta_soc'][mask])], [-1.0]*2, 'r:', label='lower bound')
                ax.legend()
                ax.set_xlabel('ΔSOC')
                ax.set_ylabel('Fuel Consumption Error (%)')
                ax.set_title(f'Fuel Consumption Error by ΔSOC over {cyc_name}')
                ax.set_ylim(-8, 8)
                fig.tight_layout()
                fig.savefig(f'test_that_soc_correction_method_works_pct_error_{cyc_name}.png', dpi=300)
                plt.close()
                fig = None