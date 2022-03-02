"""
Tests an HEV correction methodology versus other techniques
"""
import unittest

import numpy as np

import fastsim


DO_PLOTS = False


class TestSocCorrection(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_that_soc_correction_method_works(self):
        "Test using an SOC equivalency method versus other techniques"
        veh = fastsim.vehicle.Vehicle(11) # Toyota Highlander Hybrid
        cyc_names = ['udds', 'us06', 'hwfet'] #, 'longHaulDriveCycle']
        results = {
            'cycle': [],
            'init_soc': [],
            'delta_soc': [],
            'fuel_kJ': [],
            'soc_corrected_fuel_kJ': [],
            'equivalent_fuel_kJ': [],
        }
        for cyc_name in cyc_names:
            #print(f"CYCLE: {cyc_name}")
            cyc = fastsim.cycle.Cycle(cyc_name)

            sd0 = fastsim.simdrive.SimDriveClassic(cyc, veh)
            sd0.sim_drive() # with SOC correction
            for initSoc in np.linspace(veh.minSoc, veh.maxSoc, 10, endpoint=True): # [(veh.minSoc + veh.maxSoc)/2.0]
                #print(f"initSoc: {initSoc}")
                sd = fastsim.simdrive.SimDriveClassic(cyc, veh)
                sd.sim_drive(initSoc)
                delta_soc = sd.soc[-1] - sd.soc[0]
                #print(f"SOC0: {initSoc} ({sd.soc[0]}) -- these two numbers should be the same")
                #print(f"dSOC: {delta_soc}")
                essDischgKj = sd.essKwOutAch * sd.cyc.dt_s
                #print(f"type(sd.essDischgKj): {type(sd.essDischgKj)}")
                charge = essDischgKj < 0.0
                discharge = essDischgKj > 0.0
                ess_chg_kJ = -1 * essDischgKj[charge].sum()
                dt_chg_s = sd.cyc.dt_s[charge].sum()
                ess_dis_kJ = essDischgKj[discharge].sum()
                dt_dis_s = sd.cyc.dt_s[discharge].sum()
                avg_pwr_ess_chg_kW = ess_chg_kJ / dt_chg_s
                avg_pwr_ess_dis_kW = ess_dis_kJ / dt_dis_s
                #print(f"Time charging (s): {dt_chg_s}")
                #print(f"Energy charging (kJ): {ess_chg_kJ}")
                #print(f"Average power charging (kW): {avg_pwr_ess_chg_kW}")
                #print(f"Time discharging (s): {dt_dis_s}")
                #print(f"Energy discharging (kJ): {ess_dis_kJ}")
                #print(f"Average power discharging (kW): {avg_pwr_ess_dis_kW}")
                ess_eff = np.sqrt(veh.essRoundTripEff)
                ess_chg_eff = veh.essRoundTripEff ** 0.5
                ess_dis_eff = veh.essRoundTripEff ** 0.5
                mc_chg_eff = np.abs(sd.mcElecKwInAch[charge].sum() / sd.mcMechKwOutAch[charge].sum())
                if not discharge.any():
                    mc_dis_eff = mc_chg_eff
                else:
                    mc_dis_eff = np.abs(sd.mcMechKwOutAch[discharge].sum() / sd.mcElecKwInAch[discharge].sum())
                #print(f"ESS charge efficiency: {ess_chg_eff*100.0} %")
                #print(f"ESS discharge efficiency: {ess_dis_eff*100.0} %")
                #print(f"MC charge efficiency: {mc_chg_eff*100.0} %")
                #print(f"MC discharge efficiency: {mc_dis_eff*100.0} %")
                fc_eff = sd.fcKwOutAch.sum() / sd.fcKwInAch.sum()
                regen_kJ = sd.transKwInAch[charge] * sd.cyc.dt_s[charge]
                regen_kJ[regen_kJ > 0.0] = 0.0
                regen_kJ = -1.0 * regen_kJ
                E_regen_kJ = regen_kJ.sum()
                fc_charge_kJ = (sd.fcKwOutAch[charge] * sd.cyc.dt_s[charge] - regen_kJ)
                fc_charge_kJ[fc_charge_kJ < 0] = 0.0
                mc_regen_in_kJ = sd.mcMechKwOutAch[charge] * sd.cyc.dt_s[charge]
                mc_regen_in_kJ[mc_regen_in_kJ > 0.0] = 0.0
                mc_regen_in_kJ = -1.0 * mc_regen_in_kJ
                fc_charge_frac = fc_charge_kJ.sum() / mc_regen_in_kJ.sum()
                # what was the total engine contribution of energy to the ESS?
                charge = sd.mcMechKwOutAch < 0.0
                E_chg_kJ = (sd.mcMechKwOutAch[charge] * sd.cyc.dt_s[charge]).sum()
                mask = np.logical_and(charge, sd.transKwInAch < 0.0)
                E_tx_chg_kJ = (sd.transKwInAch[mask] * sd.cyc.dt_s[mask]).sum()
                fc_chg_frac = (E_chg_kJ - E_tx_chg_kJ) / E_chg_kJ
                motoring = sd.transKwInAch > 0.0
                #print(f"FC charge fraction: {fc_chg_frac} (new one)")
                #print(f"FC charge fraction: {fc_charge_frac}")
                #print(f"FC efficiency: {fc_eff*100.0} %")
                fc_eff = sd.fcKwOutAch[motoring].sum() / sd.fcKwInAch[motoring].sum()
                #print(f"FC efficiency*: {fc_eff*100.0} %")
                #print(f"MC efficiency*: {mc_dis_eff*100.0} %")
                mask = sd.mcMechKwOutAch > 0.0
                if not mask.any():
                    mc_dis_eff = mc_chg_eff
                else:
                    mc_dis_eff = np.abs(sd.mcMechKwOutAch[mask].sum() / sd.mcElecKwInAch[mask].sum())
                #print(f"MC efficiency*: {mc_dis_eff*100.0} %")
                kJ__kWh = 3600.0
                mask = sd.mcElecKwInAch > 0.0
                if not mask.any():
                    ess_traction_frac = 1.0
                else:
                    ess_traction_frac = sd.mcElecKwInAch[mask].sum() / sd.essKwOutAch[mask].sum()
                #print(f"ESS traction fraction: {ess_traction_frac*100.0} %")
                my_k = 1.0
                if delta_soc >= 0.0:
                    # need to discharge; the engine charged too much during the cycle
                    #equivalency_factor = -1.0 / (fc_eff * mc_dis_eff * ess_dis_eff)
                    # fuel_lhv_kJ__kg = fastsim.params.get_fuel_lhv_kJ__kg()
                    #equivalent_fuel_kJ = (equivalency_factor * avg_pwr_ess_dis_kW) / (ess_dis_eff)
                    # IDEA: add a fraction of mcMotorInKJ/EssOutKJ which tells us the fraction that is non-aux
                    #print(f"mc_dis_eff: {mc_dis_eff * 100.0} %")
                    #print(f"ess_traction-frac: {ess_traction_frac * 100.0} %")
                    #print(f"fc_eff: {fc_eff * 100.0} %")
                    #print(f"delta_soc: {delta_soc * 100.0} %")
                    my_k = (veh.maxEssKwh * kJ__kWh * ess_eff * mc_dis_eff * ess_traction_frac) / fc_eff
                    #print(f"c: {my_k / 100.0} kJ/dSOCx100")
                    equivalent_fuel_kJ = -1.0 * delta_soc * my_k
                else:
                    #print(f"mc_dis_eff: {mc_dis_eff * 100.0} %")
                    #print(f"ess_traction-frac: {ess_traction_frac * 100.0} %")
                    #print(f"fc_eff: {fc_eff * 100.0} %")
                    #print(f"delta_soc: {delta_soc * 100.0} %")
                    my_k = (veh.maxEssKwh * kJ__kWh) / (ess_eff * mc_chg_eff * fc_eff)
                    equivalent_fuel_kJ = -1.0 * delta_soc * my_k
                my_c = my_k / 100.0
                results['cycle'].append(cyc_name)
                results['init_soc'].append(initSoc)
                results['delta_soc'].append(delta_soc)
                results['fuel_kJ'].append(sd.fuelKj)
                results['equivalent_fuel_kJ'].append(equivalent_fuel_kJ)
                results['soc_corrected_fuel_kJ'].append(sd0.fuelKj)
                expected_fuel_estimate_kJ = sd.fuelKj + equivalent_fuel_kJ
                actual_fuel_estimate_kJ = fastsim.simdrive.estimate_corrected_fuel_kJ(sd)
                self.assertAlmostEqual(
                    expected_fuel_estimate_kJ, actual_fuel_estimate_kJ, -1,
                    msg=f'Discrepancy for soc0={initSoc} & cycle={cyc_name}'
                )
                #print(f"Equivalent Fuel Energy: {equivalent_fuel_kJ} kJ")
                #print(f"Unadjusted Fuel Energy (kJ): {sd.fuelKj}")
                #print(f"Adjusted Fuel Energy (kJ): {sd.fuelKj + equivalent_fuel_kJ}")
                #print(f"SOC Corrected Fuel Energy (kJ): {sd0.fuelKj}")
                if delta_soc == 0.0:
                    c = 0.0
                else:
                    c = (sd.fuelKj - sd0.fuelKj) / (delta_soc * 100) 
                #print(f"c: {c} kJ/dSOCx100")
                #if my_c == 0.0:
                #    print(f"c/my_c: my_c is 0.0")
                #else:
                #    print(f"c/my_c: {c/my_c}")
                #if c == 0.0:
                #    print(f"my_c/c: c is 0.0")
                #else:
                #    print(f"my_c/c: {my_c/c}")
                #print("="*60)
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