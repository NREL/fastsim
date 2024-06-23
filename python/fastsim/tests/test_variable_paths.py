import unittest
import fastsim as fsim

class TestParamPath(unittest.TestCase):
    def test_param_path_list(self):
        # load 2012 Ford Fusion from file
        veh = fsim.Vehicle.from_file(
            str(fsim.package_root() / "../../tests/assets/2012_Ford_Fusion.yaml")
        )
        baseline_variable_paths = ['fc.eff_min', 'fc.pwr_out_frac_interp', 'fc.pwr_ramp_lag_seconds', 'fc.pwr_ramp_lag_hours', 'fc.pwr_idle_fuel_watts', 'fc.mass_kg', 'fc.eff_range', 
                                   'fc.state.pwr_loss_watts', 'fc.state.pwr_tractive_watts', 'fc.state.pwr_aux_watts', 'fc.state.pwr_fuel_watts', 'fc.state.energy_fuel_joules', 
                                   'fc.state.energy_tractive_joules', 'fc.state.eff', 'fc.state.energy_loss_joules', 'fc.state.energy_aux_joules', 'fc.state.fc_on', 
                                   'fc.state.pwr_prop_max_watts', 'fc.state.i', 'fc.eff_max', 'fc.specific_pwr_kw_per_kg', 'fc.pwr_out_max_watts', 'fc.history.i', 
                                   'fc.history.pwr_prop_max_watts', 'fc.history.pwr_tractive_watts', 'fc.history.energy_aux_joules', 'fc.history.pwr_fuel_watts', 
                                   'fc.history.energy_loss_joules', 'fc.history.fc_on', 'fc.history.pwr_aux_watts', 'fc.history.energy_tractive_joules', 'fc.history.energy_fuel_joules', 
                                   'fc.history.pwr_loss_watts', 'fc.history.eff', 'fc.pwr_out_max_init_watts', 'fc.eff_interp', 'fc.save_interval', 'pwr_aux_watts', 'trans_eff', 'state', 
                                   'chassis', 'year', 'save_interval', 'history', 'name']
        baseline_history_variable_paths = ['fc.history.i', 'fc.history.pwr_prop_max_watts', 'fc.history.pwr_tractive_watts', 'fc.history.energy_aux_joules', 'fc.history.pwr_fuel_watts', 
                                           'fc.history.energy_loss_joules', 'fc.history.fc_on', 'fc.history.pwr_aux_watts', 'fc.history.energy_tractive_joules', 'fc.history.energy_fuel_joules', 
                                           'fc.history.pwr_loss_watts', 'fc.history.eff', 'history', 'fc.history.i', 'fc.history.pwr_prop_max_watts', 'fc.history.pwr_tractive_watts', 
                                           'fc.history.energy_aux_joules', 'fc.history.pwr_fuel_watts', 'fc.history.energy_loss_joules', 'fc.history.fc_on', 'fc.history.pwr_aux_watts', 
                                           'fc.history.energy_tractive_joules', 'fc.history.energy_fuel_joules', 'fc.history.pwr_loss_watts', 'fc.history.eff', 'history']
        assert(baseline_variable_paths.sort()==veh.variable_path_list().sort())
        assert(baseline_history_variable_paths.sort()==veh.history_path_list().sort())


if __name__ == '__main__':
    unittest.main()