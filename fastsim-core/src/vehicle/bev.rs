use super::*;
// use crate::imports::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Battery electric vehicle
pub struct BatteryElectricVehicle {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub em: ElectricMachine,
}

impl BatteryElectricVehicle {
    pub fn new(
        reversible_energy_storage: ReversibleEnergyStorage,
        electric_machine: ElectricMachine,
    ) -> Self {
        BatteryElectricVehicle {
            res: reversible_energy_storage,
            em: electric_machine,
        }
    }
}

impl SaveInterval for BatteryElectricVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in BatteryElectricVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.res.save_interval = save_interval;
        self.em.save_interval = save_interval;
        Ok(())
    }
}

impl Powertrain for Box<BatteryElectricVehicle> {
    /// Solve energy consumption for the current power output required
    /// Arguments:
    /// - pwr_out_req: tractive power required
    /// - dt: time step size
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        self.em.set_pwr_in_req(pwr_out_req, dt)?;
        if self.em.state.pwr_elec_prop_in > si::Power::ZERO {
            // positive traction
            self.res
                .solve_energy_consumption(self.em.state.pwr_elec_prop_in, pwr_aux, dt)?;
        } else {
            // negative traction
            self.res.solve_energy_consumption(
                self.em.state.pwr_elec_prop_in,
                // limit aux power to whatever is actually available
                // TODO: add more detail/nuance to this
                pwr_aux
                    // whatever power is available from regen plus normal
                    .min(self.res.state.pwr_prop_out_max - self.em.state.pwr_elec_prop_in)
                    .max(si::Power::ZERO),
                dt,
            )?;
        }
        Ok(())
    }

    fn get_curr_pwr_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        todo!();
        // self.res.set_cur_pwr_out_max(pwr_aux.unwrap(), None, None)?;
        // self.em.set_cur_pwr_max_out(self.res.state.pwr_prop_out_max, None)?;
        // self.em.set_cur_pwr_regen_max(self.res.state.pwr_regen_out_max)?;

        // // power rate is never limiting in BEL, but assuming dt will be same
        // // in next time step, we can synthesize a rate
        // self.em.set_pwr_rate_out_max(
        //     (self.em.state.pwr_mech_out_max - self.em.state.pwr_mech_prop_out) / dt,
        // );
    }
}
