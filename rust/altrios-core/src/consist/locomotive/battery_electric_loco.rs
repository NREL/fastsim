use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::reversible_energy_storage::ReversibleEnergyStorage;
use super::powertrain::ElectricMachine;
use super::LocoTrait;
use crate::imports::*;

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Battery electric locomotive
pub struct BatteryElectricLoco {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub edrv: ElectricDrivetrain,
}

impl BatteryElectricLoco {
    pub fn new(
        reversible_energy_storage: ReversibleEnergyStorage,
        electric_drivetrain: ElectricDrivetrain,
    ) -> Self {
        BatteryElectricLoco {
            res: reversible_energy_storage,
            edrv: electric_drivetrain,
        }
    }

    /// Solve energy consumption for the current power output required
    /// Arguments:
    /// - pwr_out_req: tractive power required
    /// - dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        pwr_aux: si::Power,
    ) -> anyhow::Result<()> {
        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        if self.edrv.state.pwr_elec_prop_in > si::Power::ZERO {
            // positive traction
            self.res
                .solve_energy_consumption(self.edrv.state.pwr_elec_prop_in, pwr_aux, dt)?;
        } else {
            // negative traction
            self.res.solve_energy_consumption(
                self.edrv.state.pwr_elec_prop_in,
                // limit aux power to whatever is actually available
                // TODO: add more detail/nuance to this
                pwr_aux
                    // whatever power is available from regen plus normal
                    .min(self.res.state.pwr_prop_out_max - self.edrv.state.pwr_elec_prop_in)
                    .max(si::Power::ZERO),
                dt,
            )?;
        }
        Ok(())
    }
}

impl LocoTrait for BatteryElectricLoco {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: I think this is where I want to feed in the catenary
        self.res.set_cur_pwr_out_max(pwr_aux.unwrap(), None, None)?;
        self.edrv
            .set_cur_pwr_max_out(self.res.state.pwr_prop_out_max, None)?;
        self.edrv
            .set_cur_pwr_regen_max(self.res.state.pwr_regen_out_max)?;

        // power rate is never limiting in BEL, but assuming dt will be same
        // in next time step, we can synthesize a rate
        self.edrv.set_pwr_rate_out_max(
            (self.edrv.state.pwr_mech_out_max - self.edrv.state.pwr_mech_prop_out) / dt,
        );
        Ok(())
    }

    fn save_state(&mut self) {
        self.save_state();
    }

    fn step(&mut self) {
        self.step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.res.state.energy_loss + self.edrv.state.energy_loss
    }
}
