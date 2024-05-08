use super::*;
// use crate::imports::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
/// Battery electric vehicle
pub struct BatteryElectricVehicle {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub em: ElectricMachine,
    pub(crate) mass: Option<si::Mass>,
}

impl Mass for BatteryElectricVehicle {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self.derived_mass()?;
        match (derived_mass, self.mass) {
            (Some(derived_mass), Some(set_mass)) => {
                ensure!(
                    utils::almost_eq_uom(&set_mass, &derived_mass, None),
                    format!(
                        "{}",
                        format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                    )
                );
                Ok(Some(set_mass))
            }
            _ => Ok(self.mass.or(derived_mass)),
        }
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        ensure!(
            side_effect == MassSideEffect::None,
            "At the powertrain level, only `MassSideEffect::None` is allowed"
        );
        let derived_mass = self.derived_mass()?;
        self.mass = match new_mass {
            // Set using provided `new_mass`, setting constituent mass fields to `None` to match if inconsistent
            Some(new_mass) => {
                if let Some(dm) = derived_mass {
                    if dm != new_mass {
                        log::warn!(
                            "Derived mass does not match provided mass, setting `{}` consituent mass fields to `None`",
                            stringify!(BatteryElectricVehicle));
                        self.expunge_mass_fields();
                    }
                }
                Some(new_mass)
            }
            // Set using `derived_mass()`, failing if it returns `None`
            None => Some(derived_mass.with_context(|| {
                format!(
                    "Not all mass fields in `{}` are set and no mass was provided.",
                    stringify!(BatteryElectricVehicle)
                )
            })?),
        };
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let res_mass = self.res.mass()?;
        let em_mass = self.em.mass()?;
        match (res_mass, em_mass) {
            (Some(res_mass), Some(em_mass)) => Ok(Some(em_mass + res_mass)),
            (None, None) => Ok(None),
            _ => bail!(
                "`{}` field masses are not consistently set to `Some` or `None`",
                stringify!(BatteryElectricVehicle)
            ),
        }
    }

    fn expunge_mass_fields(&mut self) {
        self.res.expunge_mass_fields();
        self.em.expunge_mass_fields();
        self.mass = None;
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

impl Powertrain for BatteryElectricVehicle {
    /// Solve energy consumption for the current power output required
    /// # Arguments:
    /// - pwr_out_req: tractive power required
    /// - dt: time step size
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let pwr_out_req_from_res = self.em.get_pwr_in_req(pwr_out_req, pwr_aux, dt)?;
        // TODO: revisit this if...else block
        if self.em.state.pwr_elec_prop_in > si::Power::ZERO {
            // positive traction
            self.res.solve(pwr_out_req_from_res, pwr_aux, dt)?;
        } else {
            // negative traction (should this be different from positive traction here?)
            self.res.solve(
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

    fn get_cur_pwr_tract_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<(si::Power, si::Power)> {
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
