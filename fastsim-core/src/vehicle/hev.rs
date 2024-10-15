use crate::prelude::{ElectricMachineState, FuelConverterState};

use super::{vehicle_model::VehicleState, *};

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
/// Hybrid vehicle with both engine and reversible energy storage (aka battery)
/// This type of vehicle is not likely to be widely prevalent due to modularity of consists.
pub struct HybridElectricVehicle {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub em: ElectricMachine,
    /// control strategy for distributing power demand between `fc` and `res`
    #[serde(default)]
    pub pt_cntrl: HEVPowertrainControls,
    /// control strategy for distributing aux power demand between `fc` and `res`
    #[serde(default)]
    pub aux_cntrl: HEVAuxControls,
    /// hybrid powertrain mass
    pub(crate) mass: Option<si::Mass>,
}

impl SaveInterval for HybridElectricVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in HybridElectricVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.res.save_interval = save_interval;
        self.em.save_interval = save_interval;
        Ok(())
    }
}

impl Init for HybridElectricVehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        self.fc.init().with_context(|| anyhow!(format_dbg!()))?;
        self.res.init().with_context(|| anyhow!(format_dbg!()))?;
        self.em.init().with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }
}

impl Powertrain for Box<HybridElectricVehicle> {
    fn set_curr_pwr_prop_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: account for transmission efficiency in here
        self.fc
            .set_curr_pwr_out_max(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.res
            .set_curr_pwr_out_max(None, None)
            .with_context(|| anyhow!(format_dbg!()))?;
        let (pwr_aux_res, pwr_aux_fc) = {
            match self.aux_cntrl {
                HEVAuxControls::AuxOnResPriority => {
                    if pwr_aux <= self.res.state.pwr_disch_max {
                        (pwr_aux, si::Power::ZERO)
                    } else {
                        (si::Power::ZERO, pwr_aux)
                    }
                }
                HEVAuxControls::AuxOnFcPriority => (si::Power::ZERO, pwr_aux),
            }
        };
        self.fc
            .set_curr_pwr_prop_max(pwr_aux_fc)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.res
            .set_curr_pwr_prop_max(pwr_aux_res)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.em
            .set_curr_pwr_prop_out_max(
                // TODO: add means of controlling whether fc can provide power to em and also how much
                // Try out a 'power out type' enum field on the fuel converter with variants for mechanical and electrical
                self.res.state.pwr_prop_max,
                self.res.state.pwr_regen_max,
                dt,
            )
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }

    fn get_curr_pwr_prop_out_max(&self) -> anyhow::Result<(si::Power, si::Power)> {
        Ok((
            self.em.state.pwr_mech_fwd_out_max + self.fc.state.pwr_prop_max,
            self.em.state.pwr_mech_bwd_out_max,
        ))
    }

    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        _veh_state: &VehicleState,
        _enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let (fc_pwr_out_req, em_pwr_out_req) =
            self.pt_cntrl
                .get_pwr_fc_and_em(pwr_out_req, &self.fc.state, &self.em.state)?;
        // TODO: replace with a stop/start model
        // TODO: figure out fancier way to handle apportionment of `pwr_aux` between `fc` and `res`
        let enabled = true;

        self.fc
            .solve(fc_pwr_out_req, enabled, dt)
            .with_context(|| format_dbg!())?;
        let res_pwr_out_req = self
            .em
            .get_pwr_in_req(em_pwr_out_req, dt)
            .with_context(|| format_dbg!())?;
        self.res
            .solve(res_pwr_out_req, dt)
            .with_context(|| format_dbg!())?;
        Ok(())
    }

    fn pwr_regen(&self) -> si::Power {
        // When `pwr_mech_prop_out` is negative, regen is happening.  First, clip it at 0, and then negate it.
        // see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=e8f7af5a6e436dd1163fa3c70931d18d
        // for example
        -self.em.state.pwr_mech_prop_out.min(0. * uc::W)
    }
}

impl Mass for HybridElectricVehicle {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
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
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.mass = match new_mass {
            // Set using provided `new_mass`, setting constituent mass fields to `None` to match if inconsistent
            Some(new_mass) => {
                if let Some(dm) = derived_mass {
                    if dm != new_mass {
                        #[cfg(feature = "logging")]
                        log::warn!(
                            "Derived mass does not match provided mass, setting `{}` consituent mass fields to `None`",
                            stringify!(HybridElectricVehicle));
                        self.expunge_mass_fields();
                    }
                }
                Some(new_mass)
            }
            // Set using `derived_mass()`, failing if it returns `None`
            None => Some(derived_mass.with_context(|| {
                format!(
                    "Not all mass fields in `{}` are set and no mass was provided.",
                    stringify!(HybridElectricVehicle)
                )
            })?),
        };
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let fc_mass = self.fc.mass().with_context(|| anyhow!(format_dbg!()))?;
        let fs_mass = self.fs.mass().with_context(|| anyhow!(format_dbg!()))?;
        let res_mass = self.res.mass().with_context(|| anyhow!(format_dbg!()))?;
        let em_mass = self.em.mass().with_context(|| anyhow!(format_dbg!()))?;
        match (fc_mass, fs_mass, res_mass, em_mass) {
            (Some(fc_mass), Some(fs_mass), Some(res_mass), Some(em_mass)) => {
                Ok(Some(fc_mass + fs_mass + em_mass + res_mass))
            }
            (None, None, None, None) => Ok(None),
            _ => bail!(
                "`{}` field masses are not consistently set to `Some` or `None`",
                stringify!(HybridElectricVehicle)
            ),
        }
    }

    fn expunge_mass_fields(&mut self) {
        self.fc.expunge_mass_fields();
        self.fs.expunge_mass_fields();
        self.res.expunge_mass_fields();
        self.em.expunge_mass_fields();
        self.mass = None;
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default)]
pub enum HEVAuxControls {
    /// If feasible, use [ReversibleEnergyStorage] to handle aux power demand
    #[default]
    AuxOnResPriority,
    /// If feasible, use [FuelConverter] to handle aux power demand
    AuxOnFcPriority,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default)]
pub enum HEVPowertrainControls {
    /// Controls that attempt to exactly match fastsim-2
    Fastsim2,
    /// Purely greedy controls that favor charging or discharging the
    /// battery as much as possible.
    #[default]
    RESGreedy,
    // TODO: add `SpeedAware` to enable buffers similar to fastsim-2 but without
    // the feature from fastsim-2 that forces the fc to be greedily meet power demand
    // when it's on
}

impl HEVPowertrainControls {
    fn get_pwr_fc_and_em(
        &self,
        pwr_out_req: si::Power,
        fc_state: &FuelConverterState,
        em_state: &ElectricMachineState,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        if pwr_out_req >= si::Power::ZERO {
            // positive net power out of the powertrain
            match self {
                Self::Fastsim2 => {
                    bail!("{}\nnot yet implemented!", format_dbg!())
                }
                Self::RESGreedy => {
                    // cannot exceed ElectricMachine max output power
                    let em_pwr = pwr_out_req.min(em_state.pwr_mech_fwd_out_max);
                    let fc_pwr = pwr_out_req - em_pwr;

                    ensure!(
                        fc_pwr >= si::Power::ZERO,
                        format_dbg!(fc_pwr >= si::Power::ZERO)
                    );
                    ensure!(
                        pwr_out_req <= em_state.pwr_mech_fwd_out_max + fc_state.pwr_prop_max,
                        "{}\n`pwr_out_req`: {} kW\n`em_state.pwr_mech_fwd_out_max`: {} kW",
                        format_dbg!(pwr_out_req <= em_state.pwr_mech_fwd_out_max),
                        pwr_out_req.get::<si::kilowatt>(),
                        em_state.pwr_mech_fwd_out_max.get::<si::kilowatt>()
                    );

                    Ok((fc_pwr, em_pwr))
                }
            }
        } else {
            // negative net power out of the powertrain -- i.e. positive net power _into_ powertrain
            match self {
                Self::Fastsim2 => {
                    bail!("{}\nnot yet implemented!", format_dbg!())
                }
                Self::RESGreedy => {
                    // if `em_pwr` is less than magnitude of `pwr_out_req`, friction brakes can handle excess
                    let em_pwr = -em_state.pwr_mech_bwd_out_max.min(-pwr_out_req);
                    Ok((0. * uc::W, em_pwr))
                }
            }
        }
    }
}
