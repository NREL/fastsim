use crate::prelude::{FuelConverterState, ReversibleEnergyStorageState};

use super::{vehicle::VehicleState, *};

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
    /// hybrid powertrain mass
    pub hev_controls: HEVControls,
    pub(crate) mass: Option<si::Mass>,
    // TODO: add enum for controling fraction of aux pwr handled by battery vs engine
    // TODO: add enum for controling fraction of tractive pwr handled by battery vs engine -- there
    // might be many ways we'd want to do this, especially since there will be thermal models involved
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
        self.fc.init()?;
        self.res.init()?;
        self.em.init()?;
        Ok(())
    }
}

impl Powertrain for Box<HybridElectricVehicle> {
    fn get_cur_pwr_tract_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        let (pwr_res_tract_max, pwr_res_regen_max) =
            self.res.get_cur_pwr_out_max(pwr_aux, None, None)?;
        self.em
            .get_cur_pwr_tract_out_max(pwr_res_tract_max, pwr_res_regen_max, pwr_aux, dt)
    }
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        pwr_aux: si::Power,
        enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: replace with actual logic.  Should probably have vehicle controls enum in `HybridElectricVehicle`
        let (fc_pwr_out_req, em_pwr_out_req) = (0.5 * pwr_out_req, 0.5 * pwr_out_req);

        let enabled = true; // TODO: replace with a stop/start model
        self.fc.solve(fc_pwr_out_req, pwr_aux, enabled, dt)?;
        // self.fs.solve()
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
        let fc_mass = self.fc.mass()?;
        let fs_mass = self.fs.mass()?;
        let res_mass = self.res.mass()?;
        let em_mass = self.em.mass()?;
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

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub enum HEVControls {
    /// Controls that attempt to exactly match fastsim-2
    Fastsim2,
    /// Purely greedy controls
    RESGreedy,
    // TODO: add `SpeedAware` to enable buffers similar to fastsim-2 but without
    // the feature from fastsim-2 that forces the fc to be greedily meet power demand
    // when it's on
}

impl HEVControls {
    fn get_pwr_fc_and_res(
        &self,
        pwr_out_req: si::Power,
        veh_state: &VehicleState,
        fc_state: &FuelConverterState,
        res_state: &ReversibleEnergyStorageState,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        if pwr_out_req >= uc::W * 0. {
            match self {
                Self::Fastsim2 => {
                    todo!()
                }
                Self::RESGreedy => {
                    let fc_pwr = fc_state.pwr_out_max
                        / (fc_state.pwr_out_max + res_state.pwr_prop_max)
                        * pwr_out_req;
                    let res_pwr = res_state.pwr_prop_max
                        / (fc_state.pwr_out_max + res_state.pwr_prop_max)
                        * pwr_out_req;

                    Ok((fc_pwr, res_pwr))
                }
            }
        } else {
            match self {
                Self::Fastsim2 => {
                    todo!()
                }
                Self::RESGreedy => {
                    let res_pwr = res_state.pwr_regen_max.min(-pwr_out_req);
                    Ok((0. * uc::W, res_pwr))
                }
            }
        }
    }
}
