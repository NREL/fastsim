use super::*;

#[serde_api]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Battery electric vehicle
pub struct BatteryElectricVehicle {
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub em: ElectricMachine,
    #[has_state]
    pub transmission: Transmission,
    pub(crate) mass: Option<si::Mass>,
}

#[named_struct_pyo3_api]
impl BatteryElectricVehicle {}

impl Init for BatteryElectricVehicle {
    fn init(&mut self) -> Result<(), Error> {
        self.res
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.em
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.transmission
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        Ok(())
    }
}

impl SerdeAPI for BatteryElectricVehicle {}

impl Mass for BatteryElectricVehicle {
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
        let res_mass = self.res.mass().with_context(|| anyhow!(format_dbg!()))?;
        let em_mass = self.em.mass().with_context(|| anyhow!(format_dbg!()))?;
        let transmission_mass = self
            .transmission
            .mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        match (res_mass, em_mass, transmission_mass) {
            (Some(res_mass), Some(em_mass), Some(transmission_mass)) => {
                Ok(Some(em_mass + res_mass + transmission_mass))
            }
            (None, None, None) => Ok(None),
            _ => bail!(
                "`{}` field masses are not consistently set to `Some` or `None`",
                stringify!(BatteryElectricVehicle)
            ),
        }
    }

    fn expunge_mass_fields(&mut self) {
        self.res.expunge_mass_fields();
        self.em.expunge_mass_fields();
        self.transmission.expunge_mass_fields();
        self.mass = None;
    }
}

impl HistoryMethods for BatteryElectricVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in BatteryElectricVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.res.set_save_interval(save_interval)?;
        self.em.set_save_interval(save_interval)?;
        self.transmission.set_save_interval(save_interval)?;
        Ok(())
    }
    fn clear(&mut self) {
        self.res.clear();
        self.em.clear();
        self.transmission.clear();
    }
}

impl Powertrain for BatteryElectricVehicle {
    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        _enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let pwr_in_transmission = self
            .transmission
            .get_pwr_in_req(pwr_out_req)
            .with_context(|| anyhow!(format_dbg!()))?;
        let pwr_in_em = self
            .em
            .get_pwr_in_req(pwr_in_transmission, dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.res
            .solve(pwr_in_em, dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }

    fn get_curr_pwr_prop_out_max(&self) -> anyhow::Result<(si::Power, si::Power)> {
        Ok((
            *self
                .em
                .state
                .pwr_mech_fwd_out_max
                .get_fresh(|| format_dbg!())?,
            *self
                .em
                .state
                .pwr_mech_regen_max
                .get_fresh(|| format_dbg!())?,
        ))
    }

    fn set_curr_pwr_prop_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
        _veh_state: &VehicleState,
    ) -> anyhow::Result<()> {
        // TODO: account for transmission efficiency in here
        let disch_buffer = si::Energy::ZERO;
        let chrg_buffer = si::Energy::ZERO;
        self.res
            .set_curr_pwr_out_max(dt, disch_buffer, chrg_buffer)
            .with_context(|| anyhow!(format_dbg!()))?;

        self.res
            .set_curr_pwr_prop_max(pwr_aux)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.em
            .set_curr_pwr_prop_out_max(
                *self.res.state.pwr_prop_max.get_fresh(|| format_dbg!())?,
                *self.res.state.pwr_regen_max.get_fresh(|| format_dbg!())?,
                dt,
            )
            .with_context(|| anyhow!(format_dbg!()))?;

        Ok(())
    }

    /// Regen braking power, positive means braking is happening
    fn pwr_regen(&self) -> anyhow::Result<si::Power> {
        // When `pwr_mech_prop_out` is negative, regen is happening.  First, clip it at 0, and then negate it.
        // see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=e8f7af5a6e436dd1163fa3c70931d18d
        // for example
        Ok(-self
            .em
            .state
            .pwr_mech_prop_out
            .get_fresh(|| format_dbg!())?
            .max(si::Power::ZERO))
    }
}

impl BatteryElectricVehicle {
    /// Solve change in temperature and other thermal effects
    /// # Arguments
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_hvac_to_res`: thermal power flowing from [Vehicle::hvac] system to [ReversibleEnergyStorage::thrml]
    /// - `te_cab`: cabin temperature for heat transfer interaction with [ReversibleEnergyStorage]
    /// - `dt`: simulation time step size
    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_hvac_to_res: si::Power,
        te_cab: Option<si::Temperature>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.res
            .solve_thermal(te_amb, pwr_thrml_hvac_to_res, te_cab, dt)
            .with_context(|| format_dbg!())?;
        Ok(())
    }
}
