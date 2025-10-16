use super::*;

#[serde_api]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, StateMethods, SetCumulative)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Conventional vehicle with only a FuelConverter as a power source
pub struct ConventionalVehicle {
    pub fs: FuelStorage,
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub transmission: Transmission,
    pub(crate) mass: Option<si::Mass>,
    /// Alternator efficiency used to calculate aux mechanical power demand on engine
    pub alt_eff: si::Ratio,
}

#[pyo3_api]
impl ConventionalVehicle {}

impl SerdeAPI for ConventionalVehicle {}
impl Init for ConventionalVehicle {
    fn init(&mut self) -> Result<(), Error> {
        self.fc
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.fs
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        self.transmission
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        Ok(())
    }
}

impl HistoryMethods for ConventionalVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in ConventionalVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        // self.fs.set_save_interval(save_interval)?;
        self.fc.set_save_interval(save_interval)?;
        self.transmission.set_save_interval(save_interval)?;
        Ok(())
    }
    fn clear(&mut self) {
        self.fc.clear();
        self.transmission.clear();
    }
}

impl Powertrain for Box<ConventionalVehicle> {
    fn set_curr_pwr_prop_out_max(
        &mut self,
        _pwr_upstream: (si::Power, si::Power),
        pwr_aux: si::Power,
        dt: si::Time,
        _veh_state: &VehicleState,
    ) -> anyhow::Result<()> {
        // TODO: account for transmission efficiency in here
        self.fc
            .set_curr_pwr_out_max(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.fc
            .set_curr_pwr_prop_max(pwr_aux / self.alt_eff)
            .with_context(|| anyhow!(format_dbg!()))?;
        self.transmission
            .set_curr_pwr_prop_out_max(
                (
                    *self.fc.state.pwr_prop_max.get_fresh(|| format_dbg!())?,
                    si::Power::ZERO,
                ),
                f64::NAN * uc::W,
                dt,
                _veh_state,
            )
            .with_context(|| format_dbg!())?;
        Ok(())
    }

    fn get_curr_pwr_prop_out_max(&self) -> anyhow::Result<(si::Power, si::Power)> {
        self.transmission
            .get_curr_pwr_prop_out_max()
            .with_context(|| format_dbg!())
    }

    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        _enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<Option<si::Power>> {
        // NOTE: think about the possibility of engine braking, not urgent
        ensure!(pwr_out_req >= si::Power::ZERO, format_dbg!());
        ensure!(almost_le_uom(
            &pwr_out_req,
            self.transmission
                .state
                .pwr_out_fwd_max
                .get_fresh(|| format_dbg!())?,
            None
        ));
        ensure!(almost_le_uom(
            &pwr_out_req,
            self.transmission
                .state
                .pwr_out_fwd_max
                .get_fresh(|| format_dbg!())?,
            None
        ));
        let enabled = true; // TODO: replace with a stop/start model
        let pwr_in_transmission = self
            .transmission
            .solve(pwr_out_req, true, dt)
            .with_context(|| format_dbg!())?
            .with_context(|| format!("{}\nExpected `Some`", format_dbg!()))?;
        self.fc.solve(pwr_in_transmission, enabled, dt)?;
        Ok(None)
    }

    fn pwr_regen(&self) -> anyhow::Result<si::Power> {
        Ok(si::Power::ZERO)
    }
}

impl ConventionalVehicle {
    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: Option<si::Power>,
        veh_state: &mut VehicleState,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.fc
            .solve_thermal(te_amb, pwr_thrml_fc_to_cab, veh_state, dt)
    }
}

impl TryFrom<&fastsim_2::vehicle::RustVehicle> for ConventionalVehicle {
    type Error = anyhow::Error;
    fn try_from(f2veh: &fastsim_2::vehicle::RustVehicle) -> anyhow::Result<ConventionalVehicle> {
        let conv = ConventionalVehicle {
            fs: {
                let mut fs = FuelStorage {
                    pwr_out_max: f2veh.fs_max_kw * uc::KW,
                    pwr_ramp_lag: f2veh.fs_secs_to_peak_pwr * uc::S,
                    energy_capacity: f2veh.fs_kwh * uc::KWH,
                    specific_energy: Some(
                        super::vehicle_model::FUEL_LHV_MJ_PER_KG * uc::MJ / uc::KG,
                    ),
                    mass: None,
                };
                fs.set_mass(None, MassSideEffect::None)
                    .with_context(|| anyhow!(format_dbg!()))?;
                fs
            },
            fc: FuelConverter::try_from(f2veh.clone())?,
            transmission: Transmission::try_from(f2veh.clone())?,
            mass: None,
            alt_eff: f2veh.alt_eff * uc::R,
        };
        Ok(conv)
    }
}

impl Mass for ConventionalVehicle {
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
                    stringify!(ConventionalVehicle)
                )
            })?),
        };
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let fc_mass = self.fc.mass().with_context(|| anyhow!(format_dbg!()))?;
        let fs_mass = self.fs.mass().with_context(|| anyhow!(format_dbg!()))?;
        let transmission_mass = self
            .transmission
            .mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        match (fc_mass, fs_mass, transmission_mass) {
            (Some(fc_mass), Some(fs_mass), Some(transmission_mass)) => {
                Ok(Some(fc_mass + fs_mass + transmission_mass))
            }
            (None, None, None) => Ok(None),
            _ => bail!(
                "`{}` field masses are not consistently set to `Some` or `None`",
                stringify!(ConventionalVehicle)
            ),
        }
    }

    fn expunge_mass_fields(&mut self) {
        self.fc.expunge_mass_fields();
        self.fs.expunge_mass_fields();
        self.transmission.expunge_mass_fields();
        self.mass = None;
    }
}
