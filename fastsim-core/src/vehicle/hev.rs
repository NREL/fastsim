use super::{vehicle_model::VehicleState, *};
use crate::prelude::ElectricMachineState;

#[fastsim_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
#[non_exhaustive]
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
    #[has_state]
    pub transmission: Transmission,
    /// control strategy for distributing power demand between `fc` and `res`
    #[serde(default)]
    pub pt_cntrl: HEVPowertrainControls,
    /// control strategy for distributing aux power demand between `fc` and `res`
    #[serde(default)]
    pub aux_cntrl: HEVAuxControls,
    /// hybrid powertrain mass
    pub(crate) mass: Option<si::Mass>,
    #[serde(default)]
    pub sim_params: HEVSimulationParams,
    /// field for tracking current state
    #[serde(default)]
    pub state: HEVState,
    /// vector of [Self::state]
    #[serde(default, skip_serializing_if = "HEVStateHistoryVec::is_empty")]
    pub history: HEVStateHistoryVec,
    /// vector of SOC balance iterations
    #[serde(default)]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub soc_bal_iter_history: Vec<Self>,
}

impl SaveInterval for HybridElectricVehicle {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        bail!("`save_interval` is not implemented in HybridElectricVehicle")
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.res.save_interval = save_interval;
        self.em.save_interval = save_interval;
        self.transmission.save_interval = save_interval;
        Ok(())
    }
}

impl Init for HybridElectricVehicle {
    fn init(&mut self) -> anyhow::Result<()> {
        self.fc.init().with_context(|| anyhow!(format_dbg!()))?;
        self.res.init().with_context(|| anyhow!(format_dbg!()))?;
        self.em.init().with_context(|| anyhow!(format_dbg!()))?;
        self.transmission
            .init()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.pt_cntrl
            .init()
            .with_context(|| anyhow!(format_dbg!()))?;
        self.state.init().with_context(|| anyhow!(format_dbg!()))?;
        Ok(())
    }
}

impl SerdeAPI for HybridElectricVehicle {}

impl Powertrain for Box<HybridElectricVehicle> {
    fn set_curr_pwr_prop_out_max(
        &mut self,
        pwr_aux: si::Power,
        dt: si::Time,
        veh_state: VehicleState,
    ) -> anyhow::Result<()> {
        // TODO: account for transmission efficiency in here
        self.state.fc_on_causes.clear();
        match &self.pt_cntrl {
            HEVPowertrainControls::RGWDB(rgwb) => {
                if self.fc.state.fc_on && self.fc.state.time_on
                    < rgwb.fc_min_time_on.with_context(|| {
                    anyhow!(
                        "{}\n Expected `ResGreedyWithBuffers::init` to have been called beforehand.",
                        format_dbg!()
                    )
                })? {
                    self.state.fc_on_causes.push(FCOnCause::OnTimeTooShort)
                }
            }
            HEVPowertrainControls::Placeholder => {
                todo!()
            }
        };
        self.fc
            .set_curr_pwr_out_max(dt)
            .with_context(|| anyhow!(format_dbg!()))?;
        let disch_buffer: si::Energy = match &self.pt_cntrl {
            HEVPowertrainControls::RGWDB(rgwb) => {
                (0.5 * veh_state.mass
                    * (rgwb
                        .speed_soc_disch_buffer
                        .with_context(|| format_dbg!())?
                        .powi(typenum::P2::new())
                        - veh_state.speed_ach.powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_disch_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
            HEVPowertrainControls::Placeholder => {
                todo!()
            }
        };
        let chrg_buffer: si::Energy = match &self.pt_cntrl {
            HEVPowertrainControls::RGWDB(rgwb) => {
                (0.5 * veh_state.mass
                    * (veh_state.speed_ach.powi(typenum::P2::new())
                        - rgwb
                            .speed_soc_regen_buffer
                            .with_context(|| format_dbg!())?
                            .powi(typenum::P2::new())))
                .max(si::Energy::ZERO)
                    * rgwb
                        .speed_soc_regen_buffer_coeff
                        .with_context(|| format_dbg!())?
            }
            HEVPowertrainControls::Placeholder => {
                todo!()
            }
        };
        self.res
            .set_curr_pwr_out_max(dt, disch_buffer, chrg_buffer)
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
        if pwr_aux_fc > si::Power::ZERO {
            self.state.fc_on_causes.push(FCOnCause::AuxPowerDemand);
        }
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
            self.em.state.pwr_mech_regen_max,
        ))
    }

    fn solve(
        &mut self,
        pwr_out_req: si::Power,
        veh_state: VehicleState,
        _enabled: bool,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        // TODO: address these concerns
        // - add a transmission here
        // - what happens when the fc is on and producing more power than the
        //   transmission requires? It seems like the excess goes straight to the battery,
        //   but it should probably go thourgh the em somehow.
        let (fc_pwr_out_req, em_pwr_out_req) = self
            .pt_cntrl
            .get_pwr_fc_and_em(
                pwr_out_req,
                veh_state,
                &mut self.state,
                &self.fc,
                &self.em.state,
                &self.res,
            )
            .with_context(|| format_dbg!())?;
        let fc_on: bool = !self.state.fc_on_causes.is_empty();

        self.fc
            .solve(fc_pwr_out_req, fc_on, dt)
            .with_context(|| format_dbg!())?;
        let res_pwr_out_req = self
            .em
            .get_pwr_in_req(em_pwr_out_req, dt)
            .with_context(|| format_dbg!())?;
        // TODO: `res_pwr_out_req` probably does not include charging from the engine
        self.res
            .solve(res_pwr_out_req, dt)
            .with_context(|| format_dbg!())?;
        Ok(())
    }

    /// Regen braking power, positive means braking is happening
    fn pwr_regen(&self) -> si::Power {
        // When `pwr_mech_prop_out` is negative, regen is happening.  First, clip it at 0, and then negate it.
        // see https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=e8f7af5a6e436dd1163fa3c70931d18d
        // for example
        -(self.em.state.pwr_mech_prop_out.max(si::Power::ZERO))
    }
}

impl HybridElectricVehicle {
    /// # Arguments
    /// - `te_amb`: ambient temperature
    /// - `pwr_thrml_fc_to_cab`: thermal power flow from [FuelConverter::thrml]
    ///     to [Vehicle::cabin], if cabin is equipped
    /// - `veh_state`: current [VehicleState]
    /// - `pwr_thrml_hvac_to_res`: thermal power flow from [Vehicle::hvac] --
    ///     zero if `None` is passed
    /// - `te_cab`: cabin temperature, required if [ReversibleEnergyStorage::thrml] is `Some`
    /// - `dt`: simulation time step size
    pub fn solve_thermal(
        &mut self,
        te_amb: si::Temperature,
        pwr_thrml_fc_to_cab: Option<si::Power>,
        veh_state: &mut VehicleState,
        pwr_thrml_hvac_to_res: Option<si::Power>,
        te_cab: Option<si::Temperature>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.fc
            .solve_thermal(te_amb, pwr_thrml_fc_to_cab, veh_state, dt)
            .with_context(|| format_dbg!())?;
        self.res
            .solve_thermal(
                te_amb,
                pwr_thrml_hvac_to_res.unwrap_or_default(),
                te_cab,
                dt,
            )
            .with_context(|| format_dbg!())?;
        Ok(())
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
        let transmission_mass = self
            .transmission
            .mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        match (fc_mass, fs_mass, res_mass, em_mass, transmission_mass) {
            (Some(fc_mass), Some(fs_mass), Some(res_mass), Some(em_mass), Some(transmission_mass)) => {
                Ok(Some(fc_mass + fs_mass + res_mass + em_mass + transmission_mass))
            }
            (None, None, None, None, None) => Ok(None),
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
        self.transmission.expunge_mass_fields();
        self.mass = None;
    }
}

#[fastsim_api]
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct FCOnCauses(Vec<FCOnCause>);
impl Init for FCOnCauses {}
impl SerdeAPI for FCOnCauses {}
impl FCOnCauses {
    fn clear(&mut self) {
        self.0.clear();
    }

    #[allow(dead_code)]
    fn pop(&mut self) -> Option<FCOnCause> {
        self.0.pop()
    }

    fn push(&mut self, new: FCOnCause) {
        self.0.push(new)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// TODO: figure out why this is not appearing in the dataframe but is in the pydict
#[fastsim_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[non_exhaustive]
#[serde(default)]
pub struct HEVState {
    /// time step index
    pub i: usize,
    /// Vector of posssible reasons the fc is forced on
    pub fc_on_causes: FCOnCauses,
    /// Number of `walk` iterations required to achieve SOC balance (i.e. SOC
    /// ends at same starting value, ensuring no net [ReversibleEnergyStorage] usage)
    pub soc_bal_iters: u32,
}

impl Init for HEVState {}
impl SerdeAPI for HEVState {}

// Custom serialization
impl Serialize for FCOnCauses {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let joined = self
            .0
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join(", ");
        serializer.serialize_str(&format!("\"[{}]\"", joined))
    }
}

use serde::de::{self, Visitor};
struct FCOnCausesVisitor;
impl Visitor<'_> for FCOnCausesVisitor {
    type Value = FCOnCauses;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "String form of `FCOnCauses`, e.g. `\"[VehicleSpeedTooHigh, FCTemperatureTooLow]\"`",
        )
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Self::visit_str(self, &v)
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let inner: String = v
            .replace("\"", "") // this solves a problem in interactive mode
            .strip_prefix("[")
            .ok_or("Missing leading `[`")
            .map_err(|err| de::Error::custom(err))?
            .strip_suffix("]")
            .ok_or("Missing trailing`]`")
            .map_err(|err| de::Error::custom(err))?
            .to_string();
        let fc_on_causes_str = inner.split(",").map(|x| x.trim()).collect::<Vec<&str>>();
        let fc_on_causes_unchecked = fc_on_causes_str
            .iter()
            .map(|x| {
                if x.is_empty() {
                    None
                } else {
                    Some(FromStr::from_str(x))
                }
            })
            .collect::<Vec<Option<Result<FCOnCause, derive_more::FromStrError>>>>();
        let mut fc_on_causes: FCOnCauses = FCOnCauses(vec![]);
        for (fc_on_cause_unchecked, fc_on_cause_str) in
            fc_on_causes_unchecked.into_iter().zip(fc_on_causes_str)
        {
            if let Some(fc_on_cause_unchecked) = fc_on_cause_unchecked {
                fc_on_causes.0.push(fc_on_cause_unchecked.map_err(|err| {
                    de::Error::custom(format!(
                        "{}\nfc_on_cause_unchecked: {:?}\nfc_on_cause_str: {}",
                        err, fc_on_cause_unchecked, fc_on_cause_str
                    ))
                })?)
            }
        }
        Ok(fc_on_causes)
    }
}

// Custom deserialization
impl<'de> Deserialize<'de> for FCOnCauses {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(FCOnCausesVisitor)
    }
}

impl std::fmt::Display for FCOnCauses {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

#[fastsim_enum_api]
#[derive(
    Clone, Copy, Debug, Deserialize, Serialize, PartialEq, IsVariant, From, TryInto, FromStr,
)]
pub enum FCOnCause {
    /// Engine must be on to self heat if thermal model is enabled
    FCTemperatureTooLow,
    /// Engine must be on for high vehicle speed to ensure powertrain can meet
    /// any spikes in power demand
    VehicleSpeedTooHigh,
    /// Engine has not been on long enough (usually 30 s)
    OnTimeTooShort,
    /// Powertrain power demand exceeds motor and/or battery capabilities
    PropulsionPowerDemand,
    /// Powertrain power demand exceeds optimal motor and/or battery output
    PropulsionPowerDemandSoft,
    /// Aux power demand exceeds battery capability
    AuxPowerDemand,
    /// SOC is below min buffer so FC is charging RES
    ChargingForLowSOC,
}
impl SerdeAPI for FCOnCause {}
impl Init for FCOnCause {}
impl fmt::Display for FCOnCause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

/// Options for controlling simulation behavior
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[non_exhaustive]
pub struct HEVSimulationParams {
    /// [ReversibleEnergyStorage] per [FuelConverter]
    pub res_per_fuel_lim: si::Ratio,
    /// Threshold of SOC balancing iteration for triggering error
    pub soc_balance_iter_err: u32,
    /// Whether to allow iteration to achieve SOC balance
    pub balance_soc: bool,
    /// Whether to save each SOC balance iteration    
    pub save_soc_bal_iters: bool,
}

impl Default for HEVSimulationParams {
    fn default() -> Self {
        Self {
            res_per_fuel_lim: uc::R * 0.005,
            soc_balance_iter_err: 5,
            balance_soc: true,
            save_soc_bal_iters: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default, IsVariant, From, TryInto)]
pub enum HEVAuxControls {
    /// If feasible, use [ReversibleEnergyStorage] to handle aux power demand
    #[default]
    AuxOnResPriority,
    /// If feasible, use [FuelConverter] to handle aux power demand
    AuxOnFcPriority,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, IsVariant, From, TryInto)]
pub enum HEVPowertrainControls {
    /// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
    /// and discharge power inside of static min and max SOC range.  Also, includes
    /// buffer for forcing [FuelConverter] to be active/on.
    RGWDB(Box<RESGreedyWithDynamicBuffers>),
    /// place holder for future variants
    Placeholder,
}

impl Default for HEVPowertrainControls {
    fn default() -> Self {
        Self::RGWDB(Default::default())
    }
}

impl Init for HEVPowertrainControls {
    fn init(&mut self) -> anyhow::Result<()> {
        match self {
            Self::RGWDB(rgwb) => rgwb.init()?,
            Self::Placeholder => {
                todo!()
            }
        }
        Ok(())
    }
}

impl HEVPowertrainControls {
    /// Determines power split between engine and electric machine
    ///
    /// # Arguments
    /// - `pwr_prop_req`: tractive power required
    /// - `veh_state`: vehicle state
    /// - `hev_state`: HEV powertrain state
    /// - `fc`: fuel converter
    /// - `em_state`: electric machine state
    /// - `res`: reversible energy storage (e.g. high voltage battery)
    fn get_pwr_fc_and_em(
        &mut self,
        pwr_prop_req: si::Power,
        veh_state: VehicleState,
        hev_state: &mut HEVState,
        fc: &FuelConverter,
        em_state: &ElectricMachineState,
        res: &ReversibleEnergyStorage,
    ) -> anyhow::Result<(si::Power, si::Power)> {
        let fc_state = &fc.state;
        ensure!(
            // `almost` is in case of negligible numerical precision discrepancies
            almost_le_uom(
                &pwr_prop_req,
                &(em_state.pwr_mech_fwd_out_max + fc_state.pwr_prop_max),
                None
            ),
            "{}
`pwr_out_req`: {} kW
`em_state.pwr_mech_fwd_out_max`: {} kW
`fc_state.pwr_prop_max`: {} kW
`res.state.soc`: {}",
            format_dbg!(),
            pwr_prop_req.get::<si::kilowatt>(),
            em_state.pwr_mech_fwd_out_max.get::<si::kilowatt>(),
            fc_state.pwr_prop_max.get::<si::kilowatt>(),
            res.state.soc.get::<si::ratio>()
        );

        // # Brain dump for thermal stuff
        // TODO: engine on/off w.r.t. thermal stuff should not come into play
        // if there is no component (e.g. cabin) demanding heat from the engine.  My 2019
        // Hyundai Ioniq will turn the engine off if there is no heat demand regardless of
        // the coolant temperature
        // TODO: make sure idle fuel gets converted to heat correctly
        let (fc_pwr, em_pwr) = match self {
            Self::RGWDB(ref mut rgwdb) => {
                handle_fc_on_causes_for_temp(fc, rgwdb, hev_state)?;
                handle_fc_on_causes_for_speed(veh_state, rgwdb, hev_state)?;
                handle_fc_on_causes_for_low_soc(res, rgwdb, hev_state, veh_state)?;
                // `handle_fc_*` below here are asymmetrical for positive tractive power only
                handle_fc_on_causes_for_pwr_demand(
                    rgwdb,
                    pwr_prop_req,
                    em_state,
                    fc_state,
                    hev_state,
                )?;

                // Tractive power `em` must provide before deciding power
                // split, cannot exceed ElectricMachine max output power.
                // Excess demand will be handled by `fc`.  Favors drawing
                // power from `em` before engine
                let em_pwr = pwr_prop_req
                    .min(em_state.pwr_mech_fwd_out_max)
                    .max(-em_state.pwr_mech_regen_max);
                // tractive power handled by fc
                if hev_state.fc_on_causes.is_empty() {
                    // engine is off, and `em_pwr` has already been limited within bounds
                    (si::Power::ZERO, em_pwr)
                } else {
                    // engine has been forced on
                    let frac_of_pwr_for_peak_eff: si::Ratio = rgwdb
                        .frac_of_most_eff_pwr_to_run_fc
                        .with_context(|| format_dbg!())?;
                    let fc_pwr = if pwr_prop_req < si::Power::ZERO {
                        // negative tractive power
                        // max power system can receive from engine during negative traction
                        (em_state.pwr_mech_regen_max + pwr_prop_req)
                            // or peak efficiency power if it's lower than above
                            .min(fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff)
                            // but not negative
                            .max(si::Power::ZERO)
                    } else {
                        // positive tractive power
                        if pwr_prop_req - em_pwr > fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff {
                            // engine needs to run higher than peak efficiency point
                            pwr_prop_req - em_pwr
                        } else {
                            // engine does not need to run higher than peak
                            // efficiency point to make tractive demand

                            // fc handles all power not covered by em
                            (pwr_prop_req - em_pwr)
                                // and if that's less than the
                                // efficiency-focused value, then operate at
                                // that value
                                .max(fc.pwr_for_peak_eff * frac_of_pwr_for_peak_eff)
                                // but don't exceed what what the battery can
                                // absorb + tractive demand
                                .min(pwr_prop_req + em_state.pwr_mech_regen_max)
                        }
                    }
                    // and don't exceed what the fc can do
                    .min(fc_state.pwr_prop_max);

                    // recalculate `em_pwr` based on `fc_pwr`
                    let em_pwr_corrected =
                        (pwr_prop_req - fc_pwr).max(-em_state.pwr_mech_regen_max);
                    (fc_pwr, em_pwr_corrected)
                }
            }
            Self::Placeholder => todo!(),
        };

        Ok((fc_pwr, em_pwr))
    }
}

/// Determines whether power demand requires engine to be on.  Not needed during
/// negative traction.
fn handle_fc_on_causes_for_pwr_demand(
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    pwr_out_req: si::Power,
    em_state: &ElectricMachineState,
    fc_state: &FuelConverterState,
    hev_state: &mut HEVState,
) -> Result<(), anyhow::Error> {
    let frac_pwr_demand_fc_forced_on: si::Ratio = rgwdb
        .frac_pwr_demand_fc_forced_on
        .with_context(|| format_dbg!())?;
    if pwr_out_req
        > frac_pwr_demand_fc_forced_on * (em_state.pwr_mech_fwd_out_max + fc_state.pwr_out_max)
    {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemandSoft);
    }
    if pwr_out_req - em_state.pwr_mech_fwd_out_max >= si::Power::ZERO {
        hev_state
            .fc_on_causes
            .push(FCOnCause::PropulsionPowerDemand);
    }
    Ok(())
}

/// Detemrines whether engine must be on to charge battery
fn handle_fc_on_causes_for_low_soc(
    res: &ReversibleEnergyStorage,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HEVState,
    veh_state: VehicleState,
) -> anyhow::Result<()> {
    rgwdb.state.soc_fc_on_buffer = {
        let energy_delta_to_buffer_speed = 0.5
            * veh_state.mass
            * (rgwdb
                .speed_soc_fc_on_buffer
                .with_context(|| format_dbg!())?
                .powi(typenum::P2::new())
                - veh_state.speed_ach.powi(typenum::P2::new()));
        energy_delta_to_buffer_speed.max(si::Energy::ZERO)
            * rgwdb
                .speed_soc_fc_on_buffer_coeff
                .with_context(|| format_dbg!())?
    } / res.energy_capacity_usable()
        + res.min_soc;
    if res.state.soc < rgwdb.state.soc_fc_on_buffer {
        hev_state.fc_on_causes.push(FCOnCause::ChargingForLowSOC)
    }
    Ok(())
}

/// Determines whether enigne must be on for high speed
fn handle_fc_on_causes_for_speed(
    veh_state: VehicleState,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HEVState,
) -> anyhow::Result<()> {
    if veh_state.speed_ach > rgwdb.speed_fc_forced_on.with_context(|| format_dbg!())? {
        hev_state.fc_on_causes.push(FCOnCause::VehicleSpeedTooHigh);
    }
    Ok(())
}

/// Determines whether engine needs to be on due to low temperature and pushes
/// appropriate variant to `fc_on_causes`
fn handle_fc_on_causes_for_temp(
    fc: &FuelConverter,
    rgwdb: &mut Box<RESGreedyWithDynamicBuffers>,
    hev_state: &mut HEVState,
) -> anyhow::Result<()> {
    match (
        fc.temperature(),
        fc.temp_prev(),
        rgwdb.temp_fc_forced_on,
        rgwdb.temp_fc_allowed_off,
    ) {
        (None, None, None, None) => {}
        (
            Some(temperature),
            Some(temp_prev),
            Some(temp_fc_forced_on),
            Some(temp_fc_allowed_off),
        ) => {
            if
            // temperature is currently below forced on threshold
            temperature < temp_fc_forced_on ||
            // temperature was below forced on threshold and still has not exceeded allowed off threshold
            (temp_prev < temp_fc_forced_on && temperature < temp_fc_allowed_off)
            {
                hev_state.fc_on_causes.push(FCOnCause::FCTemperatureTooLow);
            }
        }
        _ => {
            bail!(
                "{}\n`fc.temperature()`, `fc.temp_prev()`, `rgwdb.temp_fc_forced_on`, and 
`rgwdb.temp_fc_allowed_off` must all be `None` or `Some` because these controls are necessary
for an HEV equipped with thermal models or superfluous otherwise",
                format_dbg!((
                    fc.temperature(),
                    fc.temp_prev(),
                    rgwdb.temp_fc_forced_on,
                    rgwdb.temp_fc_allowed_off
                ))
            );
        }
    }
    Ok(())
}

/// Greedily uses [ReversibleEnergyStorage] with buffers that derate charge
/// and discharge power inside of static min and max SOC range.  Also, includes
/// buffer for forcing [FuelConverter] to be active/on. See [Self::init] for
/// default values.
#[fastsim_api]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, Default)]
#[non_exhaustive]
pub struct RESGreedyWithDynamicBuffers {
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers ramp down in RES discharge.
    pub speed_soc_disch_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of accel buffer
    pub speed_soc_disch_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from minimum SOC corresponding to kinetic energy of
    /// vehicle at this speed that triggers FC to be forced on.
    pub speed_soc_fc_on_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of [Self::speed_soc_fc_on_buffer]
    pub speed_soc_fc_on_buffer_coeff: Option<si::Ratio>,
    /// RES energy delta from maximum SOC corresponding to kinetic energy of
    /// vehicle at current speed minus kinetic energy of vehicle at this speed
    /// triggers ramp down in RES discharge
    pub speed_soc_regen_buffer: Option<si::Velocity>,
    /// Coefficient for modifying amount of regen buffer
    pub speed_soc_regen_buffer_coeff: Option<si::Ratio>,
    /// Minimum time engine must remain on if it was on during the previous
    /// simulation time step.
    pub fc_min_time_on: Option<si::Time>,
    /// Speed at which [FuelConverter] is forced on.
    pub speed_fc_forced_on: Option<si::Velocity>,
    /// Fraction of total aux and powertrain rated power at which
    /// [FuelConverter] is forced on.
    pub frac_pwr_demand_fc_forced_on: Option<si::Ratio>,
    /// Force engine, if on, to run at this fraction of power at which peak
    /// efficiency occurs or the required power, whichever is greater. If SOC is
    /// below min buffer or engine is otherwise forced on and battery has room
    /// to receive charge, engine will run at this level and charge.
    pub frac_of_most_eff_pwr_to_run_fc: Option<si::Ratio>,
    /// Fraction of available charging capacity to use toward running the engine
    /// efficiently.
    // NOTE: this is inherited from fastsim-2 and has no effect here.  After
    // further thought, either remove it or use it.
    pub frac_res_chrg_for_fc: si::Ratio,
    // TODO: put `save_interval` in here
    // NOTE: this is inherited from fastsim-2 and has no effect here.  After
    // further thought, either remove it or use it.
    /// Fraction of available discharging capacity to use toward running the
    /// engine efficiently.
    pub frac_res_dschrg_for_fc: si::Ratio,
    /// temperature at which engine is forced on to warm up
    #[serde(default)]
    pub temp_fc_forced_on: Option<si::Temperature>,
    /// temperature at which engine is allowed to turn off due to being sufficiently warm
    #[serde(default)]
    pub temp_fc_allowed_off: Option<si::Temperature>,
    /// current state of control variables
    #[serde(default)]
    pub state: RGWDBState,
    #[serde(default, skip_serializing_if = "RGWDBStateHistoryVec::is_empty")]
    /// history of current state
    pub history: RGWDBStateHistoryVec,
}

impl Init for RESGreedyWithDynamicBuffers {
    fn init(&mut self) -> anyhow::Result<()> {
        // TODO: make sure these values propagate to the documented defaults above
        self.speed_soc_disch_buffer = self.speed_soc_disch_buffer.or(Some(40.0 * uc::MPH));
        self.speed_soc_disch_buffer_coeff = self.speed_soc_disch_buffer_coeff.or(Some(1.0 * uc::R));
        self.speed_soc_fc_on_buffer = self
            .speed_soc_fc_on_buffer
            .or(Some(self.speed_soc_disch_buffer.unwrap() * 1.1));
        self.speed_soc_fc_on_buffer_coeff = self.speed_soc_fc_on_buffer_coeff.or(Some(1.0 * uc::R));
        self.speed_soc_regen_buffer = self.speed_soc_regen_buffer.or(Some(30. * uc::MPH));
        self.speed_soc_regen_buffer_coeff = self.speed_soc_regen_buffer_coeff.or(Some(1.0 * uc::R));
        self.fc_min_time_on = self.fc_min_time_on.or(Some(uc::S * 5.0));
        self.speed_fc_forced_on = self.speed_fc_forced_on.or(Some(uc::MPH * 75.));
        self.frac_pwr_demand_fc_forced_on =
            self.frac_pwr_demand_fc_forced_on.or(Some(uc::R * 0.75));
        self.frac_of_most_eff_pwr_to_run_fc =
            self.frac_of_most_eff_pwr_to_run_fc.or(Some(1.0 * uc::R));
        Ok(())
    }
}
impl SerdeAPI for RESGreedyWithDynamicBuffers {}

#[fastsim_api]
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative)]
#[serde(default)]
/// State for [RESGreedyWithDynamicBuffers ]
pub struct RGWDBState {
    /// time step index
    pub i: usize,
    /// Vector of posssible reasons the fc is forced on
    pub fc_on_causes: FCOnCauses,
    /// Number of `walk` iterations required to achieve SOC balance (i.e. SOC
    /// ends at same starting value, ensuring no net [ReversibleEnergyStorage] usage)
    pub soc_bal_iters: u32,
    /// buffer at which FC is forced on
    pub soc_fc_on_buffer: si::Ratio,
}

impl Init for RGWDBState {}
impl SerdeAPI for RGWDBState {}
