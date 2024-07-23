//! Module for electric machine (i.e. bidirectional electromechanical device), generator, or motor

use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;
use crate::utils::abs_checked_x_val;

#[pyo3_api(
    // #[new]
    // fn __new__(
    //     pwr_out_frac_interp: Vec<f64>,
    //     eff_interp: Vec<f64>,
    //     pwr_out_max_watts: f64,
    //     save_interval: Option<usize>,
    // ) -> anyhow::Result<Self> {
    //     Self::new(
    //         pwr_out_frac_interp,
    //         eff_interp,
    //         pwr_out_max_watts,
    //         save_interval,
    //     )
    // }

    // #[setter]
    // pub fn set_eff_interp(&mut self, new_value: Vec<f64>) -> anyhow::Result<()> {
    //     self.eff_interp = new_value;
    //     self.set_pwr_in_frac_interp()
    // }

    // #[getter("eff_max")]
    // fn get_eff_max_py(&self) -> f64 {
    //     self.get_eff_max()
    // }

    // #[setter("__eff_max")]
    // fn set_eff_max_py(&mut self, eff_max: f64) -> PyResult<()> {
    //     self.set_eff_max(eff_max).map_err(PyValueError::new_err)
    // }

    // #[getter("eff_min")]
    // fn get_eff_min_py(&self) -> f64 {
    //     self.get_eff_min()
    // }

    // #[getter("eff_range")]
    // fn get_eff_range_py(&self) -> f64 {
    //     self.get_eff_range()
    // }

    // #[setter("__eff_range")]
    // fn set_eff_range_py(&mut self, eff_range: f64) -> PyResult<()> {
    //     self.set_eff_range(eff_range).map_err(PyValueError::new_err)
    // }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling electric machines.  This lumps performance and efficiency of motor and power
/// electronics.
pub struct ElectricMachine {
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    /// struct for tracking current state
    pub state: ElectricMachineState,
    /// Shaft output power fraction array at which efficiencies are evaluated.
    /// This can range from 0 to 1 or -1 to 1, dependending on whether the efficiency is
    /// directionally symmetrical.
    pub pwr_out_frac_interp: Vec<f64>,
    #[api(skip_set)]
    /// Efficiency array corresponding to [Self::pwr_out_frac_interp] and [Self::pwr_in_frac_interp]
    pub eff_interp: Vec<f64>,
    /// Electrical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    #[serde(skip)]
    #[api(skip_set)]
    pub pwr_in_frac_interp: Vec<f64>,
    /// ElectricMachine maximum output power \[W\]
    #[serde(rename = "pwr_out_max_watts")]
    pub pwr_out_max: si::Power,
    /// ElectricMachine specific power
    // TODO: fix `extract_type_from_option` to allow for not having this line
    #[api(skip_get, skip_set)]
    pub specific_pwr: Option<si::SpecificPower>,
    /// ElectricMachine mass
    // TODO: fix `extract_type_from_option` to allow for not having this line
    #[api(skip_get, skip_set)]
    pub(in super::super) mass: Option<si::Mass>,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    #[serde(skip_serializing_if = "ElectricMachineStateHistoryVec::is_empty")]
    pub history: ElectricMachineStateHistoryVec,
}

impl ElectricMachine {
    /// Returns maximum possible positive and negative propulsion-related powers
    /// this component/system can produce, accounting for any aux-related power
    /// required.
    /// # Arguments
    /// - `pwr_in_fwd_max`: positive-propulsion-related power available to this
    /// component. Positive values indicate that the upstream component can supply
    /// positive tractive power.
    /// - `pwr_in_bwd_max`: negative-propulsion-related power available to this
    /// component. Zero means no power can be sent to upstream compnents and positive
    /// values indicate upstream components can absorb energy.
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    pub fn set_cur_pwr_prop_out_max(
        &mut self,
        pwr_in_fwd_max: si::Power,
        pwr_in_bwd_max: si::Power,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            pwr_in_fwd_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero for `{}`",
            stringify!(pwr_in_fwd_max),
            pwr_in_fwd_max.get::<si::watt>().format_eng(None),
            stringify!(ElectricMachine::get_cur_pwr_tract_out_max)
        );
        ensure!(
            pwr_in_bwd_max >= uc::W * 0.,
            "`{}` ({} W) must be greater than or equal to zero for `{}`",
            stringify!(pwr_in_bwd_max),
            pwr_in_bwd_max.get::<si::watt>().format_eng(None),
            stringify!(ElectricMachine::get_cur_pwr_tract_out_max)
        );

        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()
                .with_context(|| format_dbg!())?;
        }
        let eff_pos = uc::R
            * interp1d(
                &abs_checked_x_val(
                    (pwr_in_fwd_max / self.pwr_out_max).get::<si::ratio>(),
                    &self.pwr_in_frac_interp,
                )?,
                &self.pwr_in_frac_interp,
                &self.eff_interp,
                // Extrapolation does not trigger error because `self.pwr_out_max` is limiting regardless
                Extrapolate::No,
            )
            .with_context(|| {
                anyhow!(
                    "{}\n failed to calculate {}",
                    format_dbg!(),
                    stringify!(eff_pos)
                )
            })?;
        // TODO: scrutinize this variable assignment
        let eff_neg = uc::R
            * interp1d(
                &abs_checked_x_val(
                    (pwr_in_bwd_max / self.pwr_out_max).get::<si::ratio>(),
                    &self.pwr_in_frac_interp,
                )?,
                &self.pwr_in_frac_interp,
                &self.eff_interp,
                // Extrapolation does not trigger error because `self.pwr_out_max` is limiting regardless
                Extrapolate::No,
            )
            .with_context(|| {
                anyhow!(
                    "{}\n failed to calculate {}",
                    format_dbg!(),
                    stringify!(eff_neg)
                )
            })?;

        // maximum power in forward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_fwd_out_max = self.pwr_out_max.min(pwr_in_fwd_max * eff_pos);
        // maximum power in backward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power in bacward direction (i.e. regen) based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_bwd_out_max = self.pwr_out_max.min(pwr_in_bwd_max / eff_neg);
        Ok(())
    }

    /// Solves for this powertrain system/component efficiency and sets/returns power input required.
    /// # Arguments
    /// - `pwr_out_req`: propulsion-related power output required
    /// - `dt`: time step size
    pub fn get_pwr_in_req(
        &mut self,
        pwr_out_req: si::Power,
        _dt: si::Time,
    ) -> anyhow::Result<si::Power> {
        //TODO: update this function to use `pwr_mech_regen_out_max`
        ensure!(
            pwr_out_req <= self.pwr_out_max,
            format!(
                "{}\nedrv required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_out_req.abs() <= self.pwr_out_max),
                pwr_out_req.get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        self.state.pwr_out_req = pwr_out_req;

        self.state.eff = uc::R
            * interp1d(
                {
                    let pwr = |pwr_uncorrected: f64| -> anyhow::Result<f64> {
                        Ok({
                            if self
                                .pwr_out_frac_interp
                                .first()
                                .with_context(|| anyhow!(format_dbg!()))?
                                >= &0.
                            {
                                pwr_uncorrected.max(0.)
                            } else {
                                pwr_uncorrected
                            }
                        })
                    };
                    &pwr((pwr_out_req / self.pwr_out_max).get::<si::ratio>())?
                },
                &self.pwr_out_frac_interp,
                &self.eff_interp,
                Extrapolate::Error,
            )
            .with_context(|| {
                anyhow!(
                    "{}\n failed to calculate {}",
                    format_dbg!(),
                    stringify!(self.state.eff)
                )
            })?;

        // `pwr_mech_prop_out` is `pwr_out_req` unless `pwr_out_req` is more negative than `pwr_mech_regen_max`,
        // in which case, excess is handled by `pwr_mech_dyn_brake`
        self.state.pwr_mech_prop_out = pwr_out_req.max(-self.state.pwr_mech_bwd_out_max);

        self.state.pwr_mech_dyn_brake = -(pwr_out_req - self.state.pwr_mech_prop_out);
        ensure!(
            self.state.pwr_mech_dyn_brake >= si::Power::ZERO,
            "Mech Dynamic Brake Power cannot be below 0.0"
        );

        // if pwr_out_req is negative, need to multiply by eff
        self.state.pwr_elec_prop_in = if pwr_out_req > si::Power::ZERO {
            self.state.pwr_mech_prop_out / self.state.eff
        } else {
            self.state.pwr_mech_prop_out * self.state.eff
        };

        self.state.pwr_elec_dyn_brake = self.state.pwr_mech_dyn_brake * self.state.eff;

        // loss does not account for dynamic braking
        self.state.pwr_loss = (self.state.pwr_mech_prop_out - self.state.pwr_elec_prop_in).abs();

        Ok(self.state.pwr_elec_prop_in)
    }

    pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
        // make sure vector has been created
        self.pwr_in_frac_interp = self
            .pwr_out_frac_interp
            .iter()
            .zip(self.eff_interp.iter())
            .map(|(x, y)| x / y)
            .collect();
        Ok(())
    }

    impl_get_set_eff_max_min!();
    impl_get_set_eff_range!();
}

impl SerdeAPI for ElectricMachine {}
impl Init for ElectricMachine {
    fn init(&mut self) -> anyhow::Result<()> {
        let _ = self.mass().with_context(|| anyhow!(format_dbg!()))?;
        let _ = check_interp_frac_data(&self.pwr_out_frac_interp, InterpRange::Either)
            .with_context(|| 
                "Invalid values for `ElectricMachine::pwr_out_frac_interp`; must range from [-1..1] or [0..1].")?;
        self.state.init().with_context(|| anyhow!(format_dbg!()))?;
        // TODO: make use of `use fastsim_2::params::{LARGE_BASELINE_EFF, LARGE_MOTOR_POWER_KW, SMALL_BASELINE_EFF,SMALL_MOTOR_POWER_KW};`
        // to set
        // if let None = self.pwr_out_frac_interp {
        //     self.pwr_out_frac_interp =
        // }
        // TODO: verify that `pwr_in_frac_interp` is set somewhere and if it is, maybe move it to here???
        Ok(())
    }
}

impl SetCumulative for ElectricMachine {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

impl Mass for ElectricMachine {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        if let (Some(derived_mass), Some(set_mass)) = (derived_mass, self.mass) {
            ensure!(
                utils::almost_eq_uom(&set_mass, &derived_mass, None),
                format!(
                    "{}",
                    format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                )
            );
        }
        Ok(self.mass)
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                log::info!(
                    "Derived mass from `self.specific_pwr` and `self.pwr_out_max` does not match {}",
                    "provided mass. Updating based on `side_effect`"
                );
                match side_effect {
                    MassSideEffect::Extensive => {
                        self.pwr_out_max = self.specific_pwr.with_context(|| {
                            format!(
                                "{}\nExpected `self.specific_pwr` to be `Some`.",
                                format_dbg!()
                            )
                        })? * new_mass;
                    }
                    MassSideEffect::Intensive => {
                        self.specific_pwr = Some(self.pwr_out_max / new_mass);
                    }
                    MassSideEffect::None => {
                        self.specific_pwr = None;
                    }
                }
            }
        } else if new_mass.is_none() {
            log::debug!("Provided mass is None, setting `self.specific_pwr` to None");
            self.specific_pwr = None;
        }
        self.mass = new_mass;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        Ok(self
            .specific_pwr
            .map(|specific_pwr| self.pwr_out_max / specific_pwr))
    }

    fn expunge_mass_fields(&mut self) {
        self.specific_pwr = None;
        self.mass = None;
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative,
)]
#[pyo3_api]
pub struct ElectricMachineState {
    /// time step index
    pub i: usize,
    /// Component efficiency based on current power demand.
    pub eff: si::Ratio,
    // Component limits
    /// Maximum possible positive traction power.
    pub pwr_mech_fwd_out_max: si::Power,
    /// Maximum possible regeneration power going to ReversibleEnergyStorage.
    pub pwr_mech_bwd_out_max: si::Power,
    /// max ramp-up rate
    pub pwr_rate_out_max: si::PowerRate,

    // Current values
    /// Raw power requirement from boundary conditions
    pub pwr_out_req: si::Power,
    /// Integral of [Self::pwr_out_req]
    pub energy_out_req: si::Energy,
    /// Electrical power to propulsion from ReversibleEnergyStorage and Generator.
    /// negative value indicates regenerative braking
    pub pwr_elec_prop_in: si::Power,
    /// Integral of [Self::pwr_elec_prop_in]
    pub energy_elec_prop_in: si::Energy,
    /// Mechanical power to propulsion, corrected by efficiency, from ReversibleEnergyStorage and Generator.
    /// Negative value indicates regenerative braking.
    pub pwr_mech_prop_out: si::Power,
    /// Integral of [Self::pwr_mech_prop_out]
    pub energy_mech_prop_out: si::Energy,
    /// Mechanical power from dynamic braking.  Positive value indicates braking; this should be zero otherwise.
    pub pwr_mech_dyn_brake: si::Power,
    /// Integral of [Self::pwr_mech_dyn_brake]
    pub energy_mech_dyn_brake: si::Energy,
    /// Electrical power from dynamic braking, dissipated as heat.
    pub pwr_elec_dyn_brake: si::Power,
    /// Integral of [Self::pwr_elec_dyn_brake]
    pub energy_elec_dyn_brake: si::Energy,
    /// Power lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub pwr_loss: si::Power,
    /// Integral of [Self::pwr_loss]
    pub energy_loss: si::Energy,
}

impl Init for ElectricMachineState {}
impl SerdeAPI for ElectricMachineState {}
