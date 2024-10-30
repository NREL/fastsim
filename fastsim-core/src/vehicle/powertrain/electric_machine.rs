//! Module for electric machine (i.e. bidirectional electromechanical device), generator, or motor

use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[fastsim_api(
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
    /// Shaft output power fraction array at which efficiencies are evaluated.
    /// This can range from 0 to 1 or -1 to 1, dependending on whether the efficiency is
    /// directionally symmetrical.
    // /// this is x-data that will how be in eff_interp_fwd
    // pub pwr_out_frac_interp: Vec<f64>,
    #[api(skip_set, skip_get)]
    /// Efficiency array corresponding to [Self::pwr_out_frac_interp] and [Self::pwr_in_frac_interp]
    /// eff_interp_fwd and eff_interp_bwd have the same f_x but different x
    /// note that the Extrapolate field of this variable is changed in get_pwr_in_req()
    pub eff_interp_fwd: utils::interp::Interpolator,
    #[serde(skip)]
    #[api(skip_set, skip_get)]
    /// if it is not provided, will be set during init
    /// note that the Extrapolate field of this variable is changed in set_cur_pwr_prop_out_max()
    pub eff_interp_at_max_input: Option<utils::interp::Interpolator>,
    /// Electrical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    // /// this will disappear and instead be in eff_interp_bwd
    // pub pwr_in_frac_interp: Vec<f64>,
    /// ElectricMachine maximum output power \[W\]
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
    /// struct for tracking current state
    #[serde(default)]
    #[serde(skip_serializing_if = "EqDefault::eq_default")]
    pub state: ElectricMachineState,
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
    /// - `pwr_in_fwd_lim`: positive-propulsion-related power available to this
    ///    component. Positive values indicate that the upstream component can supply
    ///    positive tractive power.
    /// - `pwr_in_bwd_lim`: negative-propulsion-related power available to this
    ///     component. Zero means no power can be sent to upstream compnents and positive
    ///     values indicate upstream components can absorb energy.
    /// - `pwr_aux`: aux-related power required from this component
    /// - `dt`: time step size
    pub fn set_curr_pwr_prop_out_max(
        &mut self,
        pwr_in_fwd_lim: si::Power,
        pwr_in_bwd_lim: si::Power,
        _dt: si::Time,
    ) -> anyhow::Result<()> {
        ensure!(
            pwr_in_fwd_lim >= si::Power::ZERO,
            "`{}` ({} W) must be greater than or equal to zero for `{}`",
            stringify!(pwr_in_fwd_lim),
            pwr_in_fwd_lim.get::<si::watt>().format_eng(None),
            stringify!(ElectricMachine::get_curr_pwr_prop_out_max)
        );
        ensure!(
            pwr_in_bwd_lim >= si::Power::ZERO,
            "`{}` ({} W) must be greater than or equal to zero for `{}`",
            stringify!(pwr_in_bwd_lim),
            pwr_in_bwd_lim.get::<si::watt>().format_eng(None),
            stringify!(ElectricMachine::get_curr_pwr_prop_out_max)
        );

        // ensuring Extrapolate is Clamp in preparation for calculating eff_pos

        self.eff_interp_at_max_input
            .as_mut()
            .with_context(|| {
                "eff_interp_bwd is None, which should never be the case at this point."
            })?
            .set_extrapolate(Extrapolate::Clamp)?;

        // TODO: make sure `fwd` and `bwd` are clearly documented somewhere
        self.state.eff_fwd_at_max_input = uc::R
            * self
                .eff_interp_at_max_input
                .as_ref()
                .map(|interpolator| {
                    interpolator.interpolate(&[abs_checked_x_val(
                        (pwr_in_fwd_lim / self.pwr_out_max).get::<si::ratio>(),
                        &interpolator.x()?,
                    )?])
                })
                .ok_or(anyhow!(
                    "eff_interp_bwd is None, which should never be the case at this point."
                ))?
                .with_context(|| {
                    anyhow!(
                        "{}\n failed to calculate {}",
                        format_dbg!(),
                        stringify!(eff_pos)
                    )
                })?;
        self.state.eff_bwd_at_max_input = uc::R
            * self
                .eff_interp_at_max_input
                .as_ref()
                .map(|interpolator| {
                    interpolator.interpolate(&[abs_checked_x_val(
                        (pwr_in_bwd_lim / self.pwr_out_max).get::<si::ratio>(),
                        &interpolator.x()?,
                    )?])
                })
                .ok_or(anyhow!(
                    "eff_interp_bwd is None, which should never be the case at this point."
                ))?
                .with_context(|| {
                    anyhow!(
                        "{}\n failed to calculate {}",
                        format_dbg!(),
                        stringify!(eff_neg)
                    )
                })?;

        // maximum power in forward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_fwd_out_max = self
            .pwr_out_max
            .min(pwr_in_fwd_lim * self.state.eff_fwd_at_max_input);
        // maximum power in backward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power in bacward direction (i.e. regen) based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_bwd_out_max = self
            .pwr_out_max
            .min(pwr_in_bwd_lim / self.state.eff_bwd_at_max_input);
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
            pwr_out_req.abs() <= self.pwr_out_max,
            format!(
                "{}\nedrv required power ({} kW) exceeds static max power ({} kW)",
                format_dbg!(pwr_out_req.abs() <= self.pwr_out_max),
                pwr_out_req.get::<si::kilowatt>().format_eng(Some(9)),
                self.pwr_out_max.get::<si::kilowatt>().format_eng(Some(9))
            ),
        );
        ensure!(
            almost_le_uom(&pwr_out_req , &self.state.pwr_mech_fwd_out_max, None),
            format!(
                "{}\nedrv required propulsion power ({} kW) exceeds current max propulsion power ({} kW) by {} kW",
                format_dbg!(pwr_out_req <= self.state.pwr_mech_fwd_out_max),
                pwr_out_req.get::<si::kilowatt>().format_eng(Some(6)),
                self.state
                    .pwr_mech_fwd_out_max
                    .get::<si::kilowatt>()
                    .format_eng(Some(6)),
                    (pwr_out_req - self.state.pwr_mech_fwd_out_max).get::<si::kilowatt>().format_eng(Some(6))
            ),
        );
        ensure!(
            -pwr_out_req <= self.state.pwr_mech_bwd_out_max,
            format!(
                "{}\nedrv required charge power ({:.6} kW) exceeds current max charge power ({:.6} kW)",
                format_dbg!(pwr_out_req <= self.state.pwr_mech_bwd_out_max),
                pwr_out_req.get::<si::kilowatt>(),
                self.state.pwr_mech_bwd_out_max.get::<si::kilowatt>()
            ),
        );

        self.state.pwr_out_req = pwr_out_req;

        // ensuring eff_interp_fwd has Extrapolate set to Error before calculating self.state.eff
        self.eff_interp_fwd.set_extrapolate(Extrapolate::Error)?;

        self.state.eff = uc::R
            * self
                .eff_interp_fwd
                .interpolate(
                    &[{
                        let pwr = |pwr_uncorrected: f64| -> anyhow::Result<f64> {
                            Ok({
                                if self
                                    .eff_interp_fwd
                                    .x()?
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
                        pwr((pwr_out_req / self.pwr_out_max).get::<si::ratio>())?
                    }], // &self.eff_interp_fwd.x()?,
                        // &self.eff_interp_fwd,
                        // Extrapolate::Error,
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

    // pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
    //     // make sure vector has been created
    //     self.eff_interp_bwd.set_x(
    //         self.eff_interp_fwd
    //
    //             .x()?
    //             .iter()
    //             .zip(self.eff_interp_fwd.0.f_x()?.iter())
    //             .map(|(x, y)| x / y)
    //             .collect(),
    //     );
    //     Ok(())
    // }
}

impl SerdeAPI for ElectricMachine {}
impl Init for ElectricMachine {
    fn init(&mut self) -> anyhow::Result<()> {
        let _ = self.mass().with_context(|| anyhow!(format_dbg!()))?;
        let _ = check_interp_frac_data(&self.eff_interp_fwd.x()?, InterpRange::Either)
            .with_context(||
                "Invalid values for `ElectricMachine::pwr_out_frac_interp`; must range from [-1..1] or [0..1].")?;
        self.state.init().with_context(|| anyhow!(format_dbg!()))?;
        // TODO: make use of `use fastsim_2::params::{LARGE_BASELINE_EFF, LARGE_MOTOR_POWER_KW, SMALL_BASELINE_EFF,SMALL_MOTOR_POWER_KW};`
        // to set
        // if let None = self.pwr_out_frac_interp {
        //     self.pwr_out_frac_interp =
        // }
        // TODO: verify that `pwr_in_frac_interp` is set somewhere and if it is, maybe move it to here???
        if self.eff_interp_at_max_input.is_none() {
            // sets eff_interp_bwd to eff_interp_fwd, but changes the x-value.
            // TODO: what should the default strategy be for eff_interp_bwd?
            let eff_interp_bwd_new = Interp1D::new(
                self.eff_interp_fwd
                    .x()?
                    .iter()
                    .zip(self.eff_interp_fwd.f_x()?.iter())
                    .map(|(x, y)| x / y)
                    .collect(),
                self.eff_interp_fwd.f_x()?,
                // TODO: should these be set to be the same as eff_interp_fwd,
                // as currently is done, or should they be set to be specific
                // Extrapolate and Strategy types?
                self.eff_interp_fwd.strategy()?,
                self.eff_interp_fwd.extrapolate()?,
            )?;
            self.eff_interp_at_max_input =
                Some(utils::interp::Interpolator::Interp1D(eff_interp_bwd_new));
        }
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

impl ElectricMachine {
    /// Returns max value of `eff_interp_fwd`
    pub fn get_eff_max_fwd(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_fwd
            .f_x()?
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)))
    }

    /// Returns max value of `eff_interp_bwd`
    pub fn get_eff_max_bwd(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(match self.eff_interp_at_max_input.as_ref() {
            Some(interp) => interp
                .f_x()?
                .iter()
                .fold(f64::NEG_INFINITY, |acc, curr| acc.max(*curr)),
            None => bail!("eff_interp_bwd should be Some by this point."),
        })
    }

    /// Scales eff_interp_fwd and eff_interp_bwd by ratio of new `eff_max` per current calculated max
    pub fn set_eff_max(&mut self, eff_max: f64) -> anyhow::Result<()> {
        if (0.0..=1.0).contains(&eff_max) {
            let old_max_fwd = self.get_eff_max_fwd()?;
            let old_max_bwd = self.get_eff_max_bwd()?;
            let f_x_fwd = self.eff_interp_fwd.f_x()?;
            match &mut self.eff_interp_fwd {
                Interpolator::Interp1D(interp1d) => {
                    interp1d
                        .set_f_x(f_x_fwd.iter().map(|x| x * eff_max / old_max_fwd).collect())?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
            let f_x_bwd = self
                .eff_interp_at_max_input
                .as_ref()
                .ok_or(anyhow!(
                    "eff_interp_bwd is None, which should never be the case at this point."
                ))?
                .f_x()?;
            match &mut self.eff_interp_at_max_input {
                Some(Interpolator::Interp1D(interp1d)) => {
                    // let old_interp = interp1d;
                    interp1d.set_f_x(
                        f_x_bwd
                            .iter()
                            .map(|x| x * eff_max / old_max_bwd)
                            .collect(),
                    )?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed. eff_interp_bwd should be Some by this point."),
            }
            Ok(())
        } else {
            Err(anyhow!(
                "`eff_max` ({:.3}) must be between 0.0 and 1.0",
                eff_max,
            ))
        }
    }

    /// Returns min value of `eff_interp_fwd`
    pub fn get_eff_min_fwd(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_fwd
            .f_x()
            .with_context(|| "eff_interp_fwd does not have f_x field")?
            .iter()
            .fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }

    /// Returns min value of `eff_interp_bwd`
    pub fn get_eff_min_bwd(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_at_max_input
            .as_ref()
            .ok_or(anyhow!("eff_interp_bwd should be Some by this point."))?
            .f_x()
            .with_context(|| "eff_interp_bwd does not have f_x field")?
            .iter()
            .fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }

    /// Max value of `eff_interp_fwd` minus min value of `eff_interp_fwd`.
    pub fn get_eff_range_fwd(&self) -> anyhow::Result<f64> {
        Ok(self.get_eff_max_fwd()? - self.get_eff_min_fwd()?)
    }

    /// Max value of `eff_interp_bwd` minus min value of `eff_interp_bwd`.
    pub fn get_eff_range_bwd(&self) -> anyhow::Result<f64> {
        Ok(self.get_eff_max_bwd()? - self.get_eff_min_bwd()?)
    }

    /// Scales values of `eff_interp_fwd.f_x` and `eff_interp_bwd.f_x` without changing max such that max - min
    /// is equal to new range.  Will change max if needed to ensure no values are
    /// less than zero.
    pub fn set_eff_range(&mut self, eff_range: f64) -> anyhow::Result<()> {
        let eff_max_fwd = self.get_eff_max_fwd()?;
        let eff_max_bwd = self.get_eff_max_bwd()?;
        if eff_range == 0.0 {
            let f_x_fwd = vec![
                eff_max_fwd;
                self.eff_interp_fwd
                    .f_x()
                    .with_context(|| "eff_interp_fwd does not have f_x field")?
                    .len()
            ];
            self.eff_interp_fwd.set_f_x(f_x_fwd)?;
            let f_x_bwd = vec![
                eff_max_bwd;
                match &self.eff_interp_at_max_input {
                    Some(interp) => {
                        interp
                            .f_x()
                            .with_context(|| "eff_interp_bwd does not have f_x field")?
                            .len()
                    }
                    None => bail!("eff_interp_bwd should be Some by this point."),
                }
            ];
            self.eff_interp_at_max_input
                .as_mut()
                .map(|interpolator| interpolator.set_f_x(f_x_bwd))
                .transpose()?;
            Ok(())
        } else if (0.0..=1.0).contains(&eff_range) {
            let old_min = self.get_eff_min_fwd()?;
            let old_range = self.get_eff_max_fwd()? - old_min;
            if old_range == 0.0 {
                return Err(anyhow!(
                    "`eff_range` is already zero so it cannot be modified."
                ));
            }
            let f_x_fwd = self.eff_interp_fwd.f_x()?;
            match &mut self.eff_interp_fwd {
                Interpolator::Interp1D(interp1d) => {
                    interp1d.set_f_x(
                        f_x_fwd
                            .iter()
                            .map(|x| eff_max_fwd + (x - eff_max_fwd) * eff_range / old_range)
                            .collect(),
                    )?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
            if self.get_eff_min_fwd()? < 0.0 {
                let x_neg = self.get_eff_min_fwd()?;
                let f_x_fwd = self.eff_interp_fwd.f_x()?;
                match &mut self.eff_interp_fwd {
                    Interpolator::Interp1D(interp1d) => {
                        interp1d.set_f_x(f_x_fwd.iter().map(|x| x - x_neg).collect())?;
                    }
                    _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
                }
            }
            if self.get_eff_max_fwd()? > 1.0 {
                return Err(anyhow!(format!(
                    "`eff_max` ({:.3}) must be no greater than 1.0",
                    self.get_eff_max_fwd()?
                )));
            }
            let old_min = self.get_eff_min_bwd()?;
            let old_range = self.get_eff_max_bwd()? - old_min;
            if old_range == 0.0 {
                return Err(anyhow!(
                    "`eff_range` is already zero so it cannot be modified."
                ));
            }

            let new_f_x: Vec<f64> = self
                .eff_interp_at_max_input
                .as_ref()
                .ok_or(anyhow!("eff_interp_bwd should be Some by this point."))?
                .f_x()?
                .iter()
                .map(|x| eff_max_bwd + (x - eff_max_bwd) * eff_range / old_range)
                .collect();

            self.eff_interp_at_max_input
                .as_mut()
                .map(|interpolator| interpolator.set_f_x(new_f_x))
                .transpose()?;

            if self.get_eff_min_bwd()? < 0.0 {
                let x_neg = self.get_eff_min_bwd()?;
                let new_f_x: Vec<f64> = self
                    .eff_interp_at_max_input
                    .as_ref()
                    .ok_or(anyhow!("eff_interp_bwd should be Some by this point."))?
                    .f_x()?
                    .iter()
                    .map(|x| x - x_neg)
                    .collect();
                self.eff_interp_at_max_input
                    .as_mut()
                    .map(|interpolator| interpolator.set_f_x(new_f_x))
                    .transpose()?;
            }
            if self.get_eff_max_bwd()? > 1.0 {
                return Err(anyhow!(format!(
                    "`eff_max` ({:.3}) must be no greater than 1.0",
                    self.get_eff_max_bwd()?
                )));
            }
            Ok(())
        } else {
            Err(anyhow!(format!(
                "`eff_range` ({:.3}) must be between 0.0 and 1.0",
                eff_range,
            )))
        }
    }
}

#[fastsim_api]
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, HistoryVec, SetCumulative,
)]
pub struct ElectricMachineState {
    /// time step index
    pub i: usize,
    /// Component efficiency based on current power demand.
    pub eff: si::Ratio,
    // Component limits
    /// Maximum possible positive traction power.
    pub pwr_mech_fwd_out_max: si::Power,
    /// efficiency in forward direction at max possible input power from `FuelConverter` and `ReversibleEnergyStorage`
    pub eff_fwd_at_max_input: si::Ratio,
    /// Maximum possible regeneration power going to ReversibleEnergyStorage.
    pub pwr_mech_bwd_out_max: si::Power,
    /// efficiency in backward direction at max possible input power from `FuelConverter` and `ReversibleEnergyStorage`
    pub eff_bwd_at_max_input: si::Ratio,
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
