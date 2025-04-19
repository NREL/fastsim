//! Module for electric machine (i.e. bidirectional electromechanical device), generator, or motor

use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[serde_api]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, StateMethods)]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "pyo3", pyclass(module = "fastsim", subclass, eq))]
/// Struct for modeling electric machines.  This lumps performance and efficiency of motor and power
/// electronics.
pub struct ElectricMachine {
    /// Efficiency interpolator corresponding to achieved output power
    ///
    /// Note that the Extrapolate field of this variable is changed in [Self::get_pwr_in_req]
    pub eff_interp_achieved: Interpolator,
    /// Efficiency interpolator corresponding to max input power
    /// If `None`, will be set during [Self::init].
    ///
    /// Note that the Extrapolate field of this variable is changed in [Self::set_curr_pwr_prop_out_max]
    pub eff_interp_at_max_input: Option<Interpolator>,
    /// Electrical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    // /// this will disappear and instead be in eff_interp_bwd
    // pub pwr_in_frac_interp: Vec<f64>,
    /// ElectricMachine maximum output power \[W\]
    pub pwr_out_max: si::Power,
    /// ElectricMachine specific power
    pub specific_pwr: Option<si::SpecificPower>,
    /// ElectricMachine mass
    pub(in super::super) mass: Option<si::Mass>,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// struct for tracking current state
    #[serde(default)]
    pub state: ElectricMachineState,
    /// Custom vector of [Self::state]
    #[serde(
        default,
        skip_serializing_if = "ElectricMachineStateHistoryVec::is_empty"
    )]
    pub history: ElectricMachineStateHistoryVec,
}

#[named_struct_pyo3_api(ElectricMachine)]
impl ElectricMachine {
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

    #[getter("eff_fwd_max")]
    fn get_eff_max_fwd_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_fwd_max()?)
    }

    #[setter("__eff_fwd_max")]
    fn set_eff_fwd_max_py(&mut self, eff_max: f64) -> PyResult<()> {
        self.set_eff_fwd_max(eff_max)?;
        Ok(())
    }

    #[getter("eff_min_fwd")]
    fn get_eff_min_fwd_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_min_fwd()?)
    }

    #[getter("eff_fwd_range")]
    fn get_eff_fwd_range_py(&self) -> PyResult<f64> {
        Ok(self.get_eff_fwd_range()?)
    }

    #[setter("__eff_fwd_range")]
    fn set_eff_fwd_range_py(&mut self, eff_range: f64) -> PyResult<()> {
        self.set_eff_fwd_range(eff_range)?;
        Ok(())
    }
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
    /// - `dt`: simulation time step size
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

        self.state.eff_fwd_at_max_input.update(
            uc::R
                * self
                    .eff_interp_at_max_input
                    .as_ref()
                    .map(|interpolator| {
                        interpolator
                            .interpolate(&[abs_checked_x_val(
                                (pwr_in_fwd_lim / self.pwr_out_max).get::<si::ratio>(),
                                interpolator.x().map_err(|e| anyhow!(e))?,
                            )?])
                            .map_err(|e| anyhow!(e))
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
                    })?,
            || format_dbg!(),
        )?;
        self.state.eff_at_max_regen.update(
            uc::R
                * self
                    .eff_interp_at_max_input
                    .as_ref()
                    .map(|interpolator| {
                        interpolator
                            .interpolate(&[abs_checked_x_val(
                                (pwr_in_bwd_lim / self.pwr_out_max).get::<si::ratio>(),
                                interpolator.x().map_err(|e| anyhow!(e))?,
                            )?])
                            .map_err(|e| anyhow!(e))
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
                    })?,
            || format_dbg!(),
        )?;

        // maximum power in forward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_fwd_out_max.update(
            self.pwr_out_max.min(
                pwr_in_fwd_lim
                    * *self
                        .state
                        .eff_fwd_at_max_input
                        .get_fresh(|| format_dbg!())?,
            ),
            || format_dbg!(),
        )?;
        // maximum power in backward direction is minimum of component `pwr_out_max` parameter or time-varying max
        // power in bacward direction (i.e. regen) based on what the ReversibleEnergyStorage can provide
        self.state.pwr_mech_regen_max.update(
            self.pwr_out_max
                .min(pwr_in_bwd_lim / *self.state.eff_at_max_regen.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;
        Ok(())
    }

    /// Solves for this powertrain system/component efficiency and sets/returns power input required.
    /// # Arguments
    /// - `pwr_out_req`: propulsion-related power output required
    /// - `dt`: simulation time step size
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
            almost_le_uom(&pwr_out_req , self.state.pwr_mech_fwd_out_max.get_fresh(|| format_dbg!())?, None),
            format!(
                "{}\nedrv required propulsion power ({} kW) exceeds current max propulsion power ({} kW) by {} kW",
                format_dbg!(pwr_out_req <= *self.state.pwr_mech_fwd_out_max.get_fresh(|| format_dbg!())?),
                pwr_out_req.get::<si::kilowatt>().format_eng(Some(6)),
                self.state
                    .pwr_mech_fwd_out_max
                    .get_fresh(|| format_dbg!())?
                    .get::<si::kilowatt>()
                    .format_eng(Some(6)),
                    (pwr_out_req - *self.state.pwr_mech_fwd_out_max.get_fresh(|| format_dbg!())?).get::<si::kilowatt>().format_eng(Some(6))
            ),
        );
        if pwr_out_req < si::Power::ZERO {
            ensure!(
                almost_le_uom(
                    &pwr_out_req.abs(),
                    self.state.pwr_mech_regen_max.get_fresh(|| format_dbg!())?,
                    None
                ),
                format!(
                    "{}\nedrv charge power ({:.6} kW) exceeds current max charge power ({:.6} kW)",
                    format_dbg!(),
                    -pwr_out_req.get::<si::kilowatt>(),
                    self.state
                        .pwr_mech_regen_max
                        .get_fresh(|| format_dbg!())?
                        .get::<si::kilowatt>()
                ),
            );
        }

        self.state
            .pwr_out_req
            .update(pwr_out_req, || format_dbg!())?;

        // ensuring eff_interp_fwd has Extrapolate set to Error before calculating self.state.eff
        self.eff_interp_achieved
            .set_extrapolate(Extrapolate::Error)?;

        self.state.eff.update(
            uc::R
                * self
                    .eff_interp_achieved
                    .interpolate(
                        &[{
                            let pwr = |pwr_uncorrected: f64| -> anyhow::Result<f64> {
                                Ok({
                                    if self
                                        .eff_interp_achieved
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
                    })?,
            || format_dbg!(),
        )?;

        // `pwr_mech_prop_out` is `pwr_out_req` unless `pwr_out_req` is more negative than `pwr_mech_regen_max`,
        // in which case, excess is handled by `pwr_mech_dyn_brake`
        self.state.pwr_mech_prop_out.update(
            pwr_out_req.max(-*self.state.pwr_mech_regen_max.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;

        self.state.pwr_mech_dyn_brake.update(
            -(pwr_out_req - *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?),
            || format_dbg!(),
        )?;
        ensure!(
            *self.state.pwr_mech_dyn_brake.get_fresh(|| format_dbg!())? >= si::Power::ZERO,
            "Mech Dynamic Brake Power cannot be below 0.0"
        );

        // if pwr_out_req is negative, need to multiply by eff
        self.state.pwr_elec_prop_in.update(
            if pwr_out_req > si::Power::ZERO {
                *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                    / *self.state.eff.get_fresh(|| format_dbg!())?
            } else {
                *self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                    * *self.state.eff.get_fresh(|| format_dbg!())?
            },
            || format_dbg!(),
        )?;

        self.state.pwr_elec_dyn_brake.update(
            *self.state.pwr_mech_dyn_brake.get_fresh(|| format_dbg!())?
                * *self.state.eff.get_fresh(|| format_dbg!())?,
            || format_dbg!(),
        )?;

        // loss does not account for dynamic braking
        self.state.pwr_loss.update(
            (*self.state.pwr_mech_prop_out.get_fresh(|| format_dbg!())?
                - *self.state.pwr_elec_prop_in.get_fresh(|| format_dbg!())?)
            .abs(),
            || format_dbg!(),
        )?;

        Ok(*self.state.pwr_elec_prop_in.get_fresh(|| format_dbg!())?)
    }
}

impl SerdeAPI for ElectricMachine {}
impl Init for ElectricMachine {
    fn init(&mut self) -> Result<(), Error> {
        let _ = self
            .mass()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        let _ = check_interp_frac_data(self.eff_interp_achieved.x()?, InterpRange::Either)
            .map_err(|err|
                Error::InitError(format!(
                    "{}\nInvalid values for `ElectricMachine::pwr_out_frac_interp`; must range from [-1..1] or [0..1].",
                    format_dbg!(err)
                )
             ))?;
        self.state
            .init()
            .map_err(|err| Error::InitError(format_dbg!(err)))?;
        // sets eff_interp_bwd to eff_interp_fwd, but changes the x-value.
        // TODO: what should the default strategy be for eff_interp_bwd?
        let eff_interp_at_max_input = Interpolator::new_1d(
            self.eff_interp_achieved
                .x()?
                .iter()
                .zip(self.eff_interp_achieved.f_x()?)
                .map(|(x, y)| x / y)
                .collect(),
            self.eff_interp_achieved.f_x()?.to_owned(),
            // TODO: should these be set to be the same as eff_interp_fwd,
            // as currently is done, or should they be set to be specific
            // Extrapolate and Strategy types?
            self.eff_interp_achieved.strategy()?.to_owned(),
            self.eff_interp_achieved.extrapolate()?.to_owned(),
        )
        .map_err(ninterp::error::Error::from)?;
        self.eff_interp_at_max_input = Some(eff_interp_at_max_input);
        Ok(())
    }
}
impl HistoryMethods for ElectricMachine {
    fn save_interval(&self) -> anyhow::Result<Option<usize>> {
        Ok(self.save_interval)
    }
    fn set_save_interval(&mut self, save_interval: Option<usize>) -> anyhow::Result<()> {
        self.save_interval = save_interval;
        Ok(())
    }
    fn clear(&mut self) {
        self.history.clear();
    }
}

impl SetCumulative for ElectricMachine {
    fn set_cumulative(&mut self, dt: si::Time) -> anyhow::Result<()> {
        self.state.set_cumulative(dt)?;
        Ok(())
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
    pub fn get_eff_fwd_max(&self) -> anyhow::Result<f64> {
        // since efficiency is all f64 between 0 and 1, NEG_INFINITY is safe
        Ok(self
            .eff_interp_achieved
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
    pub fn set_eff_fwd_max(&mut self, eff_max: f64) -> anyhow::Result<()> {
        if (0.0..=1.0).contains(&eff_max) {
            let old_max_fwd = self.get_eff_fwd_max()?;
            let old_max_bwd = self.get_eff_max_bwd()?;
            let f_x_fwd = self.eff_interp_achieved.f_x()?.to_owned();
            match &mut self.eff_interp_achieved {
                interp @ Interpolator::Interp1D(..) => {
                    interp.set_f_x(f_x_fwd.iter().map(|x| x * eff_max / old_max_fwd).collect())?;
                }
                _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
            }
            let f_x_bwd = self
                .eff_interp_at_max_input
                .as_ref()
                .ok_or(anyhow!(
                    "eff_interp_bwd is None, which should never be the case at this point."
                ))?
                .f_x()?
                .to_owned();
            match &mut self.eff_interp_at_max_input {
                Some(interp @ Interpolator::Interp1D(..)) => {
                    interp.set_f_x(
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
            .eff_interp_achieved
            .f_x()
            .with_context(|| "eff_interp_fwd does not have f_x field")?
            .iter()
            .fold(f64::INFINITY, |acc, curr| acc.min(*curr)))
    }

    /// Returns min value of `eff_interp_bwd`
    pub fn get_eff_min_at_max_input(&self) -> anyhow::Result<f64> {
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
    pub fn get_eff_fwd_range(&self) -> anyhow::Result<f64> {
        Ok(self.get_eff_fwd_max()? - self.get_eff_min_fwd()?)
    }

    /// Max value of `eff_interp_bwd` minus min value of `eff_interp_bwd`.
    pub fn get_eff_range_bwd(&self) -> anyhow::Result<f64> {
        Ok(self.get_eff_max_bwd()? - self.get_eff_min_at_max_input()?)
    }

    /// Scales values of `eff_interp_fwd.f_x` and `eff_interp_bwd.f_x` without changing max such that max - min
    /// is equal to new range.  Will change max if needed to ensure no values are
    /// less than zero.
    pub fn set_eff_fwd_range(&mut self, eff_range: f64) -> anyhow::Result<()> {
        let eff_max_fwd = self.get_eff_fwd_max()?;
        let eff_max_bwd = self.get_eff_max_bwd()?;
        if eff_range == 0.0 {
            let f_x_fwd = vec![
                eff_max_fwd;
                self.eff_interp_achieved
                    .f_x()
                    .with_context(|| "eff_interp_fwd does not have f_x field")?
                    .len()
            ];
            self.eff_interp_achieved.set_f_x(f_x_fwd)?;
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
            let old_range = self.get_eff_fwd_max()? - old_min;
            if old_range == 0.0 {
                return Err(anyhow!(
                    "`eff_range` is already zero so it cannot be modified."
                ));
            }
            let f_x_fwd = self.eff_interp_achieved.f_x()?.to_owned();
            match &mut self.eff_interp_achieved {
                interp @ Interpolator::Interp1D(..) => {
                    interp.set_f_x(
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
                let f_x_fwd = self.eff_interp_achieved.f_x()?.to_owned();
                match &mut self.eff_interp_achieved {
                    interp @ Interpolator::Interp1D(..) => {
                        interp.set_f_x(f_x_fwd.iter().map(|x| x - x_neg).collect())?;
                    }
                    _ => bail!("{}\n", "Only `Interpolator::Interp1D` is allowed."),
                }
            }
            if self.get_eff_fwd_max()? > 1.0 {
                return Err(anyhow!(format!(
                    "`eff_max` ({:.3}) must be no greater than 1.0",
                    self.get_eff_fwd_max()?
                )));
            }
            let old_min = self.get_eff_min_at_max_input()?;
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

            if self.get_eff_min_at_max_input()? < 0.0 {
                let x_neg = self.get_eff_min_at_max_input()?;
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

#[serde_api]
#[derive(
    Clone,
    Debug,
    Default,
    Deserialize,
    Serialize,
    PartialEq,
    HistoryVec,
    SetCumulative,
    StateMethods,
)]
#[non_exhaustive]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct ElectricMachineState {
    /// time step index
    pub i: TrackedState<usize>,
    /// Component efficiency based on current power demand.
    pub eff: TrackedState<si::Ratio>,
    // Component limits
    /// Maximum possible positive traction power.
    pub pwr_mech_fwd_out_max: TrackedState<si::Power>,
    /// efficiency in forward direction at max possible input power from `FuelConverter` and `ReversibleEnergyStorage`
    pub eff_fwd_at_max_input: TrackedState<si::Ratio>,
    /// Maximum possible regeneration power going to ReversibleEnergyStorage.
    pub pwr_mech_regen_max: TrackedState<si::Power>,
    /// efficiency in backward direction at max possible input power from `FuelConverter` and `ReversibleEnergyStorage`
    pub eff_at_max_regen: TrackedState<si::Ratio>,

    // Current values
    /// Raw power requirement from boundary conditions
    pub pwr_out_req: TrackedState<si::Power>,
    /// Integral of [Self::pwr_out_req]
    pub energy_out_req: TrackedState<si::Energy>,
    /// Electrical power to propulsion from ReversibleEnergyStorage and Generator.
    /// negative value indicates regenerative braking
    pub pwr_elec_prop_in: TrackedState<si::Power>,
    /// Integral of [Self::pwr_elec_prop_in]
    pub energy_elec_prop_in: TrackedState<si::Energy>,
    /// Mechanical power to propulsion, corrected by efficiency, from ReversibleEnergyStorage and Generator.
    /// Negative value indicates regenerative braking.
    pub pwr_mech_prop_out: TrackedState<si::Power>,
    /// Integral of [Self::pwr_mech_prop_out]
    pub energy_mech_prop_out: TrackedState<si::Energy>,
    /// Mechanical power from dynamic braking.  Positive value indicates braking; this should be zero otherwise.
    pub pwr_mech_dyn_brake: TrackedState<si::Power>,
    /// Integral of [Self::pwr_mech_dyn_brake]
    pub energy_mech_dyn_brake: TrackedState<si::Energy>,
    /// Electrical power from dynamic braking, dissipated as heat.
    pub pwr_elec_dyn_brake: TrackedState<si::Power>,
    /// Integral of [Self::pwr_elec_dyn_brake]
    pub energy_elec_dyn_brake: TrackedState<si::Energy>,
    /// Power lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub pwr_loss: TrackedState<si::Power>,
    /// Integral of [Self::pwr_loss]
    pub energy_loss: TrackedState<si::Energy>,
}

impl Init for ElectricMachineState {}
impl SerdeAPI for ElectricMachineState {}
