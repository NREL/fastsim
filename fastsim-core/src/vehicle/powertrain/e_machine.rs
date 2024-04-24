use super::*;

#[allow(unused_imports)]
#[cfg(feature = "pyo3")]
use crate::pyo3::*;

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
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods, SerdeAPI)]
/// Struct for modeling electric machines.  This lumps performance and efficiency of motor and power
/// electronics.
pub struct ElectricMachine {
    #[serde(default)]
    #[serde(skip_serializing_if = "IsDefault::is_default")]
    /// struct for tracking current state
    pub state: ElectricMachineState,
    /// Shaft output power fraction array at which efficiencies are evaluated.
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

impl SetCumulative for ElectricMachine {
    fn set_cumulative(&mut self, dt: si::Time) {
        self.state.set_cumulative(dt);
    }
}

impl Mass for ElectricMachine {
    fn mass(&self) -> anyhow::Result<si::Mass> {
        let derived_mass = self.get_checked_mass()?;
        Ok(self.mass.unwrap_or(derived_mass.with_context(|| {
            format!(
                "Not all mass fields in `{}` are set and mass field is `None`.",
                stringify!(ElectricMachine)
            )
        })?))
    }

    fn set_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()> {
        self.mass = match mass {
            Some(mass) => {
                self.specific_pwr = Some(self.pwr_out_max / mass);
                Some(mass)
            }
            None => Some(
                self.pwr_out_max
                    / self.specific_pwr.with_context(|| {
                        format!(
                            "{}\n{}",
                            format_dbg!(),
                            "`mass` must be provided, or `self.specific_pwr` must be set"
                        )
                    })?,
            ),
        };
        Ok(())
    }

    fn get_checked_mass(&self) -> anyhow::Result<()> {
        if self.mass.is_some() && self.specific_pwr.is_some() {
            ensure!(
                self.pwr_out_max / self.specific_pwr.unwrap() == self.mass.unwrap(),
                "{}\n{}",
                format_dbg!(),
                "`pwr_out_max`, `specific_pwr`, and `mass` fields are not consistent"
            )
        };
        Ok(())
    }
}

impl ElectricMachine {
    pub fn new(
        pwr_out_frac_interp: Vec<f64>,
        eff_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        specific_pwr_kw_per_kg: Option<f64>,
        mass_kg: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            eff_interp.len() == pwr_out_frac_interp.len(),
            format!(
                "{}\nElectricMachine `eff_interp` and `pwr_out_frac_interp` must be the same length",
                eff_interp.len() == pwr_out_frac_interp.len()
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x >= 0.0),
            format!(
                "{}\nElectricMachine `pwr_out_frac_interp` must be non-negative",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x >= 0.0))
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x <= 1.0),
            format!(
                "{}\nElectricMachine `pwr_out_frac_interp` must be less than or equal to 1.0",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x <= 1.0))
            )
        );

        let state = ElectricMachineState::default();
        let pwr_out_max_watts = uc::W * pwr_out_max_watts;
        let specific_pwr_kw_per_kg =
            specific_pwr_kw_per_kg.map(|specific_pwr| uc::KW / uc::KG * specific_pwr);
        let mass_kg = mass_kg.map(|mass| uc::KG * mass);
        let history = ElectricMachineStateHistoryVec::new();

        let mut e_machine = ElectricMachine {
            state,
            pwr_out_frac_interp,
            eff_interp,
            pwr_in_frac_interp: Vec::new(),
            pwr_out_max: pwr_out_max_watts,
            specific_pwr: specific_pwr_kw_per_kg,
            mass: mass_kg,
            save_interval,
            history,
        };
        e_machine.set_pwr_in_frac_interp()?;
        e_machine.get_checked_mass()?;
        Ok(e_machine)
    }

    pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
        // make sure vector has been created
        self.pwr_in_frac_interp = self
            .pwr_out_frac_interp
            .iter()
            .zip(self.eff_interp.iter())
            .map(|(x, y)| x / y)
            .collect();
        // verify monotonicity
        ensure!(
            self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1]),
            format!(
                "{}\nElectricMachine `pwr_in_frac_interp` ({:?}) must be monotonically increasing",
                format_dbg!(self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1])),
                self.pwr_in_frac_interp
            )
        );
        Ok(())
    }

    pub fn set_cur_pwr_regen_max(&mut self, pwr_max_regen_in: si::Power) -> anyhow::Result<()> {
        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()?;
        }
        let eff = uc::R
            * interp1d(
                &(pwr_max_regen_in / self.pwr_out_max)
                    .get::<si::ratio>()
                    .abs(),
                &self.pwr_out_frac_interp,
                &self.eff_interp,
                Default::default(),
            )?;
        self.state.pwr_mech_regen_max = (pwr_max_regen_in * eff).min(self.pwr_out_max);
        ensure!(self.state.pwr_mech_regen_max >= si::Power::ZERO);
        Ok(())
    }

    /// Set `pwr_in_req` required to achieve desired `pwr_out_req` with time step size `dt`.
    pub fn set_pwr_in_req(&mut self, pwr_out_req: si::Power, dt: si::Time) -> anyhow::Result<()> {
        ensure!(
            pwr_out_req <= self.pwr_out_max,
            format!(
                "{}\ne_machine required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                format_dbg!(pwr_out_req.abs() <= self.pwr_out_max),
                pwr_out_req.get::<si::megawatt>(),
                self.pwr_out_max.get::<si::megawatt>()
            ),
        );

        self.state.pwr_out_req = pwr_out_req;

        self.state.eff = uc::R
            * interp1d(
                &(pwr_out_req / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_out_frac_interp,
                &self.eff_interp,
                Default::default(),
            )?;
        ensure!(
            self.state.eff >= 0.0 * uc::R || self.state.eff <= 1.0 * uc::R,
            format!(
                "{}\ne_machine eff ({}) must be between 0 and 1",
                format_dbg!(self.state.eff >= 0.0 * uc::R || self.state.eff <= 1.0 * uc::R),
                self.state.eff.get::<si::ratio>()
            )
        );

        // `pwr_mech_prop_out` is `pwr_out_req` unless `pwr_out_req` is more negative than `pwr_mech_regen_max`,
        // in which case, excess is handled by `pwr_mech_dyn_brake`
        self.state.pwr_mech_prop_out = pwr_out_req.max(-self.state.pwr_mech_regen_max);
        self.state.energy_mech_prop_out += self.state.pwr_mech_prop_out * dt;

        self.state.pwr_mech_dyn_brake = -(pwr_out_req - self.state.pwr_mech_prop_out);
        self.state.energy_mech_dyn_brake += self.state.pwr_mech_dyn_brake * dt;
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
        self.state.energy_elec_prop_in += self.state.pwr_elec_prop_in * dt;

        self.state.pwr_elec_dyn_brake = self.state.pwr_mech_dyn_brake * self.state.eff;
        self.state.energy_elec_dyn_brake += self.state.pwr_elec_dyn_brake * dt;

        // loss does not account for dynamic braking
        self.state.pwr_loss = (self.state.pwr_mech_prop_out - self.state.pwr_elec_prop_in).abs();
        self.state.energy_loss += self.state.pwr_loss * dt;

        Ok(())
    }

    impl_get_set_eff_max_min!();
    impl_get_set_eff_range!();

    /// Set current max possible output power, `pwr_mech_out_max`,
    /// given `pwr_in_max` from upstream component.
    pub fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        ensure!(pwr_aux.is_none(), format_dbg!(pwr_aux.is_none()));
        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()?;
        }
        let eff = uc::R
            * interp1d(
                &(pwr_in_max / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_in_frac_interp,
                &self.eff_interp,
                Default::default(),
            )?;

        self.state.pwr_mech_out_max = self.pwr_out_max.min(pwr_in_max * eff);
        Ok(())
    }

    /// Set current power out max ramp rate, `pwr_rate_out_max` given `pwr_rate_in_max`
    /// from upstream component.
    pub fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) {
        self.state.pwr_rate_out_max = pwr_rate_in_max
            * if self.state.eff > si::Ratio::ZERO {
                self.state.eff
            } else {
                uc::R * 1.0
            };
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
    pub pwr_mech_out_max: si::Power,
    /// Maximum possible regeneration power going to ReversibleEnergyStorage.
    pub pwr_mech_regen_max: si::Power,
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
impl ElectricMachineState {
    pub fn new() -> Self {
        Self {
            i: 1,
            ..Default::default()
        }
    }
}
