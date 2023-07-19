use crate::consist::locomotive::powertrain::ElectricMachine;
use crate::imports::*;

#[cfg(feature = "pyo3")]
use crate::pyo3::*;

#[altrios_api(
    #[new]
    fn __new__(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            pwr_out_frac_interp,
            eta_interp,
            pwr_out_max_watts,
            save_interval,
        )
    }

    #[setter]
    pub fn set_eta_interp(&mut self, new_value: Vec<f64>) -> anyhow::Result<()> {
        self.eta_interp = new_value;
        self.set_pwr_in_frac_interp()
    }

    #[getter("eta_max")]
    fn get_eta_max_py(&self) -> f64 {
        self.get_eta_max()
    }

    #[setter("__eta_max")]
    fn set_eta_max_py(&mut self, eta_max: f64) -> PyResult<()> {
        self.set_eta_max(eta_max).map_err(PyValueError::new_err)
    }

    #[getter("eta_min")]
    fn get_eta_min_py(&self) -> f64 {
        self.get_eta_min()
    }

    #[getter("eta_range")]
    fn get_eta_range_py(&self) -> f64 {
        self.get_eta_range()
    }

    #[setter("__eta_range")]
    fn set_eta_range_py(&mut self, eta_range: f64) -> PyResult<()> {
        self.set_eta_range(eta_range).map_err(PyValueError::new_err)
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods, SerdeAPI)]
/// Struct for modeling electric drivetrain.  This includes power electronics, motor, axle ...
/// everything involved in converting high voltage electrical power to force exerted by the wheel on the track.  
pub struct ElectricDrivetrain {
    #[serde(default)]
    /// struct for tracking current state
    pub state: ElectricDrivetrainState,
    /// Shaft output power fraction array at which efficiencies are evaluated.
    pub pwr_out_frac_interp: Vec<f64>,
    #[api(skip_set)]
    /// Efficiency array corresponding to [Self::pwr_out_frac_interp] and [Self::pwr_in_frac_interp]
    pub eta_interp: Vec<f64>,
    /// Electrical input power fraction array at which efficiencies are evaluated.
    /// Calculated during runtime if not provided.
    #[serde(skip)]
    #[api(skip_set)]
    pub pwr_in_frac_interp: Vec<f64>,
    /// ElectricDrivetrain maximum output power \[W\]
    #[serde(rename = "pwr_out_max_watts")]
    pub pwr_out_max: si::Power,
    /// Time step interval between saves. 1 is a good option. If None, no saving occurs.
    pub save_interval: Option<usize>,
    /// Custom vector of [Self::state]
    #[serde(default)]
    pub history: ElectricDrivetrainStateHistoryVec,
}

impl ElectricDrivetrain {
    pub fn new(
        pwr_out_frac_interp: Vec<f64>,
        eta_interp: Vec<f64>,
        pwr_out_max_watts: f64,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            eta_interp.len() == pwr_out_frac_interp.len(),
            format!(
                "{}\nedrv eta_interp and pwr_out_frac_interp must be the same length",
                eta_interp.len() == pwr_out_frac_interp.len()
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x >= 0.0),
            format!(
                "{}\nedrv pwr_out_frac_interp must be non-negative",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x >= 0.0))
            )
        );

        ensure!(
            pwr_out_frac_interp.iter().all(|x| *x <= 1.0),
            format!(
                "{}\nedrv pwr_out_frac_interp must be less than or equal to 1.0",
                format_dbg!(pwr_out_frac_interp.iter().all(|x| *x <= 1.0))
            )
        );

        let history = ElectricDrivetrainStateHistoryVec::new();
        let pwr_out_max_watts = uc::W * pwr_out_max_watts;
        let state = ElectricDrivetrainState::default();

        let mut edrv = ElectricDrivetrain {
            state,
            pwr_out_frac_interp,
            eta_interp,
            pwr_in_frac_interp: Vec::new(),
            pwr_out_max: pwr_out_max_watts,
            save_interval,
            history,
        };
        edrv.set_pwr_in_frac_interp()?;
        Ok(edrv)
    }

    pub fn set_pwr_in_frac_interp(&mut self) -> anyhow::Result<()> {
        // make sure vector has been created
        self.pwr_in_frac_interp = self
            .pwr_out_frac_interp
            .iter()
            .zip(self.eta_interp.iter())
            .map(|(x, y)| x / y)
            .collect();
        // verify monotonicity
        ensure!(
            self.pwr_in_frac_interp.windows(2).all(|w| w[0] < w[1]),
            format!(
                "{}\nedrv pwr_in_frac_interp ({:?}) must be monotonically increasing",
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
        let eta = uc::R
            * interp1d(
                &(pwr_max_regen_in / self.pwr_out_max)
                    .get::<si::ratio>()
                    .abs(),
                &self.pwr_out_frac_interp,
                &self.eta_interp,
                false,
            )?;
        self.state.pwr_mech_regen_max = (pwr_max_regen_in * eta).min(self.pwr_out_max);
        ensure!(self.state.pwr_mech_regen_max >= si::Power::ZERO);
        Ok(())
    }

    /// Set `pwr_in_req` required to achieve desired `pwr_out_req` with time step size `dt`.
    pub fn set_pwr_in_req(&mut self, pwr_out_req: si::Power, dt: si::Time) -> anyhow::Result<()> {
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

        self.state.eta = uc::R
            * interp1d(
                &(pwr_out_req / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_out_frac_interp,
                &self.eta_interp,
                false,
            )?;
        ensure!(
            self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R,
            format!(
                "{}\nedrv eta ({}) must be between 0 and 1",
                format_dbg!(self.state.eta >= 0.0 * uc::R || self.state.eta <= 1.0 * uc::R),
                self.state.eta.get::<si::ratio>()
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

        // if pwr_out_req is negative, need to multiply by eta
        self.state.pwr_elec_prop_in = if pwr_out_req > si::Power::ZERO {
            self.state.pwr_mech_prop_out / self.state.eta
        } else {
            self.state.pwr_mech_prop_out * self.state.eta
        };
        self.state.energy_elec_prop_in += self.state.pwr_elec_prop_in * dt;

        self.state.pwr_elec_dyn_brake = self.state.pwr_mech_dyn_brake * self.state.eta;
        self.state.energy_elec_dyn_brake += self.state.pwr_elec_dyn_brake * dt;

        // loss does not account for dynamic braking
        self.state.pwr_loss = (self.state.pwr_mech_prop_out - self.state.pwr_elec_prop_in).abs();
        self.state.energy_loss += self.state.pwr_loss * dt;

        Ok(())
    }

    impl_get_set_eta_max_min!();
    impl_get_set_eta_range!();
}

// failed attempt at making path to default platform independent
// const EDRV_DEFAULT_PATH_STR: &'static str = include_str!(concat!(
//     env!("CARGO_MANIFEST_DIR"),
//     "/src/consist/locomotive/powertrain/electric_drivetrain.default.yaml"
// ));

impl Default for ElectricDrivetrain {
    fn default() -> Self {
        // let file_contents = include_str!(EDRV_DEFAULT_PATH_STR);
        let file_contents = include_str!("electric_drivetrain.default.yaml");
        serde_yaml::from_str::<ElectricDrivetrain>(file_contents).unwrap()
    }
}

impl ElectricMachine for ElectricDrivetrain {
    /// Set current max possible output power, `pwr_mech_out_max`,
    /// given `pwr_in_max` from upstream component.
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_in_max: si::Power,
        pwr_aux: Option<si::Power>,
    ) -> anyhow::Result<()> {
        ensure!(pwr_aux.is_none(), format_dbg!(pwr_aux.is_none()));
        if self.pwr_in_frac_interp.is_empty() {
            self.set_pwr_in_frac_interp()?;
        }
        let eta = uc::R
            * interp1d(
                &(pwr_in_max / self.pwr_out_max).get::<si::ratio>().abs(),
                &self.pwr_in_frac_interp,
                &self.eta_interp,
                false,
            )?;

        self.state.pwr_mech_out_max = self.pwr_out_max.min(pwr_in_max * eta);
        Ok(())
    }

    /// Set current power out max ramp rate, `pwr_rate_out_max` given `pwr_rate_in_max`
    /// from upstream component.  
    fn set_pwr_rate_out_max(&mut self, pwr_rate_in_max: si::PowerRate) {
        self.state.pwr_rate_out_max = pwr_rate_in_max
            * if self.state.eta > si::Ratio::ZERO {
                self.state.eta
            } else {
                uc::R * 1.0
            };
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, HistoryVec)]
#[altrios_api]
pub struct ElectricDrivetrainState {
    /// index
    pub i: usize,
    /// Component efficiency based on current power demand.
    pub eta: si::Ratio,
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
    /// Electrical power to propulsion from ReversibleEnergyStorage and Generator.
    /// negative value indicates regenerative braking
    pub pwr_elec_prop_in: si::Power,
    /// Mechanical power to propulsion, corrected by efficiency, from ReversibleEnergyStorage and Generator.
    /// Negative value indicates regenerative braking.
    pub pwr_mech_prop_out: si::Power,
    /// Mechanical power from dynamic braking.  Positive value indicates braking; this should be zero otherwise.
    pub pwr_mech_dyn_brake: si::Power,
    /// Electrical power from dynamic braking, dissipated as heat.
    pub pwr_elec_dyn_brake: si::Power,
    /// Power lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub pwr_loss: si::Power,

    // Cumulative energy values
    /// cumulative mech energy in from fc
    pub energy_elec_prop_in: si::Energy,
    /// cumulative elec energy out
    pub energy_mech_prop_out: si::Energy,
    /// cumulative energy has lost due to imperfect efficiency
    /// Mechanical energy from dynamic braking.
    pub energy_mech_dyn_brake: si::Energy,
    /// Electrical energy from dynamic braking, dissipated as heat.
    pub energy_elec_dyn_brake: si::Energy,
    /// Cumulative energy lost in regeneratively converting mechanical power to power that can be absorbed by the battery.
    pub energy_loss: si::Energy,
}

impl Default for ElectricDrivetrainState {
    fn default() -> Self {
        Self {
            i: 1,
            eta: Default::default(),
            pwr_out_req: Default::default(),
            pwr_mech_prop_out: Default::default(),
            pwr_elec_prop_in: Default::default(),
            pwr_mech_out_max: Default::default(),
            pwr_mech_regen_max: Default::default(),
            pwr_elec_dyn_brake: Default::default(),
            pwr_mech_dyn_brake: Default::default(),
            pwr_loss: Default::default(),
            pwr_rate_out_max: Default::default(),
            energy_elec_prop_in: Default::default(),
            energy_mech_prop_out: Default::default(),
            energy_elec_dyn_brake: Default::default(),
            energy_mech_dyn_brake: Default::default(),
            energy_loss: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_edrv() -> ElectricDrivetrain {
        ElectricDrivetrain::new(vec![0.0, 1.0], vec![0.9, 0.8], 8e6, None).unwrap()
    }

    #[test]
    fn test_that_i_increments() {
        let mut edrv = test_edrv();
        edrv.step();
        assert_eq!(2, edrv.state.i);
    }

    #[test]
    fn test_that_loss_is_monotonic() {
        let mut edrv = test_edrv();
        edrv.save_interval = Some(1);
        edrv.save_state();
        edrv.set_pwr_in_req(uc::W * 2_000e3, uc::S * 1.0).unwrap();
        edrv.step();
        edrv.save_state();
        edrv.set_pwr_in_req(uc::W * -2_000e3, uc::S * 1.0).unwrap();
        edrv.step();
        edrv.save_state();
        edrv.set_pwr_in_req(uc::W * 1_500e3, uc::S * 1.0).unwrap();
        edrv.step();
        edrv.save_state();
        edrv.set_pwr_in_req(uc::W * -1_500e3, uc::S * 1.0).unwrap();
        edrv.step();
        let energy_loss_j = edrv
            .history
            .energy_loss
            .iter()
            .map(|x| x.get::<si::joule>())
            .collect::<Vec<_>>();
        for i in 1..energy_loss_j.len() {
            assert!(energy_loss_j[i] >= energy_loss_j[i - 1]);
        }
    }

    #[test]
    fn test_that_history_has_len_1() {
        let mut edrv: ElectricDrivetrain = ElectricDrivetrain::default();
        edrv.save_interval = Some(1);
        assert!(edrv.history.is_empty());
        edrv.save_state();
        assert_eq!(1, edrv.history.len());
    }

    #[test]
    fn test_that_history_has_len_0() {
        let mut edrv: ElectricDrivetrain = ElectricDrivetrain::default();
        assert!(edrv.history.is_empty());
        edrv.save_state();
        assert!(edrv.history.is_empty());
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = test_edrv();
        let eta_max = 0.9;
        let eta_min = 0.8;
        let eta_range = 0.1;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
