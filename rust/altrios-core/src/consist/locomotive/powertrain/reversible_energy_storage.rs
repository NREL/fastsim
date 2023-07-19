use super::*;

#[cfg(feature = "pyo3")]
use crate::pyo3::*;

const TOL: f64 = 1e-3;

#[altrios_api(
   #[allow(clippy::too_many_arguments)]
    #[new]
    fn __new__(
        temperature_interp_grid: Vec<f64>,
        soc_interp_grid: Vec<f64>,
        c_rate_interp_grid: Vec<f64>,
        eta_interp_values: Vec<Vec<Vec<f64>>>,
        pwr_out_max_watts: f64,
        energy_capacity_joules: f64,
        min_soc: f64,
        max_soc: f64,
        initial_soc: f64,
        initial_temperature_celcius: f64,
        soc_hi_ramp_start: Option<f64>,
        soc_lo_ramp_start: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::new(
            temperature_interp_grid,
            soc_interp_grid,
            c_rate_interp_grid,
            eta_interp_values,
            pwr_out_max_watts,
            energy_capacity_joules,
            min_soc,
            max_soc,
            initial_soc,
            initial_temperature_celcius,
            soc_hi_ramp_start,
            soc_lo_ramp_start,
            save_interval,
        )
    }

    /// pyo3 getter for soc_lo_ramp_start
    #[getter]
    pub fn get_soc_lo_ramp_start(&self) -> PyResult<f64> {
        Ok(self.soc_lo_ramp_start.unwrap().get::<si::ratio>())
    }
    /// pyo3 setter for soc_lo_ramp_start
    #[setter]
    pub fn set_soc_lo_ramp_start(&mut self, new_value: f64) -> PyResult<()> {
        self.soc_lo_ramp_start = Some(new_value * uc::R);
        Ok(())
    }
    /// pyo3 getter for soc_hi_ramp_start
    #[getter]
    pub fn get_soc_hi_ramp_start(&self) -> PyResult<f64> {
        Ok(self.soc_hi_ramp_start.unwrap().get::<si::ratio>())
    }
    /// pyo3 setter for soc_hi_ramp_start
    #[setter]
    pub fn set_soc_hi_ramp_start(&mut self, new_value: f64) -> PyResult<()> {
        self.soc_hi_ramp_start = Some(new_value * uc::R);
        Ok(())
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
    fn set_eta_range_py(&mut self, eta_range: f64) -> anyhow::Result<()> {
        self.set_eta_range(eta_range)
    }

    #[setter("__mass_kg")]
    fn update_mass_py(&mut self, mass_kg: Option<f64>) -> anyhow::Result<()> {
        self.update_mass(mass_kg.map(|m| m * uc::KG))?;
        Ok(())
    }

    #[getter("mass_kg")]
    fn get_mass_kg_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.mass()?.map(|m| m.get::<si::kilogram>()))
    }

    #[getter]
    fn get_specific_energy_kjoules_per_kg(&self) -> Option<f64> {
        self.specific_energy.map(|se| se.get::<si::kilojoule_per_kilogram>())
    }

    #[setter("__volume_m3")]
    fn update_volume_py(&mut self, volume_m3: Option<f64>) -> anyhow::Result<()> {
        let volume = volume_m3.map(|v| v * uc::M3);
        self.update_volume(volume)?;
        Ok(())
    }

    #[getter("volume_m3")]
    fn get_volume_py(&mut self) -> anyhow::Result<Option<f64>> {
        Ok(self.volume()?.map(|v| v.get::<si::cubic_meter>()))
    }

    #[getter]
    fn get_energy_density_kjoules_per_m3(&self) -> Option<f64> {
        self.specific_energy.map(|se| se.get::<si::kilojoule_per_kilogram>())
    }
)]
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, HistoryMethods)]
/// Struct for modeling technology-naive Reversible Energy Storage (e.g. battery, flywheel).
pub struct ReversibleEnergyStorage {
    /// struct for tracking current state
    #[serde(default)]
    pub state: ReversibleEnergyStorageState,
    /// ReversibleEnergyStorage mass
    #[serde(default)]
    #[api(skip_get, skip_set)]
    mass: Option<si::Mass>,
    /// ReversibleEnergyStorage volume, used as a sanity check
    #[api(skip_get, skip_set)]
    #[serde(default)]
    volume: Option<si::Volume>,
    /// ReversibleEnergyStorage specific energy
    #[api(skip_get, skip_set)]
    specific_energy: Option<si::AvailableEnergy>,
    /// ReversibleEnergyStorage energy density (note that pressure has the same units as energy density)
    #[api(skip_get, skip_set)]
    pub energy_density: Option<si::Pressure>,
    /// efficiency map grid values - indexed temp; soc; c_rate;
    pub eta_interp_grid: [Vec<f64>; 3],

    /// Values of efficiencies at grid points:
    /// - temperature
    /// - soc
    /// - c_rate
    pub eta_interp_values: Vec<Vec<Vec<f64>>>,
    #[serde(rename = "pwr_out_max_watts")]
    /// Max output (and input) power battery can produce (accept)
    pub pwr_out_max: si::Power,

    /// Total energy capacity of battery of full discharge SOC of 0.0 and 1.0
    #[serde(rename = "energy_capacity_joules")]
    pub energy_capacity: si::Energy,

    /// Hard limit on minimum SOC, e.g. 0.05
    pub min_soc: si::Ratio,
    /// Hard limit on maximum SOC, e.g. 0.95
    pub max_soc: si::Ratio,
    /// SOC at which negative/charge power begins to ramp down.
    /// Should always be slightly below [Self::max_soc].
    #[api(skip_get, skip_set)]
    pub soc_hi_ramp_start: Option<si::Ratio>,
    /// SOC at which positive/discharge power begins to ramp down.
    /// Should always be slightly above [Self::min_soc].
    #[api(skip_get, skip_set)]
    pub soc_lo_ramp_start: Option<si::Ratio>,
    /// Time step interval at which history is saved
    pub save_interval: Option<usize>,
    #[serde(default)]
    /// Custom vector of [Self::state]
    pub history: ReversibleEnergyStorageStateHistoryVec,
}

impl Default for ReversibleEnergyStorage {
    fn default() -> Self {
        let file_contents = include_str!("reversible_energy_storage.default.yaml");
        serde_yaml::from_str::<ReversibleEnergyStorage>(file_contents).unwrap()
    }
}

impl Mass for ReversibleEnergyStorage {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        self.check_mass_consistent()?;
        Ok(self.mass)
    }

    fn update_mass(&mut self, mass: Option<si::Mass>) -> anyhow::Result<()> {
        match mass {
            Some(mass) => {
                self.specific_energy = Some(self.energy_capacity / mass);
                self.mass = Some(mass)
            }
            None => match self.specific_energy {
                Some(e) => self.mass = Some(self.energy_capacity / e),
                None => {
                    bail!(format!(
                        "{}\n{}",
                        format_dbg!(),
                        "Mass must be provided or `self.specific_energy` must be set"
                    ));
                }
            },
        };

        Ok(())
    }

    fn check_mass_consistent(&self) -> anyhow::Result<()> {
        match &self.mass {
            Some(mass) => match &self.specific_energy {
                Some(e) => {
                    ensure!(self.energy_capacity / *e == *mass,
                    format!("{}\n{}", format_dbg!(), "ReversibleEnergyStorage `energy_capacity`, `specific_energy` and `mass` are not consistent"))
                }
                None => {}
            },
            None => {}
        }
        Ok(())
    }
}

impl SerdeAPI for ReversibleEnergyStorage {
    fn init(&mut self) -> anyhow::Result<()> {
        self.check_mass_consistent()?;
        Ok(())
    }
}

impl ReversibleEnergyStorage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        temperature_interp_grid: Vec<f64>,
        soc_interp_grid: Vec<f64>,
        c_rate_interp_grid: Vec<f64>,
        eta_interp_values: Vec<Vec<Vec<f64>>>,
        pwr_out_max_watts: f64,
        energy_capacity_joules: f64,
        min_soc: f64,
        max_soc: f64,
        initial_soc: f64,
        initial_temperature_celcius: f64,
        soc_hi_ramp_start: Option<f64>,
        soc_lo_ramp_start: Option<f64>,
        save_interval: Option<usize>,
    ) -> anyhow::Result<Self> {
        ensure!(
            temperature_interp_grid.len() == eta_interp_values.len(),
            format!(
                "{}\nres temperature grid size must match eta_interp_values dimension 0",
                format_dbg!(temperature_interp_grid.len() == eta_interp_values.len())
            )
        );
        ensure!(
            is_sorted(&temperature_interp_grid),
            format!(
                "{}\nres temperature grid must be sorted",
                format_dbg!(is_sorted(&temperature_interp_grid))
            )
        );
        ensure!(
            soc_interp_grid.len() == eta_interp_values[0].len(),
            format!(
                "{}\nsoc grid size must match eta_interp_values dimension 1",
                format_dbg!(soc_interp_grid.len() == eta_interp_values[0].len())
            )
        );
        ensure!(
            is_sorted(&soc_interp_grid),
            format!(
                "{}\nsoc grid must be sorted",
                format_dbg!(is_sorted(&soc_interp_grid))
            )
        );
        ensure!(
            c_rate_interp_grid.len() == eta_interp_values[0][0].len(),
            format!(
                "{}\nc rate grid size must match eta_interp_values dimension 2",
                format_dbg!(c_rate_interp_grid.len() == eta_interp_values[0][0].len())
            )
        );
        ensure!(
            is_sorted(&soc_interp_grid),
            format!(
                "{}\ncrate grid must be sorted",
                format_dbg!(is_sorted(&soc_interp_grid))
            )
        );
        ensure!(
            min_soc <= initial_soc || initial_soc <= max_soc,
            format!(
                "{}\ninitial soc must be between min and max soc, inclusive",
                format_dbg!(min_soc <= initial_soc || initial_soc <= max_soc)
            )
        );

        let initial_state = ReversibleEnergyStorageState {
            soc: uc::R * initial_soc,
            temperature_celsius: initial_temperature_celcius,
            ..Default::default()
        };
        let interp_grid = [temperature_interp_grid, soc_interp_grid, c_rate_interp_grid];
        Ok(ReversibleEnergyStorage {
            eta_interp_grid: interp_grid,
            eta_interp_values,
            pwr_out_max: uc::W * pwr_out_max_watts,
            energy_capacity: uc::J * energy_capacity_joules,
            min_soc: uc::R * min_soc,
            max_soc: uc::R * max_soc,
            soc_hi_ramp_start: soc_hi_ramp_start.map(|val| val * uc::R),
            soc_lo_ramp_start: soc_lo_ramp_start.map(|val| val * uc::R),
            state: initial_state,
            save_interval,
            history: ReversibleEnergyStorageStateHistoryVec::new(),
            ..Default::default()
        })
    }

    fn volume(&self) -> anyhow::Result<Option<si::Volume>> {
        self.check_vol_consistent()?;
        Ok(self.volume)
    }

    fn update_volume(&mut self, volume: Option<si::Volume>) -> anyhow::Result<()> {
        match volume {
            Some(volume) => {
                self.energy_density = Some(self.energy_capacity / volume);
                self.volume = Some(volume);
            }
            None => match self.energy_density {
                Some(e) => self.volume = Some(self.energy_capacity / e),
                None => {
                    bail!(format!(
                        "{}\n{}",
                        format_dbg!(),
                        "Volume must be provided or `self.energy_density` must be set"
                    ));
                }
            },
        };

        Ok(())
    }

    fn check_vol_consistent(&self) -> anyhow::Result<()> {
        match &self.volume {
            Some(vol) => match &self.energy_density {
                Some(e) => {
                    ensure!(self.energy_capacity / *e == *vol,
                    format!("{}\n{}", format_dbg!(), "ReversibleEnergyStorage `energy_capacity`, `energy_density` and `volume` are not consistent"))
                }
                None => {}
            },
            None => {}
        }
        Ok(())
    }

    /// Returns max output and max regen power based on current state
    /// Arguments:
    /// - charge_buffer: min future train energy state - current train energy state.
    /// If provided, reserves some charge capacity for future.
    /// - discharge_buffer: max future train energy state - current train energy state.
    /// If provided, reserves some discharge capacity for future.
    pub fn set_cur_pwr_out_max(
        &mut self,
        pwr_aux: si::Power,
        charge_buffer: Option<si::Energy>,
        discharge_buffer: Option<si::Energy>,
    ) -> anyhow::Result<()> {
        let mut state = &mut self.state;

        if self.soc_hi_ramp_start.is_none() {
            self.soc_hi_ramp_start = Some(self.max_soc - 0.05 * uc::R);
        }
        if self.soc_lo_ramp_start.is_none() {
            self.soc_lo_ramp_start = Some(self.min_soc + 0.05 * uc::R);
        }

        // operating lo_ramp_start and min_soc, allowing for buffer
        state.soc_lo_ramp_start = (self.soc_lo_ramp_start.unwrap()
            + charge_buffer.unwrap_or_default() / self.energy_capacity)
            .min(self.max_soc);
        // print_to_py!(
        //     "state_soc_lo_ramp_start",
        //     state.soc_lo_ramp_start.get::<si::ratio>()
        // );
        state.min_soc = (self.min_soc + charge_buffer.unwrap_or_default() / self.energy_capacity)
            .min(self.max_soc);
        // print_to_py!("state_min_soc", state.min_soc.get::<si::ratio>());

        // operating hi_ramp_start and max_soc, allowing for buffer
        state.soc_hi_ramp_start = (self.soc_hi_ramp_start.unwrap()
            - discharge_buffer.unwrap_or_default() / self.energy_capacity)
            .max(self.min_soc);
        // print_to_py!(
        //     "state_soc_hi_ramp_start",
        //     state.soc_hi_ramp_start.get::<si::ratio>()
        // );
        state.max_soc = (self.max_soc
            - discharge_buffer.unwrap_or_default() / self.energy_capacity)
            .max(self.min_soc);
        // print_to_py!("state_max_soc", state.max_soc.get::<si::ratio>());

        state.pwr_disch_max = uc::W
            * interp1d(
                &state.soc.get::<si::ratio>(),
                &[
                    state.min_soc.get::<si::ratio>(),
                    state.soc_lo_ramp_start.get::<si::ratio>(),
                ],
                &[0.0, self.pwr_out_max.get::<si::watt>()],
                false, // don't extrapolate
            )?;

        state.pwr_charge_max = uc::W
            * interp1d(
                &state.soc.get::<si::ratio>(),
                &[
                    state.soc_hi_ramp_start.get::<si::ratio>(),
                    state.max_soc.get::<si::ratio>(),
                ],
                &[self.pwr_out_max.get::<si::watt>(), 0.0],
                false, // don't extrapolate
            )?;

        state.pwr_prop_out_max = state.pwr_disch_max - pwr_aux;
        state.pwr_regen_out_max = state.pwr_charge_max + pwr_aux;

        Ok(())
    }

    pub fn solve_energy_consumption(
        &mut self,
        pwr_prop_req: si::Power,
        pwr_aux_req: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        let mut state = &mut self.state;

        ensure!(
            state.soc <= state.max_soc || pwr_prop_req >= si::Power::ZERO,
            "{}\npwr_prop_req must be greater than 0 if SOC is over max SOC\nstate.soc = {}",
            format_dbg!(state.soc <= state.max_soc || pwr_prop_req >= si::Power::ZERO),
            state.soc.get::<si::ratio>()
        );
        ensure!(
            state.soc >= state.min_soc || pwr_prop_req <= si::Power::ZERO,
            "{}\npwr_prop_req must be less than 0 if SOC is below min SOC\nstate.soc = {}",
            format_dbg!(state.soc >= state.min_soc || pwr_prop_req <= si::Power::ZERO),
            state.soc.get::<si::ratio>()
        );

        if pwr_prop_req + pwr_aux_req >= si::Power::ZERO {
            ensure!(
                utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), &self.pwr_out_max, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds static max discharge power ({:.6} MW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(
                    &(pwr_prop_req + pwr_aux_req),
                    &self.pwr_out_max,
                    Some(TOL)
                )),
                (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                state.pwr_disch_max.get::<si::megawatt>(),
                state.soc.get::<si::ratio>()
            );
            ensure!(
                utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), &state.pwr_disch_max, Some(TOL)),
                "{}\nres required power ({:.6} MW) exceeds transient max discharge power ({:.6} MW)\nstate.soc = {}",
                format_dbg!(utils::almost_le_uom(&(pwr_prop_req + pwr_aux_req), &state.pwr_disch_max, Some(TOL))),
                (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                state.pwr_disch_max.get::<si::megawatt>(),
                state.soc.get::<si::ratio>()
            );
        } else {
            ensure!(
                utils::almost_ge_uom(&(pwr_prop_req + pwr_aux_req), &-self.pwr_out_max, Some(TOL)),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds static max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_prop_req + pwr_aux_req),
                        &-self.pwr_out_max,
                        Some(TOL)
                    )),
                    (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                    state.pwr_charge_max.get::<si::megawatt>()
                )
            );
            ensure!(
                utils::almost_ge_uom(
                    &(pwr_prop_req + pwr_aux_req),
                    &-state.pwr_charge_max,
                    Some(TOL)
                ),
                format!(
                    "{}\nres required power ({:.6} MW) exceeds transient max power ({:.6} MW)",
                    format_dbg!(utils::almost_ge_uom(
                        &(pwr_prop_req + pwr_aux_req),
                        &-state.pwr_charge_max,
                        Some(TOL)
                    )),
                    (pwr_prop_req + pwr_aux_req).get::<si::megawatt>(),
                    state.pwr_charge_max.get::<si::megawatt>()
                )
            );
        }

        state.pwr_out_propulsion = pwr_prop_req;
        state.energy_out_propulsion += pwr_prop_req * dt;
        state.pwr_aux = pwr_aux_req;
        state.energy_aux += state.pwr_aux * dt;

        state.pwr_out_electrical = state.pwr_out_propulsion + state.pwr_aux;
        state.energy_out_electrical += state.pwr_out_electrical * dt;

        let c_rate = state.pwr_out_electrical.get::<si::watt>()
            / (self.energy_capacity.get::<si::watt_hour>());
        // evaluate the battery efficiency at the current state
        let eta_point = [
            state.temperature_celsius,
            state.soc.get::<si::ratio>(),
            c_rate,
        ];
        let eta = interp3d(&eta_point, &self.eta_interp_grid, &self.eta_interp_values).unwrap();

        state.eta = uc::R * eta;
        ensure!(
            state.eta >= 0.0 * uc::R || state.eta <= 1.0 * uc::R,
            format!(
                "{}\nres eta ({}) must be between 0 and 1",
                format_dbg!(state.eta >= 0.0 * uc::R || state.eta <= 1.0 * uc::R),
                state.eta.get::<si::ratio>()
            )
        );

        if state.pwr_out_electrical > si::Power::ZERO {
            // if positive, chemical power must be greater than electrical power
            // i.e. not all chemical power can be converted to electrical power
            state.pwr_out_chemical = state.pwr_out_electrical / eta;
        } else {
            // if negative, chemical power, must be less than electrical power
            // i.e. not all electrical power can be converted back to chemical power
            state.pwr_out_chemical = state.pwr_out_electrical * eta;
        }
        state.energy_out_chemical += state.pwr_out_chemical * dt;

        state.pwr_loss = (state.pwr_out_chemical - state.pwr_out_electrical).abs();
        state.energy_loss += state.pwr_loss * dt;

        let new_soc = state.soc - state.pwr_out_chemical * dt / self.energy_capacity;
        state.soc = new_soc;
        Ok(())
    }

    pub fn get_eta_max(&self) -> f64 {
        // since eta is all f64 between 0 and 1, NEG_INFINITY is safe
        self.eta_interp_values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, curr2| {
                curr2
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, curr1| {
                        curr1
                            .iter()
                            .fold(f64::NEG_INFINITY, |acc, &curr| acc.max(curr))
                            .max(acc)
                    })
                    .max(acc)
            })
    }

    /// Scales eta_interp by ratio of new `eta_max` per current calculated
    /// max linearly, such that `eta_min` is untouched
    pub fn set_eta_max(&mut self, eta_max: f64) -> Result<(), String> {
        if (self.get_eta_min()..=1.0).contains(&eta_max) {
            // this appears to be efficient way to get max of Vec<f64>
            let old_max = self.get_eta_max();
            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        .map(|v1| v1.iter().map(|val| val * eta_max / old_max).collect())
                        .collect()
                })
                .collect();
            Ok(())
        } else {
            Err(format!(
                "`eta_max` ({:.3}) must be between `eta_min` ({:.3}) and 1.0",
                eta_max,
                self.get_eta_min()
            ))
        }
    }

    pub fn get_eta_min(&self) -> f64 {
        // since eta is all f64 between 0 and 1, INFINITY is safe
        self.eta_interp_values
            .iter()
            .fold(f64::INFINITY, |acc, curr2| {
                curr2
                    .iter()
                    .fold(f64::INFINITY, |acc, curr1| {
                        curr1
                            .iter()
                            .fold(f64::INFINITY, |acc, &curr| acc.min(curr))
                            .min(acc)
                    })
                    .min(acc)
            })
    }

    /// Max value of `eta_interp` minus min value of `eta_interp`.
    pub fn get_eta_range(&self) -> f64 {
        self.get_eta_max() - self.get_eta_min()
    }

    /// Scales values of `eta_interp` without changing max such that max - min
    /// is equal to new range
    pub fn set_eta_range(&mut self, eta_range: f64) -> anyhow::Result<()> {
        let eta_max = self.get_eta_max();
        if eta_range == 0.0 {
            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        // this is sloppy but should work
                        .map(|v1| v1.iter().map(|_val| eta_max).collect())
                        .collect()
                })
                .collect();
            Ok(())
        } else if (0.0..=1.0).contains(&eta_range) {
            let old_min = self.get_eta_min();
            let old_range = self.get_eta_max() - old_min;

            self.eta_interp_values = self
                .eta_interp_values
                .iter()
                .map(|v2| {
                    v2.iter()
                        .map(|v1| {
                            v1.iter()
                                .map(|val| eta_max + (val - eta_max) * eta_range / old_range)
                                .collect()
                        })
                        .collect()
                })
                .collect();
            if self.get_eta_min() < 0.0 {
                let val_neg = self.get_eta_min();
                self.eta_interp_values = self
                    .eta_interp_values
                    .iter()
                    .map(|v2| {
                        v2.iter()
                            .map(|v1| v1.iter().map(|val| val - val_neg).collect())
                            .collect()
                    })
                    .collect();
            }
            ensure!(
                self.get_eta_max() <= 1.0,
                format!(
                    "{}\n`eta_max` ({:.3}) must be no greater than 1.0",
                    format_dbg!(self.get_eta_max() <= 1.0),
                    self.get_eta_max()
                )
            );
            Ok(())
        } else {
            bail!("`eta_range` ({:.3}) must be between 0.0 and 1.0", eta_range)
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize, Debug, PartialEq, HistoryVec)]
#[altrios_api]
// component limits
/// ReversibleEnergyStorage state variables
pub struct ReversibleEnergyStorageState {
    // limits
    // TODO: create separate binning for cat power and
    /// maximum catenary power capability
    pub pwr_cat_max: si::Power,
    /// max output power for propulsion during positive traction
    pub pwr_prop_out_max: si::Power,
    /// max regen power for propulsion during negative traction
    pub pwr_regen_out_max: si::Power,
    /// max discharge power total
    pub pwr_disch_max: si::Power,
    /// max charge power on the output side
    pub pwr_charge_max: si::Power,

    /// simulation step
    pub i: usize,

    /// state of charge (SOC)
    pub soc: si::Ratio,
    /// Chemical <-> Electrical conversion efficiency based on current power demand
    pub eta: si::Ratio,
    /// State of Health (SOH)
    pub soh: f64,

    // TODO: add `pwr_out_neg_electrical` and `pwr_out_pos_electrical` and corresponding energies
    // powers
    /// total electrical power; positive is discharging
    pub pwr_out_electrical: si::Power,
    /// electrical power going to propulsion
    pub pwr_out_propulsion: si::Power,
    /// electrical power going to aux loads
    pub pwr_aux: si::Power,
    /// power dissipated as loss
    pub pwr_loss: si::Power,
    /// chemical power; positive is discharging
    pub pwr_out_chemical: si::Power,

    // cumulative energies
    /// cumulative total electrical energy; positive is discharging
    pub energy_out_electrical: si::Energy,
    /// cumulative electrical energy going to propulsion
    pub energy_out_propulsion: si::Energy,
    /// cumulative electrical energy going to aux loads
    pub energy_aux: si::Energy,
    /// cumulative energy dissipated as loss
    pub energy_loss: si::Energy,
    /// cumulative chemical energy; positive is discharging
    pub energy_out_chemical: si::Energy,

    /// dynamically updated max SOC limit
    pub max_soc: si::Ratio,
    /// dynamically updated SOC at which negative/charge power begins to ramp down.
    pub soc_hi_ramp_start: si::Ratio,
    /// dynamically updated min SOC limit
    pub min_soc: si::Ratio,
    /// dynamically updated SOC at which positive/discharge power begins to ramp down.
    pub soc_lo_ramp_start: si::Ratio,

    /// component temperature
    pub temperature_celsius: f64,
}

impl Default for ReversibleEnergyStorageState {
    fn default() -> Self {
        Self {
            i: 1,
            // slightly less than max soc for default ReversibleEnergyStorage
            soc: uc::R * 0.95,
            soh: 1.0,
            eta: Default::default(),
            pwr_prop_out_max: Default::default(),
            pwr_regen_out_max: Default::default(),
            pwr_disch_max: Default::default(),
            pwr_charge_max: Default::default(),
            pwr_cat_max: Default::default(),
            pwr_out_electrical: Default::default(),
            pwr_out_propulsion: Default::default(),
            pwr_aux: Default::default(),
            pwr_out_chemical: Default::default(),
            pwr_loss: Default::default(),
            energy_out_electrical: Default::default(),
            energy_out_propulsion: Default::default(),
            energy_aux: Default::default(),
            energy_out_chemical: Default::default(),
            energy_loss: Default::default(),
            max_soc: uc::R * 1.0,
            soc_hi_ramp_start: uc::R * 1.0,
            min_soc: si::Ratio::ZERO,
            soc_lo_ramp_start: si::Ratio::ZERO,
            temperature_celsius: 45.0,
        }
    }
}

mod tests {
    use super::*;

    fn _mock_res() -> ReversibleEnergyStorage {
        ReversibleEnergyStorage::default()
    }

    #[test]
    fn test_res_constructor() {
        let _res = _mock_res();
    }

    #[test]
    fn test_set_cur_pwr_out_max() {
        let mut res = _mock_res();
        res.max_soc = 0.9 * uc::R;
        res.min_soc = 0.1 * uc::R;
        res.state.soc = 0.98 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_charge_max, si::Power::ZERO);
        res.soc_hi_ramp_start = Some(0.8 * uc::R);
        res.state.soc = 0.8 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_charge_max, res.pwr_out_max);
        res.state.soc = 0.85 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert!(res.state.pwr_charge_max < res.pwr_out_max / 2.0 * 1.0001);
        assert!(res.state.pwr_charge_max > res.pwr_out_max / 2.0 * 0.9999);
        res.state.soc = 0.9 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_charge_max, si::Power::ZERO);
        res.state.soc = 0.9 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_charge_max, si::Power::ZERO);
        res.soc_lo_ramp_start = Some(0.2 * uc::R);
        res.state.soc = 0.2 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_disch_max, res.pwr_out_max);
        res.state.soc = 0.15 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert!(res.state.pwr_disch_max < res.pwr_out_max / 2.0 * 1.0001);
        assert!(res.state.pwr_charge_max > res.pwr_out_max / 2.0 * 0.9999);
        res.state.soc = 0.1 * uc::R;
        res.set_cur_pwr_out_max(5e3 * uc::W, None, None).unwrap();
        assert_eq!(res.state.pwr_disch_max, si::Power::ZERO);
    }

    #[test]
    fn test_get_and_set_eta() {
        let mut res = _mock_res();
        let eta_max = 0.998958;
        let eta_min = 0.662822531196789;
        let eta_range = 0.336135468803211;

        eta_test_body!(res, eta_max, eta_min, eta_range);
    }
}
