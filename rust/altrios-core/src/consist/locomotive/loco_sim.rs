//! Module for standalone simulation of locomotive powertrains

use rayon::prelude::*;

use crate::consist::locomotive::Locomotive;
use crate::consist::LocoTrait;
use crate::imports::*;

#[altrios_api(
    #[new]
    fn __new__(
        time_seconds: Vec<f64>,
        pwr_watts: Vec<f64>,
        engine_on: Vec<Option<bool>>,
    ) -> PyResult<Self> {
        Ok(Self::new(time_seconds, pwr_watts, engine_on))
    }

    #[classmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(_cls: &PyType, pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
/// Container
pub struct PowerTrace {
    /// simulation time \[s\]
    #[serde(rename = "time_seconds")]
    pub time: Vec<si::Time>,
    /// simulation power \[W\]
    #[serde(rename = "pwr_watts")]
    pub pwr: Vec<si::Power>,
    /// Whether engine is on
    pub engine_on: Vec<Option<bool>>,
}

impl PowerTrace {
    pub fn new(time_s: Vec<f64>, pwr_watts: Vec<f64>, engine_on: Vec<Option<bool>>) -> Self {
        Self {
            time: time_s.iter().map(|x| uc::S * (*x)).collect(),
            pwr: pwr_watts.iter().map(|x| uc::W * (*x)).collect(),
            engine_on,
        }
    }

    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            pwr: Vec::new(),
            engine_on: Vec::new(),
        }
    }

    pub fn dt(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, pt_element: PowerTraceElement) {
        self.time.push(pt_element.time);
        self.pwr.push(pt_element.pwr);
        self.engine_on.push(pt_element.engine_on);
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or(self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        self.pwr = self.pwr[start_idx..end_idx].to_vec();
        self.engine_on = self.engine_on[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty cycle to be populated
        let mut pt = Self::empty();

        let file = File::open(pathbuf)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let pt_elem: PowerTraceElement = result?;
            pt.push(pt_elem);
        }
        if pt.is_empty() {
            bail!("Invalid PowerTrace file; Powertrace is empty")
        } else {
            Ok(pt)
        }
    }
}

impl Default for PowerTrace {
    fn default() -> Self {
        let pwr_max_watts = 1.5e6;
        let pwr_watts_ramp: Vec<f64> = Vec::linspace(0., pwr_max_watts, 300);
        let mut pwr_watts = pwr_watts_ramp.clone();
        pwr_watts.append(&mut vec![pwr_max_watts; 100]);
        pwr_watts.append(&mut pwr_watts_ramp.iter().rev().copied().collect());
        let time_s: Vec<f64> = (0..pwr_watts.len()).map(|x| x as f64).collect();
        let time_len = time_s.len();
        Self::new(time_s, pwr_watts, vec![Some(true); time_len])
    }
}

/// Element of `PowerTrace`.  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct PowerTraceElement {
    /// simulation time \[s\]
    #[serde(rename = "time_seconds")]
    time: si::Time,
    /// simulation power \[W\]
    #[serde(rename = "pwr_watts")]
    pwr: si::Power,
    /// Whether engine is on
    engine_on: Option<bool>,
}

#[altrios_api(
    #[new]
    fn __new__(
        loco_unit: Locomotive,
        power_trace: PowerTrace,
        save_interval: Option<usize>,
    ) -> Self {
        Self::new(loco_unit, power_trace, save_interval)
    }

    #[pyo3(name = "walk")]
    /// Exposes `walk` to Python.
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[pyo3(name = "step")]
    fn step_py(&mut self) -> anyhow::Result<()> {
        self.step()
    }

    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) -> PyResult<()> {
        self.set_save_interval(save_interval);
        Ok(())
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> PyResult<Option<usize>> {
        Ok(self.loco_unit.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> PyResult<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
/// Struct for simulating operation of a standalone locomotive.  
pub struct LocomotiveSimulation {
    pub loco_unit: Locomotive,
    pub power_trace: PowerTrace,
    pub i: usize,
}

impl LocomotiveSimulation {
    pub fn new(
        loco_unit: Locomotive,
        power_trace: PowerTrace,
        save_interval: Option<usize>,
    ) -> Self {
        let mut loco_sim = Self {
            loco_unit,
            power_trace,
            i: 1,
        };
        loco_sim.loco_unit.set_save_interval(save_interval);
        loco_sim
    }

    /// Trims off any portion of the trip that failed to run
    pub fn trim_failed_steps(&mut self) -> anyhow::Result<()> {
        if self.i <= 1 {
            bail!("`walk` method has not proceeded past first time step.")
        }
        self.power_trace.trim(None, Some(self.i))?;

        Ok(())
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.loco_unit.set_save_interval(save_interval);
    }

    pub fn get_save_interval(&self) -> Option<usize> {
        self.loco_unit.get_save_interval()
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        self.solve_step()
            .map_err(|err| err.context(format!("time step: {}", self.i)))?;
        self.save_state();
        self.i += 1;
        self.loco_unit.step();
        Ok(())
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        // linear aux model
        let engine_on = self.power_trace.engine_on[self.i];
        self.loco_unit.set_pwr_aux(engine_on);
        self.loco_unit
            .set_cur_pwr_max_out(None, self.power_trace.dt(self.i))?;
        self.solve_energy_consumption(
            self.power_trace.pwr[self.i],
            self.power_trace.dt(self.i),
            engine_on,
        )?;
        ensure!(
            utils::almost_eq_uom(
                &self.power_trace.pwr[self.i],
                &self.loco_unit.state.pwr_out,
                None
            ),
            format_dbg!(
                (utils::almost_eq_uom(
                    &self.power_trace.pwr[self.i],
                    &self.loco_unit.state.pwr_out,
                    None
                ))
            )
        );
        Ok(())
    }

    fn save_state(&mut self) {
        self.loco_unit.save_state();
    }

    /// Iterates `save_state` and `step` through all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        self.save_state();
        while self.i < self.power_trace.len() {
            self.step()?
        }
        ensure!(self.i == self.power_trace.len());
        Ok(())
    }

    /// Solves for fuel and RES consumption \[W\]
    /// Arguments:
    /// ----------
    /// pwr_out_req: float, output brake power required from fuel converter.
    /// dt: current time step size
    /// engine_on: whether or not locomotive is active
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        engine_on: Option<bool>,
    ) -> anyhow::Result<()> {
        self.loco_unit
            .solve_energy_consumption(pwr_out_req, dt, engine_on)?;
        Ok(())
    }
}

impl Default for LocomotiveSimulation {
    fn default() -> Self {
        let power_trace = PowerTrace::default();
        let loco_unit = Locomotive::default();
        Self::new(loco_unit, power_trace, None)
    }
}

#[altrios_api(
    #[pyo3(name="walk")]
    /// Exposes `walk` to Python.
    fn walk_py(&mut self, b_parallelize: Option<bool>) -> anyhow::Result<()> {
        let b_par = b_parallelize.unwrap_or(false);
        self.walk(b_par)
    }
)]
#[derive(Clone, Debug, Serialize, Deserialize, SerdeAPI)]
pub struct LocomotiveSimulationVec(pub Vec<LocomotiveSimulation>);

impl Default for LocomotiveSimulationVec {
    fn default() -> Self {
        Self(vec![LocomotiveSimulation::default(); 3])
    }
}

impl LocomotiveSimulationVec {
    /// Calls `walk` for each locomotive in vec.
    pub fn walk(&mut self, parallelize: bool) -> anyhow::Result<()> {
        if parallelize {
            self.0
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(i, loco_sim)| {
                    loco_sim
                        .walk()
                        .map_err(|err| err.context(format!("loco_sim idx:{}", i)))
                })?;
        } else {
            self.0
                .iter_mut()
                .enumerate()
                .try_for_each(|(i, loco_sim)| {
                    loco_sim
                        .walk()
                        .map_err(|err| err.context(format!("loco_sim idx:{}", i)))
                })?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::{Locomotive, LocomotiveSimulation, LocomotiveSimulationVec, PowerTrace};
    use crate::consist::locomotive::LocoType;

    #[test]
    fn test_loco_sim_vec_par() {
        let mut loco_sim_vec = LocomotiveSimulationVec::default();
        loco_sim_vec.walk(true).unwrap();
    }

    #[test]
    fn test_loco_sim_vec_ser() {
        let mut loco_sim_vec = LocomotiveSimulationVec::default();
        loco_sim_vec.walk(false).unwrap();
    }

    #[test]
    fn test_power_trace_serialize() {
        let pt = PowerTrace::default();
        let json = serde_json::to_string(&pt).unwrap();
        println!("{json}");
        let new_pt: PowerTrace = serde_json::from_str(json.as_str()).unwrap();
        println!("{new_pt:?}");
    }

    #[test]
    fn test_power_trace_serialize_yaml() {
        let pt = PowerTrace::default();
        let yaml = serde_yaml::to_string(&pt).unwrap();
        println!("{yaml}");
        let new_pt: PowerTrace = serde_yaml::from_str(yaml.as_str()).unwrap();
        println!("{new_pt:?}");
    }

    #[test]
    fn test_conventional_locomotive_sim() {
        let cl = Locomotive::default();
        let pt = PowerTrace::default();
        let mut loco_sim = LocomotiveSimulation::new(cl, pt, None);
        loco_sim.walk().unwrap();
    }

    #[test]
    fn test_hybrid_locomotive_sim() {
        // let hel = Locomotive::new(
        //     LocoType::HybridLoco(Box::default()),

        // );

        // let pt = PowerTrace::default();
        // let mut loco_sim = LocomotiveSimulation::new(hel, pt, None);
        // loco_sim.walk().unwrap();
    }

    #[test]
    fn test_battery_locomotive_sim() {
        let bel = Locomotive::default_battery_electric_loco();
        let pt = PowerTrace::default();
        let mut loco_sim = LocomotiveSimulation::new(bel, pt, None);
        loco_sim.walk().unwrap();
    }

    #[test]
    fn test_set_save_interval() {
        let mut ls = LocomotiveSimulation::default();

        assert!(ls.get_save_interval().is_none());
        assert!(ls.loco_unit.get_save_interval().is_none());
        assert!(match &ls.loco_unit.loco_type {
            LocoType::ConventionalLoco(loco) => {
                loco.fc.save_interval
            }
            _ => None,
        }
        .is_none());

        ls.set_save_interval(Some(1));

        assert_eq!(ls.get_save_interval(), Some(1));
        assert_eq!(ls.loco_unit.get_save_interval(), Some(1));
        assert_eq!(
            match &ls.loco_unit.loco_type {
                LocoType::ConventionalLoco(loco) => {
                    loco.fc.save_interval
                }
                _ => None,
            },
            Some(1)
        );

        ls.walk().unwrap();

        assert_eq!(ls.get_save_interval(), Some(1));
        assert_eq!(ls.loco_unit.get_save_interval(), Some(1));
        assert_eq!(
            match &ls.loco_unit.loco_type {
                LocoType::ConventionalLoco(loco) => {
                    loco.fc.save_interval
                }
                _ => None,
            },
            Some(1)
        );
    }

    #[test]
    fn test_power_trace_trim() {
        let pt = PowerTrace::default();
        let mut pt_test = pt.clone();

        assert!(pt == pt_test);
        pt_test.trim(None, None).unwrap();
        assert!(pt == pt_test);

        pt_test.trim(None, Some(10)).unwrap();
        assert!(pt_test != pt);
        assert!(pt_test.len() == 10);
    }
}
