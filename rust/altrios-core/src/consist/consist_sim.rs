use log::info;
use proc_macros::altrios_api;
use serde::{Deserialize, Serialize};

use crate::consist::locomotive::loco_sim::PowerTrace;
use crate::consist::Consist;
use crate::consist::LocoTrait;
use crate::imports::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
#[altrios_api(
    #[new]
    fn __new__(consist: Consist, power_trace: PowerTrace, save_interval: Option<usize>) -> Self {
        Self::new(consist, power_trace, save_interval)
    }

    #[pyo3(name = "walk")]
    /// Exposes `walk` to python
    fn walk_py(&mut self) -> anyhow::Result<()> {
        self.walk()
    }

    #[pyo3(name = "step")]
    fn step_py(&mut self) -> anyhow::Result<()> {
        self.step()
    }

    #[pyo3(name = "set_save_interval")]
    /// Set save interval and cascade to nested components.
    fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }

    #[pyo3(name = "get_save_interval")]
    fn get_save_interval_py(&self) -> PyResult<Option<usize>> {
        Ok(self.loco_con.get_save_interval())
    }

    #[pyo3(name = "trim_failed_steps")]
    fn trim_failed_steps_py(&mut self) -> PyResult<()> {
        self.trim_failed_steps()?;
        Ok(())
    }
)]
pub struct ConsistSimulation {
    pub loco_con: Consist,
    pub power_trace: PowerTrace,
    pub i: usize,
}

impl ConsistSimulation {
    pub fn new(consist: Consist, power_trace: PowerTrace, save_interval: Option<usize>) -> Self {
        let mut consist_sim = Self {
            loco_con: consist,
            power_trace,
            i: 1,
        };
        consist_sim.loco_con.set_save_interval(save_interval);
        consist_sim
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
        self.loco_con.set_save_interval(save_interval);
    }

    pub fn step(&mut self) -> anyhow::Result<()> {
        self.solve_step()
            .map_err(|err| err.context(format!("time step: {}", self.i)))?;
        self.save_state();
        self.i += 1;
        self.loco_con.step();
        Ok(())
    }

    pub fn solve_step(&mut self) -> anyhow::Result<()> {
        self.loco_con
            .set_cur_pwr_max_out(None, self.power_trace.dt(self.i))?;
        self.solve_energy_consumption(self.power_trace.pwr[self.i], self.power_trace.dt(self.i))?;
        Ok(())
    }

    fn save_state(&mut self) {
        self.loco_con.save_state();
    }

    /// Iterates step to solve all time steps.
    pub fn walk(&mut self) -> anyhow::Result<()> {
        // Just here to showcase logging;
        info!("Performing walk method on consist sim");

        self.save_state();
        while self.i < self.power_trace.len() {
            self.step()?;
        }
        Ok(())
    }

    /// Solves for fuel and RES consumption \[W\]
    /// Arguments:
    /// ----------
    /// pwr_out_req: float, output brake power required from fuel converter.
    /// dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.loco_con
            .solve_energy_consumption(pwr_out_req, dt, Some(true))?;
        Ok(())
    }
}

impl Default for ConsistSimulation {
    fn default() -> Self {
        Self::new(Consist::default(), PowerTrace::default(), Some(1))
    }
}

#[cfg(test)]
mod tests {
    use super::{Consist, ConsistSimulation};
    use crate::consist::locomotive::loco_sim::PowerTrace;

    #[test]
    fn test_consist_sim() {
        let consist = Consist::default();
        let pt = PowerTrace::default();
        let mut consist_sim = ConsistSimulation::new(consist, pt, None);
        consist_sim.walk().unwrap();
    }
}
