use argmin::prelude::*;
use argmin::solver::goldensectionsearch::GoldenSectionSearch;

use super::*;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
/// Hybrid locomotive with both engine and reversible energy storage (aka battery)  
/// This type of locomotive is not likely to be widely prevalent due to modularity of consists.  
pub struct HybridElectricVehicle {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub trans: Transmission,
    /// if 1.0, then all power comes from fuel via generator
    /// if 0.0, then all power comes from res
    /// If fuel_res_ratio is some, then fuel_res_split is initial and then dynamically updated
    pub fuel_res_split: f64,
    /// Relative cost of fuel energy to RES energy.   If large, fuel is more expensive
    /// relative to RES energy and controls will work to minimize fuel usage.
    /// if fuel_res_ratio is provided, fuel_res_split should be None
    pub fuel_res_ratio: Option<f64>,
    /// interval governing how frequently GoldenSectionSearch
    /// should be used to optimize fuel_res_split
    pub gss_interval: Option<usize>,
    /// time step needed for solving for optimal fuel/res split.
    /// should not be public!
    dt: si::Time,
    /// step counter for gss_interval check
    i: usize,
}

impl SerdeAPI for HybridElectricVehicle {}

impl VehicleTrait for Box<HybridElectricVehicle> {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        todo!();
        Ok(())
    }

    fn save_state(&mut self) {
        self.deref_mut().save_state();
    }

    fn step(&mut self) {
        self.deref_mut().step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss + self.res.state.energy_loss + self.trans.state.energy_loss
    }
}

impl HybridElectricVehicle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fuel_converter: FuelConverter,
        reversible_energy_storage: ReversibleEnergyStorage,
        trans: Transmission,
        fuel_res_split: Option<f64>,
        fuel_res_ratio: Option<f64>,
        gss_interval: Option<usize>,
    ) -> Self {
        HybridElectricVehicle {
            fc: fuel_converter,
            res: reversible_energy_storage,
            trans,
            fuel_res_split: fuel_res_split.unwrap_or(0.5),
            fuel_res_ratio,
            gss_interval,
            dt: si::Time::ZERO,
            i: 1,
        }
    }

    /// cost function for fuel and res energy expenditure
    fn get_fuel_and_res_energy_cost(&mut self, x: f64) -> anyhow::Result<f64> {
        todo!() // might deem this method as something to be deleted
    }

    /// Solve fc and res energy consumption
    /// Arguments:
    /// - pwr_out_req: tractive power require
    /// - dt: time step size
    pub fn solve_energy_consumption(
        &mut self,
        pwr_out_req: si::Power,
        dt: si::Time,
        assert_limits: bool,
    ) -> anyhow::Result<()> {
        // hybrid controls that determine the split between engine + generator and reversible_energy_storage
        todo!()
    }
}

struct CostFunc<'a> {
    locomotive: &'a HybridElectricVehicle,
}

impl ArgminOp for CostFunc<'_> {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, x: &Self::Param) -> Result<Self::Output, argmin::prelude::Error> {
        // todo: figure out how to do this without cloning
        // maybe cloning is actually good because it ensures that
        // the state is not modified from the previous time step
        let mut locomotive = self.locomotive.clone();
        locomotive
            .get_fuel_and_res_energy_cost(*x)
            .map_err(anyhow::Error::msg)
    }
}
