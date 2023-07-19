use argmin::prelude::*;
use argmin::solver::goldensectionsearch::GoldenSectionSearch;

use super::powertrain::electric_drivetrain::ElectricDrivetrain;
use super::powertrain::fuel_converter::FuelConverter;
use super::powertrain::generator::Generator;
use super::powertrain::reversible_energy_storage::ReversibleEnergyStorage;
use super::powertrain::ElectricMachine;
use super::LocoTrait;
use crate::imports::*;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, HistoryMethods)]
/// Hybrid locomotive with both engine and reversible energy storage (aka battery)  
/// This type of locomotive is not likely to be widely prevalent due to modularity of consists.  
pub struct HybridLoco {
    #[has_state]
    pub fc: FuelConverter,
    #[has_state]
    pub gen: Generator,
    #[has_state]
    pub res: ReversibleEnergyStorage,
    #[has_state]
    pub edrv: ElectricDrivetrain,
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

impl SerdeAPI for HybridLoco {}

impl Default for HybridLoco {
    fn default() -> Self {
        Self {
            fc: FuelConverter::default(),
            gen: Generator::default(),
            res: ReversibleEnergyStorage::default(),
            edrv: ElectricDrivetrain::default(),
            fuel_res_split: 0.5,
            fuel_res_ratio: Some(3.0),
            gss_interval: Some(60),
            dt: si::Time::ZERO,
            i: 1,
        }
    }
}

impl LocoTrait for Box<HybridLoco> {
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()> {
        self.res.set_cur_pwr_out_max(pwr_aux.unwrap(), None, None)?;
        self.fc.set_cur_pwr_out_max(dt)?;

        self.gen
            .set_cur_pwr_max_out(self.fc.state.pwr_out_max, pwr_aux)?;

        self.edrv.set_cur_pwr_max_out(
            self.gen.state.pwr_elec_prop_out_max + self.res.state.pwr_prop_out_max,
            None,
        )?;

        self.edrv
            .set_cur_pwr_regen_max(self.res.state.pwr_regen_out_max)?;

        self.gen
            .set_pwr_rate_out_max(self.fc.pwr_out_max / self.fc.pwr_ramp_lag);
        self.edrv
            .set_pwr_rate_out_max(self.gen.state.pwr_rate_out_max);
        Ok(())
    }

    fn save_state(&mut self) {
        self.deref_mut().save_state();
    }

    fn step(&mut self) {
        self.deref_mut().step()
    }

    fn get_energy_loss(&self) -> si::Energy {
        self.fc.state.energy_loss
            + self.gen.state.energy_loss
            + self.res.state.energy_loss
            + self.edrv.state.energy_loss
    }
}

impl HybridLoco {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fuel_converter: FuelConverter,
        generator: Generator,
        reversible_energy_storage: ReversibleEnergyStorage,
        electric_drivetrain: ElectricDrivetrain,
        fuel_res_split: Option<f64>,
        fuel_res_ratio: Option<f64>,
        gss_interval: Option<usize>,
    ) -> Self {
        HybridLoco {
            fc: fuel_converter,
            gen: generator,
            res: reversible_energy_storage,
            edrv: electric_drivetrain,
            fuel_res_split: fuel_res_split.unwrap_or(0.5),
            fuel_res_ratio,
            gss_interval,
            dt: si::Time::ZERO,
            i: 1,
        }
    }

    /// cost function for fuel and res energy expenditure
    fn get_fuel_and_res_energy_cost(&mut self, x: f64) -> anyhow::Result<f64> {
        let pwr_edrv_in = &self.edrv.state.pwr_elec_prop_in;

        let pwr_from_res = self
            .res
            .state
            // might want this to be method call
            // ensure res power draw does not exceed current max power of res
            .pwr_prop_out_max
            .min(*pwr_edrv_in * (1.0 - x));
        // limit on fc already accounted for
        let pwr_from_gen = *pwr_edrv_in - pwr_from_res;
        self.gen.set_pwr_in_req(
            // fc can only handle positive power
            pwr_from_gen,
            50e3 * uc::W, // todo; fix this
            self.dt,
        )?;

        // todo: fix this function call
        // assumes all aux load on generator
        self.fc
            .solve_energy_consumption(self.gen.state.pwr_mech_in, self.dt, true, false)?;
        self.res
            .solve_energy_consumption(pwr_from_res, si::Power::ZERO, self.dt)?;
        // total weighted energy cost at this time step
        Ok((self.res.state.pwr_out_chemical
            + self.fuel_res_ratio.unwrap() * self.fc.state.pwr_fuel)
            .get::<si::watt>())
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

        self.edrv.set_pwr_in_req(pwr_out_req, dt)?;
        self.dt = dt;

        if self.edrv.state.pwr_elec_prop_in > si::Power::ZERO {
            // positive traction
            // todo: maybe make the loco_con calculate the aux load and split it up in here

            if self.fuel_res_ratio.is_some()
                && (self.i % self.gss_interval.unwrap_or(10) == 0 || self.i == 1)
            {
                // solve for best res / generator split if fuel_res_ratio is provided
                // and `i`, the iteration counter, is a multiple of `gss_interval` or
                // it's the first iteration.  `gss_interval` default is provided in `unwrap_or`
                // above
                let cost = CostFunc { locomotive: self };
                // If self.res.state.pwr_out_max is greater than self.edrv.state.pwr_elec_in,
                // then the lower bound can be zero, meaning 100% of the power could come from the RES.
                // Otherwise, the lower bound needs to be greater than zero.
                // If self.gen.state.pwr_elec_out_max is greater than self.edrv.state.pwr_elec_in,
                // then the upper bound needs to be 1, meaning 100% of the power could come from the gen.
                // Otherwise, the upper bound needs to be less than 1.

                let gss_bounds = vec![
                    (1.0 - (self.res.state.pwr_prop_out_max / self.edrv.state.pwr_elec_prop_in)
                        .get::<si::ratio>())
                    .clamp(0.0, 1.0),
                    (self.gen.state.pwr_elec_prop_out_max / self.edrv.state.pwr_elec_prop_in)
                        .get::<si::ratio>()
                        .clamp(0.0, 1.0),
                ];

                if gss_bounds[1] - gss_bounds[0] < 0.05 {
                    // if bounds are really close together, don't run the optimizer
                    // just use mean
                    self.fuel_res_split = gss_bounds.iter().sum::<f64>() / gss_bounds.len() as f64;
                } else {
                    let solver =
                        GoldenSectionSearch::new(gss_bounds[0], gss_bounds[1]).tolerance(0.01);

                    let res = Executor::new(
                        cost,
                        solver,
                        self.fuel_res_split.clamp(gss_bounds[0], gss_bounds[1]),
                    )
                    // .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
                    .max_iters(5)
                    .run()
                    .unwrap();

                    self.fuel_res_split = res.state.best_param;
                }

                // uncomment to enable debug printing
                // Python::with_gil(|py| {
                //     let locals = pyo3::types::PyDict::new(py);
                //     locals.set_item("gss_bounds", gss_bounds).unwrap();
                //     py.run("print(\"gss_bounds:\", gss_bounds)", None, Some(locals))
                //         .expect("printing gss_bounds failed");
                // });
            }

            let pwr_from_res = self
                .res
                .state
                .pwr_prop_out_max // limit to pwr_out_max
                .min(self.edrv.state.pwr_elec_prop_in * (1.0 - self.fuel_res_split));
            // limit on fc already accounted for
            let pwr_from_gen = self.edrv.state.pwr_elec_prop_in - pwr_from_res;
            self.gen.set_pwr_in_req(
                // fc can only handle positive power
                pwr_from_gen,
                50e3 * uc::W, // todo: fix this
                dt,
            )?;
            self.fc.solve_energy_consumption(
                self.gen.state.pwr_mech_in,
                dt,
                true,
                assert_limits,
            )?; // todo: fix this function call
                // asumes all aux load on generator
            self.res
                .solve_energy_consumption(pwr_from_res, si::Power::ZERO, dt)?;
        } else {
            // negative traction
            self.res.solve_energy_consumption(
                self.edrv.state.pwr_elec_prop_in,
                // assume all aux load on generator
                si::Power::ZERO,
                dt,
            )?;
            self.gen.set_pwr_in_req(
                // fc can only handle positive power
                si::Power::ZERO,
                50e3 * uc::W, // todo: fix this
                dt,
            )?;
            self.fc.solve_energy_consumption(
                self.gen.state.pwr_mech_in,
                dt,
                true,
                assert_limits,
            )?;
            // todo: fix this function call
        }
        self.i += 1; // iterate counter
        Ok(())
    }
}

struct CostFunc<'a> {
    locomotive: &'a HybridLoco,
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
