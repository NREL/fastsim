//! Module for utility functions that support the vehicle struct.

use argmin::core::{CostFunction, Error, Executor, OptimizationResult, State};
use argmin::solver::neldermead::NelderMead;
use ndarray::{array, Array1};
use polynomial::Polynomial;

use crate::air::*;
use crate::cycle::RustCycle;
use crate::imports::*;
use crate::params::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use crate::simdrive::RustSimDrive;
use crate::vehicle::RustVehicle;

#[allow(non_snake_case)]
#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn abc_to_drag_coeffs(
    veh: &mut RustVehicle,
    a_lbf: f64,
    b_lbf__mph: f64,
    c_lbf__mph2: f64,
    custom_rho: Option<bool>,
    custom_rho_temp_degC: Option<f64>,
    custom_rho_elevation_m: Option<f64>,
    simdrive_optimize: Option<bool>,
    _show_plots: Option<bool>,
) -> (f64, f64) {
    // For a given vehicle and target A, B, and C coefficients;
    // calculate and return drag and rolling resistance coefficients.
    //
    // Arguments:
    // ----------
    // veh: vehicle.RustVehicle with all parameters correct except for drag and rolling resistance coefficients
    // a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
    // custom_rho: if True, use `air::get_rho()` to calculate the current ambient density
    // custom_rho_temp_degC: ambient temperature [degree C] for `get_rho()`;
    //     will only be used when `custom_rho` is True
    // custom_rho_elevation_m: location elevation [degree C] for `get_rho()`;
    //     will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
    // simdrive_optimize: if True, use `SimDrive` to optimize the drag and rolling resistance;
    //     otherwise, directly use target A, B, C to calculate the results
    // show_plots: if True, plots are shown

    let air_props: AirProperties = AirProperties::default();
    let props: RustPhysicalProperties = RustPhysicalProperties::default();
    let cur_ambient_air_density_kg__m3: f64 = if custom_rho.unwrap_or(false) {
        air_props.get_rho(custom_rho_temp_degC.unwrap_or(20.0), custom_rho_elevation_m)
    } else {
        props.air_density_kg_per_m3
    };

    let vmax_mph: f64 = 70.0;
    let a_newton: f64 = a_lbf * super::params::N_PER_LBF;
    let _b_newton__mps: f64 = b_lbf__mph * super::params::N_PER_LBF * super::params::MPH_PER_MPS;
    let c_newton__mps2: f64 = c_lbf__mph2
        * super::params::N_PER_LBF
        * super::params::MPH_PER_MPS
        * super::params::MPH_PER_MPS;

    let cd_len: usize = 300;

    let cyc: RustCycle = RustCycle::new(
        (0..cd_len as i32).map(f64::from).collect(),
        Array::linspace(vmax_mph / super::params::MPH_PER_MPS, 0.0, cd_len).to_vec(),
        vec![0.0; cd_len],
        vec![0.0; cd_len],
        String::from("cycle"),
    );

    // polynomial function for pounds vs speed
    let dyno_func_lb: Polynomial<f64> = Polynomial::new(vec![a_lbf, b_lbf__mph, c_lbf__mph2]);

    let drag_coef: f64;
    let wheel_rr_coef: f64;

    if simdrive_optimize.unwrap_or(true) {
        let cost: GetError = GetError {
            cycle: &cyc,
            vehicle: &veh,
            dyno_func_lb: &dyno_func_lb,
        };
        let solver: NelderMead<Array1<f64>, f64> =
            NelderMead::new(vec![array![0.0, 0.0], array![0.5, 0.0], array![0.5, 0.1]]);
        let res: OptimizationResult<_, _, _> = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .unwrap();
        let best_param: &Array1<f64> = res.state().get_best_param().unwrap();
        drag_coef = best_param[0];
        wheel_rr_coef = best_param[1];
    } else {
        drag_coef = c_newton__mps2 / (0.5 * veh.frontal_area_m2 * cur_ambient_air_density_kg__m3);
        wheel_rr_coef = a_newton / veh.veh_kg / props.a_grav_mps2;
    }

    veh.drag_coef = drag_coef;
    veh.wheel_rr_coef = wheel_rr_coef;

    return (drag_coef, wheel_rr_coef);
}

pub fn get_error_val(model: Array1<f64>, test: Array1<f64>, time_steps: Array1<f64>) -> f64 {
    // Returns time-averaged error for model and test signal.
    // Arguments:
    // ----------
    // model: array of values for signal from model
    // test: array of values for signal from test data
    // time_steps: array (or scalar for constant) of values for model time steps [s]
    // test: array of values for signal from test

    // Output:
    // -------
    // err: integral of absolute value of difference between model and
    // test per time

    assert!(
        model.len() == test.len() && test.len() == time_steps.len(),
        "{}, {}, {}",
        model.len(),
        test.len(),
        time_steps.len()
    );

    let mut err: f64 = 0.0;
    let y: Array1<f64> = (model - test).mapv(f64::abs);

    for index in 0..time_steps.len() - 1 {
        err += 0.5 * (time_steps[index + 1] - time_steps[index]) * (y[index] + y[index + 1]);
    }

    return err / (time_steps.last().unwrap() - time_steps[0]);
}

struct GetError<'a> {
    cycle: &'a RustCycle,
    vehicle: &'a RustVehicle,
    dyno_func_lb: &'a Polynomial<f64>,
}

impl CostFunction for GetError<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        let mut veh: RustVehicle = self.vehicle.clone();
        let cyc: RustCycle = self.cycle.clone();
        let dyno_func_lb: Polynomial<f64> = self.dyno_func_lb.clone();

        veh.drag_coef = x[0];
        veh.wheel_rr_coef = x[1];

        let mut sd_coast: RustSimDrive = RustSimDrive::new(self.cycle.clone(), veh);
        sd_coast.impose_coast = Array::from_vec(vec![true; sd_coast.impose_coast.len()]);
        let _sim_drive_result: Result<_, _> = sd_coast.sim_drive(None, None);

        let cutoff_vec: Vec<usize> = sd_coast
            .mps_ach
            .indexed_iter()
            .filter_map(|(index, &item)| (item < 0.1).then(|| index))
            .collect();
        let cutoff: usize;
        if cutoff_vec.len() == 0 {
            cutoff = sd_coast.mps_ach.len();
        } else {
            cutoff = cutoff_vec[0];
        }

        return Ok(get_error_val(
            (Array::from_vec(vec![1000.0; sd_coast.mps_ach.len()])
                * (sd_coast.drag_kw + sd_coast.rr_kw)
                / sd_coast.mps_ach)
                .slice_move(s![0..cutoff]),
            (sd_coast.mph_ach.map(|x| dyno_func_lb.eval(*x))
                * Array::from_vec(vec![super::params::N_PER_LBF; sd_coast.mph_ach.len()]))
            .slice_move(s![0..cutoff]),
            cyc.time_s.slice_move(s![0..cutoff]),
        ));
    }
}

#[cfg(test)]
mod vehicle_utils_tests {
    use super::*;

    #[test]
    fn test_get_error_val() {
        let time_steps: Array1<f64> = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let model: Array1<f64> = array![1.1, 4.6, 2.5, 3.7, 5.0];
        let test: Array1<f64> = array![2.1, 4.5, 3.4, 4.8, 6.3];

        let error_val: f64 = get_error_val(model, test, time_steps);
        println!("Error Value: {}", error_val);

        assert!(error_val.approx_eq(&0.8124999999999998, 1e-10));
    }

    #[test]
    fn test_abc_to_drag_coeffs() {
        let mut veh: RustVehicle = RustVehicle::mock_vehicle();
        let a: f64 = 25.91;
        let b: f64 = 0.1943;
        let c: f64 = 0.01796;

        let (drag_coef, wheel_rr_coef): (f64, f64) = abc_to_drag_coeffs(
            &mut veh,
            a,
            b,
            c,
            Some(false),
            None,
            None,
            Some(true),
            Some(false),
        );
        println!("Drag Coef: {}", drag_coef);
        println!("Wheel RR Coef: {}", wheel_rr_coef);

        assert!(drag_coef.approx_eq(&0.24676817210529464, 1e-5));
        assert!(wheel_rr_coef.approx_eq(&0.0068603812443132645, 1e-6));
        assert_eq!(drag_coef, veh.drag_coef);
        assert_eq!(wheel_rr_coef, veh.wheel_rr_coef);
    }
}
