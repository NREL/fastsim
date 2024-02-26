//! Module for utility functions that support the vehicle struct.

#[cfg(feature = "default")]
use argmin::core::{CostFunction, Executor, OptimizationResult, State};
#[cfg(feature = "default")]
use argmin::solver::neldermead::NelderMead;
use ndarray::{array, Array1};
#[cfg(feature = "default")]
use polynomial::Polynomial;
use std::{result::Result, thread, time::Duration};
use ureq::{Error as OtherError, Error::Status, Response};

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
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "default")]
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
    let props = RustPhysicalProperties::default();
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

    let cyc: RustCycle = RustCycle {
        time_s: (0..cd_len as i32).map(f64::from).collect(),
        mps: Array::linspace(vmax_mph / super::params::MPH_PER_MPS, 0.0, cd_len),
        grade: Array::zeros(cd_len),
        road_type: Array::zeros(cd_len),
        name: String::from("cycle"),
        orphaned: false,
    };

    // polynomial function for pounds vs speed
    let dyno_func_lb: Polynomial<f64> = Polynomial::new(vec![a_lbf, b_lbf__mph, c_lbf__mph2]);

    let drag_coef: f64;
    let wheel_rr_coef: f64;

    if simdrive_optimize.unwrap_or(true) {
        let cost: GetError = GetError {
            cycle: &cyc,
            vehicle: veh,
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

    (drag_coef, wheel_rr_coef)
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

#[cfg(feature = "default")]
struct GetError<'a> {
    cycle: &'a RustCycle,
    vehicle: &'a RustVehicle,
    dyno_func_lb: &'a Polynomial<f64>,
}

#[cfg(feature = "default")]
impl CostFunction for GetError<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, x: &Self::Param) -> anyhow::Result<Self::Output> {
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
            .filter_map(|(index, &item)| (item < 0.1).then_some(index))
            .collect();
        let cutoff: usize = if cutoff_vec.is_empty() {
            sd_coast.mps_ach.len()
        } else {
            cutoff_vec[0]
        };

        Ok(get_error_val(
            (Array::from_vec(vec![1000.0; sd_coast.mps_ach.len()])
                * (sd_coast.drag_kw + sd_coast.rr_kw)
                / sd_coast.mps_ach)
                .slice_move(s![0..cutoff]),
            (sd_coast.mph_ach.map(|x| dyno_func_lb.eval(*x))
                * Array::from_vec(vec![super::params::N_PER_LBF; sd_coast.mph_ach.len()]))
            .slice_move(s![0..cutoff]),
            cyc.time_s.slice_move(s![0..cutoff]),
        ))
    }
}

/// Given the path to a zip archive, print out the names of the files within that archive
pub fn list_zip_contents(filepath: &Path) -> anyhow::Result<()> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    for i in 0..zip.len() {
        let file = zip.by_index(i)?;
        println!("Filename: {}", file.name());
    }
    Ok(())
}

/// Extract zip archive at filepath to destination directory at dest_dir
pub fn extract_zip(filepath: &Path, dest_dir: &Path) -> anyhow::Result<()> {
    let f = File::open(filepath)?;
    let mut zip = zip::ZipArchive::new(f)?;
    zip.extract(dest_dir)?;
    Ok(())
}

#[derive(Deserialize)]
pub struct ObjectLinks {
    #[serde(rename = "self")]
    pub self_url: Option<String>,
    pub git: Option<String>,
    pub html: Option<String>,
}

#[derive(Deserialize)]
pub struct GitObjectInfo {
    pub name: String,
    pub path: String,
    pub sha: Option<String>,
    pub size: Option<i64>,
    pub url: String,
    pub html_url: Option<String>,
    pub git_url: Option<String>,
    pub download_url: Option<String>,
    #[serde(rename = "type")]
    pub url_type: String,
    #[serde(rename = "_links")]
    pub links: Option<ObjectLinks>,
}

const VEHICLE_REPO_LIST_URL: &'static str =
    &"https://api.github.com/repos/NREL/fastsim-vehicles/contents/public";

/// Function that takes a url and calls the url. If a 503 or 429 error is
/// thrown, it tries again after a pause, up to four times. It returns either a
/// result or an error.  
/// # Arguments  
/// - url: url to be called
/// Source: https://docs.rs/ureq/latest/ureq/enum.Error.html
fn get_response<S: AsRef<str>>(url: S) -> Result<Response, OtherError> {
    for _ in 1..4 {
        match ureq::get(url.as_ref()).call() {
            Err(Status(503, r)) | Err(Status(429, r)) | Err(Status(403, r)) => {
                let retry: Option<u64> = r.header("retry-after").and_then(|h| h.parse().ok());
                let retry = retry.unwrap_or(5);
                eprintln!("{} for {}, retry in {}", r.status(), r.get_url(), retry);
                thread::sleep(Duration::from_secs(retry));
            }
            result => return result,
        };
    }
    // Ran out of retries; try one last time and return whatever result we get.
    ureq::get(url.as_ref()).call()
}

/// Returns a list of vehicle file names in the Fastsim Vehicle Repo, or,
/// optionally, a different GitHub repo, in which case the url provided needs to
/// be the url for the file tree within GitHub for the root folder the Rust
/// objects, for example
/// "https://api.github.com/repos/NREL/fastsim-vehicles/contents/public"  
/// Note: for each file, the output will list the vehicle file name, including
/// the path from the root of the repository  
/// # Arguments  
/// - repo_url: url to the GitHub repository, Option, if None, defaults to the
///   FASTSim Vehicle Repo
pub fn fetch_github_list(repo_url: Option<String>) -> anyhow::Result<Vec<String>> {
    let repo_url = repo_url.unwrap_or(VEHICLE_REPO_LIST_URL.to_string());
    let response = get_response(repo_url)?.into_reader();
    let github_list: Vec<GitObjectInfo> =
        serde_json::from_reader(response).with_context(|| "Cannot parse github vehicle list.")?;
    let mut vehicle_name_list: Vec<String> = Vec::new();
    for object in github_list.iter() {
        if object.url_type == "dir" {
            let url = &object.url;
            let vehicle_name_sublist = fetch_github_list(Some(url.to_owned()))?;
            for name in vehicle_name_sublist.iter() {
                vehicle_name_list.push(name.to_owned());
            }
        } else if object.url_type == "file" {
            let url = url::Url::parse(&object.url)?;
            let path = &object.path;
            let format = url
                .path_segments()
                .and_then(|segments| segments.last())
                .and_then(|filename| Path::new(filename).extension())
                .and_then(OsStr::to_str)
                .with_context(|| "Could not parse file format from URL: {url:?}")?;
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => vehicle_name_list.push(path.to_owned()),
                "json" => vehicle_name_list.push(path.to_owned()),
                _ => continue,
            }
        } else {
            continue;
        }
    }
    Ok(vehicle_name_list)
}

#[cfg(test)]
mod tests {
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

    #[cfg(feature = "default")]
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

    // NOTE: this test does not seem to reliably pass. Sometimes the function
    // will give a 403 error and sometimes it will succeed -- I don't think
    // there's any way to ensure the function succeeds 100% of the time.
    #[test]
    fn test_fetch_github_list() {
        let list = fetch_github_list(Some(
            "https://api.github.com/repos/NREL/fastsim-vehicles/contents".to_owned(),
        ))
        .unwrap();
        let other_list = fetch_github_list(None).unwrap();
        println!("{:?}", list);
        println!("{:?}", other_list);
    }
}
