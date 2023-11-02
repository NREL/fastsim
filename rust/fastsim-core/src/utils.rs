//! Module containing miscellaneous utility functions.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashSet;

use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;

#[cfg(test)]
pub fn resources_path() -> PathBuf {
    let pb = PathBuf::from("../../python/fastsim/resources");
    assert!(pb.exists());
    pb
}

/// Error message for when user attempts to set value in a nested struct.
pub const NESTED_STRUCT_ERR: &str = "Setting field value on nested struct not allowed.
Assign nested struct to own variable, run the `reset_orphaned` method, and then 
modify field value. Then set the nested struct back inside containing struct.";

pub fn diff(x: &Array1<f64>) -> Array1<f64> {
    concatenate(
        Axis(0),
        &[
            array![0.0].view(),
            (&x.slice(s![1..]) - &x.slice(s![..-1])).view(),
        ],
    )
    .unwrap()
}

/// Returns a new array with a constant added starting at xs\[i\] to the end. Values prior to xs\[i\] are unchanged.
pub fn add_from(xs: &Array1<f64>, i: usize, val: f64) -> Array1<f64> {
    let mut ys = Array1::zeros(xs.len());
    for idx in 0..xs.len() {
        if idx >= i {
            ys[idx] = xs[idx] + val;
        } else {
            ys[idx] = xs[idx];
        }
    }
    ys
}

/// Return first index of `arr` greater than `cut`
pub fn first_grtr(arr: &[f64], cut: f64) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x > cut).unwrap_or(len - 1)) // unwrap_or allows for default if not found
}

/// Return first index of `arr` equal to `cut`
pub fn first_eq(arr: &[f64], cut: f64) -> Option<usize> {
    let len = arr.len();
    if len == 0 {
        return None;
    }
    Some(arr.iter().position(|&x| x == cut).unwrap_or(len - 1)) // unwrap_or allows for default if not found
}

/// return max of 2 f64
pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// return min of 2 f64
pub fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// return max of arr
pub fn arrmax(arr: &[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::max)
}

/// return min of arr
pub fn arrmin(arr: &[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::min)
}

/// return min of arr
pub fn ndarrmin(arr: &Array1<f64>) -> f64 {
    arr.to_vec().into_iter().reduce(f64::min).unwrap()
}

/// return max of arr
pub fn ndarrmax(arr: &Array1<f64>) -> f64 {
    arr.to_vec().into_iter().reduce(f64::max).unwrap()
}

/// return true if the array is all zeros
pub fn ndarrallzeros(arr: &Array1<f64>) -> bool {
    arr.iter().all(|x| *x == 0.0)
}

/// return cumsum of arr
pub fn ndarrcumsum(arr: &Array1<f64>) -> Array1<f64> {
    arr.iter()
        .scan(0.0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect()
}

/// return the unique values of the array
pub fn ndarrunique(arr: &Array1<f64>) -> Array1<f64> {
    let mut set: HashSet<usize> = HashSet::new();
    let mut new_arr: Vec<f64> = Vec::new();
    let x_min = ndarrmin(arr);
    let x_max = ndarrmax(arr);
    let dx = if x_max == x_min { 1.0 } else { x_max - x_min };
    for &x in arr.iter() {
        let y = (((x - x_min) / dx) * (usize::MAX as f64)) as usize;
        if !set.contains(&y) {
            new_arr.push(x);
            set.insert(y);
        }
    }
    Array1::from_vec(new_arr)
}

/// interpolation algorithm from <http://www.cplusplus.com/forum/general/216928/>
/// Arguments:
/// x : value at which to interpolate
pub fn interpolate(
    x: &f64,
    x_data_in: &Array1<f64>,
    y_data_in: &Array1<f64>,
    extrapolate: bool,
) -> f64 {
    assert!(x_data_in.len() == y_data_in.len());
    let mut new_x_data: Vec<f64> = Vec::new();
    let mut new_y_data: Vec<f64> = Vec::new();
    let mut last_x = x_data_in[0];
    for idx in 0..x_data_in.len() {
        if idx == 0 || (idx > 0 && x_data_in[idx] > last_x) {
            last_x = x_data_in[idx];
            new_x_data.push(x_data_in[idx]);
            new_y_data.push(y_data_in[idx]);
        }
    }
    let x_data = Array1::from_vec(new_x_data);
    let y_data = Array1::from_vec(new_y_data);
    let size = x_data.len();

    let mut i = 0;
    if x >= &x_data[size - 2] {
        i = size - 2;
    } else {
        while x > &x_data[i + 1] {
            i += 1;
        }
    }
    let xl = &x_data[i];
    let mut yl = &y_data[i];
    let xr = &x_data[i + 1];
    let mut yr = &y_data[i + 1];
    if !extrapolate {
        if x < xl {
            yr = yl;
        }
        if x > xr {
            yl = yr;
        }
    }
    let dydx = (yr - yl) / (xr - xl);
    yl + dydx * (x - xl)
}

/// interpolation algorithm from <http://www.cplusplus.com/forum/general/216928/>
/// Arguments:
/// x : value at which to interpolate
pub fn interpolate_vectors(
    x: &f64,
    x_data_in: &Vec<f64>,
    y_data_in: &Vec<f64>,
    extrapolate: bool,
) -> f64 {
    assert!(x_data_in.len() == y_data_in.len());
    let mut new_x_data: Vec<f64> = Vec::new();
    let mut new_y_data: Vec<f64> = Vec::new();
    let mut last_x = x_data_in[0];
    for idx in 0..x_data_in.len() {
        if idx == 0 || (idx > 0 && x_data_in[idx] > last_x) {
            last_x = x_data_in[idx];
            new_x_data.push(x_data_in[idx]);
            new_y_data.push(y_data_in[idx]);
        }
    }
    let x_data = new_x_data;
    let y_data = new_y_data;
    let size = x_data.len();

    let mut i = 0;
    if x >= &x_data[size - 2] {
        i = size - 2;
    } else {
        while x > &x_data[i + 1] {
            i += 1;
        }
    }
    let xl = &x_data[i];
    let mut yl = &y_data[i];
    let xr = &x_data[i + 1];
    let mut yr = &y_data[i + 1];
    if !extrapolate {
        if x < xl {
            yr = yl;
        }
        if x > xr {
            yl = yr;
        }
    }
    let dydx = (yr - yl) / (xr - xl);
    yl + dydx * (x - xl)
}
/// helper function to find where a query falls on an axis of discrete values;
/// NOTE: this assumes the axis array is sorted with values ascending and that there are no repeating values!
fn find_interp_indices(query: &f64, axis: &[f64]) -> anyhow::Result<(usize, usize)> {
    let axis_size = axis.len();
    match axis
        .windows(2)
        .position(|w| query >= &w[0] && query < &w[1])
    {
        Some(p) => {
            if query == &axis[p] {
                Ok((p, p))
            } else if query == &axis[p + 1] {
                Ok((p + 1, p + 1))
            } else {
                Ok((p, p + 1))
            }
        }
        None => {
            if query <= &axis[0] {
                Ok((0, 0))
            } else if query >= &axis[axis_size - 1] {
                Ok((axis_size - 1, axis_size - 1))
            } else {
                bail!("Unable to find where the query fits in the values, check grid.")
            }
        }
    }
}

/// Helper function to compute the difference between a value and a set of bounds
fn compute_interp_diff(value: &f64, lower: &f64, upper: &f64) -> f64 {
    if lower == upper {
        0.0
    } else {
        (value - lower) / (upper - lower)
    }
}

/// Bilinear interpolation over a structured grid;
pub fn interp2d(
    point: &[f64; 2],
    grid: &[Vec<f64>; 2],
    values: &[Vec<f64>],
) -> anyhow::Result<f64> {
    let x = point[0];
    let y = point[1];

    let x_points = &grid[0];
    let y_points = &grid[1];

    // find indeces of x-values that surround the specified x-value
    let (xi0, xi1) = find_interp_indices(&x, x_points)?;
    // which indeces of y-values that surround the specified y-value
    let (yi0, yi1) = find_interp_indices(&y, y_points)?;

    // calculate fraction of position of specified x-value between the lower and upper x bounds
    let xd = compute_interp_diff(&x, &x_points[xi0], &x_points[xi1]);
    // calculate fraction of position of specified y-value between the lower and upper y bounds
    let yd = compute_interp_diff(&y, &y_points[yi0], &y_points[yi1]);

    // extract values at four surrounding points
    let c00 = values[xi0][yi0]; // lower left
    let c10 = values[xi1][yi0]; // lower right
    let c01 = values[xi0][yi1]; // upper left
    let c11 = values[xi1][yi1]; // upper right

    // interpolate in the x-direction
    let c0 = c00 * (1.0 - xd) + c10 * xd;
    let c1 = c01 * (1.0 - xd) + c11 * xd;

    // interpolate in the y-direction
    let c = c0 * (1.0 - yd) + c1 * yd;

    // return result
    Ok(c)
}

lazy_static! {
    static ref TIRE_CODE_REGEX: Regex = Regex::new(
        r"(?i)[P|LT|ST|T]?((?:[0-9]{2,3}\.)?[0-9]+)/((?:[0-9]{1,2}\.)?[0-9]+) ?[B|D|R]?[x|\-| ]?((?:[0-9]{1,2}\.)?[0-9]+)[A|B|C|D|E|F|G|H|J|L|M|N]?"
    ).unwrap();
}

/// Calculate tire radius (in meters) from an [ISO metric tire code](https://en.wikipedia.org/wiki/Tire_code#ISO_metric_tire_codes)
///
/// **Example 1:**
///
/// ```rust
/// // Note the floating point imprecision in the result
/// use fastsim_core::utils::tire_code_to_radius;
/// let tire_code = "225/70Rx19.5G";
/// assert_eq!(tire_code_to_radius(&tire_code).unwrap(), 0.40514999999999995);
/// ```
///
/// **Example 2:**
///
/// ```rust
/// // Either `&str`, `&String`, or `String` can be passed
/// use fastsim_core::utils::tire_code_to_radius;
/// let tire_code = String::from("P205/60R16");
/// assert_eq!(tire_code_to_radius(tire_code).unwrap(), 0.3262);
/// ```
///
pub fn tire_code_to_radius<S: AsRef<str>>(tire_code: S) -> anyhow::Result<f64> {
    let tire_code = tire_code.as_ref();
    let captures = TIRE_CODE_REGEX.captures(tire_code).with_context(|| {
        format!(
            "Regex pattern does not match for {:?}: {:?}",
            tire_code,
            TIRE_CODE_REGEX.as_str(),
        )
    })?;
    let width_mm: f64 = captures[1].parse()?;
    let aspect_ratio: f64 = captures[2].parse()?;
    let rim_diameter_in: f64 = captures[3].parse()?;

    let sidewall_height_mm = width_mm * aspect_ratio / 100.0;
    let radius_mm = (rim_diameter_in * 25.4) / 2.0 + sidewall_height_mm;

    Ok(radius_mm / 1000.0)
}

#[cfg(feature = "pyo3")]
pub mod array_wrappers {
    use crate::proc_macros::add_pyo3_api;

    use super::*;
    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayU32(Array1<u32>);
    impl SerdeAPI for Pyo3ArrayU32 {}

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayI32(Array1<i32>);
    impl SerdeAPI for Pyo3ArrayI32 {}

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
    pub struct Pyo3ArrayF64(Array1<f64>);
    impl SerdeAPI for Pyo3ArrayF64 {}

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayBool(Array1<bool>);
    impl SerdeAPI for Pyo3ArrayBool {}

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
    pub struct Pyo3VecF64(Vec<f64>);

    impl SerdeAPI for Pyo3VecF64 {}
}

#[cfg(feature = "pyo3")]
pub use array_wrappers::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp2d() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [0.5, 0.5];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 0.5);
        Ok(())
    }

    #[test]
    fn test_interp2d_offset() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [0.25, 0.75];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 0.375);
        Ok(())
    }

    #[test]
    fn test_interp2d_exact_value_lower() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [0.0, 0.0];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 1.0);
        Ok(())
    }

    #[test]
    fn test_interp2d_below_value_lower() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [-1.0, -1.0];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 1.0);
        Ok(())
    }

    #[test]
    fn test_interp2d_above_value_upper() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [2.0, 2.0];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 1.0);
        Ok(())
    }

    #[test]
    fn test_interp2d_exact_value_upper() -> anyhow::Result<()> {
        // specified (x, y) point at which to interpolate value
        let point = [1.0, 1.0];
        // grid coordinates: (x0, x1), (y0, y1)
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0]];
        // values at grid points
        let values = [
            vec![
                1.0, // lower left (x0, y0)
                0.0, // upper left (x0, y1)
            ],
            vec![
                0.0, // lower right (x1, y0)
                1.0, // upper right (x1, y1)
            ],
        ];
        anyhow::ensure!(interp2d(&point, &grid, &values)? == 1.0);
        Ok(())
    }

    #[test]
    fn test_diff() {
        assert_eq!(diff(&Array1::range(0.0, 3.0, 1.0)), array![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_that_first_eq_finds_the_right_index_when_one_exists() {
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_eq(&xs, 3.3).unwrap();
        let expected_idx: usize = 2;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_first_eq_yields_last_index_when_nothing_found() {
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_eq(&xs, 7.0).unwrap();
        let expected_idx: usize = xs.len() - 1;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_first_grtr_finds_the_right_index_when_one_exists() {
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_grtr(&xs, 3.0).unwrap();
        let expected_idx: usize = 2;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_first_grtr_yields_last_index_when_nothing_found() {
        let xs: [f64; 5] = [0.0, 1.2, 3.3, 4.4, 6.6];
        let idx = first_grtr(&xs, 7.0).unwrap();
        let expected_idx: usize = xs.len() - 1;
        assert_eq!(idx, expected_idx)
    }

    #[test]
    fn test_that_ndarrmin_returns_the_min() {
        let xs = Array1::from_vec(vec![10.0, 80.0, 3.0, 3.2, 9.0]);
        let xmin = ndarrmin(&xs);
        assert_eq!(xmin, 3.0);
    }

    #[test]
    fn test_that_ndarrmax_returns_the_max() {
        let xs = Array1::from_vec(vec![10.0, 80.0, 3.0, 3.2, 9.0]);
        let xmax = ndarrmax(&xs);
        assert_eq!(xmax, 80.0);
    }

    #[test]
    fn test_ndarrcumsum_expected_output() {
        let xs = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let expected_ys = Array1::from_vec(vec![0.0, 1.0, 3.0, 6.0]);
        let ys = ndarrcumsum(&xs);
        for (i, (ye, y)) in expected_ys.iter().zip(ys.iter()).enumerate() {
            assert_eq!(ye, y, "unequal at {}", i);
        }
    }

    #[test]
    fn test_add_from_yields_expected_output() {
        let xs = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut expected_ys = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut actual_ys = add_from(&xs, 100, 1.0);
        assert_eq!(expected_ys.len(), actual_ys.len());
        assert_eq!(expected_ys, actual_ys);
        expected_ys = Array1::from_vec(vec![1.0, 2.0, 4.0, 5.0, 6.0]);
        actual_ys = add_from(&xs, 2, 1.0);
        assert_eq!(expected_ys.len(), actual_ys.len());
        assert_eq!(expected_ys, actual_ys);
    }

    #[test]
    fn test_ndarrunique_works() {
        let xs = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 10.0, 10.0, 11.0]);
        let expected = Array1::from_vec(vec![0.0, 1.0, 2.0, 10.0, 11.0]);
        let actual = ndarrunique(&xs);
        assert_eq!(expected.len(), actual.len());
        for (ex, act) in expected.iter().zip(actual.iter()) {
            assert_eq!(ex, act);
        }
    }
    // #[test]
    // fn test_that_argmax_does_the_right_thing_on_an_empty_array(){
    //     let xs: Array1<bool> = Array::from_vec(vec![]);
    //     let idx = first_grtr(&xs);
    //     // unclear what should happen here; np.argmax throws a ValueError in the case of an empty vector
    //     // ... possibly we should return an Option type?
    //     let expected_idx:Option<usize> = None;
    //     assert_eq!(idx, expected_idx);
    // }

    #[test]
    fn test_that_interpolation_works() {
        let xs = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        let x = 0.5;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 5.0;
        assert_eq!(expected_y_lookup, y_lookup);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert_eq!(expected_y_lookup, y_lookup);
    }

    #[test]
    fn test_that_interpolation_works_for_irrational_number() {
        let xs = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0, 40.0]);
        let x = 1.0 / 3.0;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 3.3333333333;
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_with_small_vectors() {
        let xs = Array1::from_vec(vec![0.0, 1.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0]);
        let x = 0.5;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 5.0;
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_when_lookup_is_at_end() {
        let xs = Array1::from_vec(vec![0.0, 1.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0]);
        let x = 1.0;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 10.0;
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_when_lookup_is_past_end_without_extrapolate() {
        let xs = Array1::from_vec(vec![0.0, 1.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0]);
        let x = 1.01;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 10.0;
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert!((expected_y_lookup - y_lookup).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_with_x_data_that_repeats() {
        let xs = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let ys = Array1::from_vec(vec![0.0, 10.0, 10.0]);
        let x = 1.0;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 10.0;
        assert_eq!(expected_y_lookup, y_lookup);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert_eq!(expected_y_lookup, y_lookup);
    }

    #[test]
    fn test_interpolate_with_non_evenly_spaced_x_data() {
        let xs = Array1::from_vec(vec![0.0, 10.0, 100.0, 1000.0]);
        let ys = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let x = 55.0;
        let y_lookup = interpolate(&x, &xs, &ys, false);
        let expected_y_lookup = 1.5;
        assert_eq!(expected_y_lookup, y_lookup);
        let y_lookup = interpolate_vectors(&x, &xs.to_vec(), &ys.to_vec(), false);
        assert_eq!(expected_y_lookup, y_lookup);
    }
}
