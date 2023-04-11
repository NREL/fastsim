//! Module containing miscellaneous utility functions.

use std::collections::HashSet;

use crate::imports::*;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;

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

/// return max <f64> of arr
pub fn arrmax(arr: &[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::max)
}

/// return min <f64> of arr
pub fn arrmin(arr: &[f64]) -> f64 {
    arr.iter().copied().fold(f64::NAN, f64::min)
}

/// return min <f64> of arr
pub fn ndarrmin(arr: &Array1<f64>) -> f64 {
    arr.to_vec().into_iter().reduce(f64::min).unwrap()
}

/// return max <f64> of arr
pub fn ndarrmax(arr: &Array1<f64>) -> f64 {
    arr.to_vec().into_iter().reduce(f64::max).unwrap()
}

/// return true if the array is all zeros
pub fn ndarrallzeros(arr: &Array1<f64>) -> bool {
    arr.iter().all(|x| *x == 0.0)
}

/// return cumsum <f64> of arr
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

#[cfg(feature = "pyo3")]
pub mod array_wrappers {
    use proc_macros::add_pyo3_api;

    use super::*;
    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayU32(Array1<u32>);

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayI32(Array1<i32>);

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
    pub struct Pyo3ArrayF64(Array1<f64>);

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
    pub struct Pyo3ArrayBool(Array1<bool>);

    /// Helper struct to allow Rust to return a Python class that will indicate to the user that it's a clone.  
    #[add_pyo3_api]
    #[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
    pub struct Pyo3VecF64(Vec<f64>);
}

#[cfg(feature = "pyo3")]
pub use array_wrappers::*;

#[cfg(test)]
mod tests {
    use super::*;

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
