//! Module containing miscellaneous utility functions.

use directories::ProjectDirs;
use itertools::Itertools;
use lazy_static::lazy_static;
use ndarray::*;
use ndarray_stats::QuantileExt;
use regex::Regex;
use std::collections::HashSet;
use url::Url;

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
    let x_min = arr.min().unwrap();
    let x_max = arr.max().unwrap();
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

/// Generate all permutations of indices for a given *N*-dimensional array shape
///
/// # Arguments
/// * `shape` - Reference to shape of the *N*-dimensional array, as returned by `ndarray::ArrayBase::shape()`
///
/// # Returns
/// A `Vec<Vec<usize>>` where each inner `Vec<usize>` is one permutation of indices
///
/// # Example
/// ```rust
/// use fastsim_core::utils::get_index_permutations;
/// let shape = [3, 2, 2];
/// assert_eq!(
///     get_index_permutations(&shape),
///     [
///         [0, 0, 0],
///         [0, 0, 1],
///         [0, 1, 0],
///         [0, 1, 1],
///         [1, 0, 0],
///         [1, 0, 1],
///         [1, 1, 0],
///         [1, 1, 1],
///         [2, 0, 0],
///         [2, 0, 1],
///         [2, 1, 0],
///         [2, 1, 1],
///     ]
/// );
/// ```
///
pub fn get_index_permutations(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.is_empty() {
        return vec![vec![]];
    }
    shape
        .iter()
        .map(|&len| 0..len)
        .multi_cartesian_product()
        .collect()
}

/// Multilinear interpolation function, accepting any dimensionality *N*.
///
/// # Arguments
/// * `point` - An *N*-length array representing the interpolation point coordinates in each dimension
/// * `grid` - A grid containing the coordinates for each dimension,
///   i.e. `[[0.0, 1.0], [-0.5, 1.5]]` indicates x<sub>0</sub> = 0.0, x<sub>1</sub> = 1.0, y<sub>0</sub> = -0.5, y<sub>1</sub> = 1.5
/// * `values` - An *N*-dimensional [`ndarray::ArrayD`] containing the values at given grid coordinates
///
/// # Errors
/// This function returns an [`InterpolationError`] if any of the validation checks from [`validate_inputs`] fail,
/// or if any values surrounding supplied `point` are `NaN`.
///
/// # Examples
/// ## 1D Example
/// ```rust
/// use ndarray::prelude::*;
/// use fastsim_core::utils::multilinear;
///
/// let grid = [vec![0.0, 1.0, 4.0]];
/// let values = array![0.0, 2.0, 4.45].into_dyn();
///
/// let point_a = [0.82];
/// assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 1.64);
/// let point_b = [2.98];
/// assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 3.617);
/// let point_c = [grid[0][2]]; // returns value at x2
/// assert_eq!(multilinear(&point_c, &grid, &values).unwrap(), values[2]);
/// ```
///
/// ## 2D Example
/// ```rust
/// use ndarray::prelude::*;
/// use fastsim_core::utils::multilinear;
///
/// let grid = [
///     vec![0.0, 1.0, 2.0], // x0, x1, x2
///     vec![0.0, 1.0, 2.0], // y0, y1, y2
/// ];
/// let values = array![
///     [0.0, 2.0, 1.9], // (x0, y0), (x0, y1), (x0, y2)
///     [2.0, 4.0, 3.1], // (x1, y0), (x1, y1), (x1, y2)
///     [5.0, 0.0, 1.4], // (x2, y0), (x2, y1), (x2, y2)
/// ]
/// .into_dyn();
///
/// let point_a = [0.5, 0.5];
/// assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 2.0);
/// let point_b = [1.52, 0.36];
/// assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 2.9696);
/// let point_c = [grid[0][2], grid[1][1]]; // returns value at (x2, y1)
/// assert_eq!(
///     multilinear(&point_c, &grid, &values).unwrap(),
///     values[[2, 1]]
/// );
/// ```
///
/// ## 3D Example
/// ```rust
/// use ndarray::prelude::*;
/// use fastsim_core::utils::multilinear;
///
/// let grid = [
///     vec![0.0, 1.0, 2.0], // x0, x1, x2
///     vec![0.0, 1.0, 2.0], // y0, y1, y2
///     vec![0.0, 1.0, 2.0], // z0, z1, z2
/// ];
/// let values = array![
///     [
///         [0.0, 1.5, 3.0], // (x0, y0, z0), (x0, y0, z1), (x0, y0, z2)
///         [2.0, 0.5, 1.4], // (x0, y1, z0), (x0, y1, z1), (x0, y1, z2)
///         [1.9, 5.3, 2.2], // (x0, y2, z0), (x0, y0, z1), (x0, y2, z2)
///     ],
///     [
///         [2.0, 5.1, 1.1], // (x1, y0, z0), (x1, y0, z1), (x1, y0, z2)
///         [4.0, 1.0, 0.5], // (x1, y1, z0), (x1, y1, z1), (x1, y1, z2)
///         [3.1, 0.9, 1.2], // (x1, y2, z0), (x1, y2, z1), (x1, y2, z2)
///     ],
///     [
///         [5.0, 0.2, 5.1], // (x2, y0, z0), (x2, y0, z1), (x2, y0, z2)
///         [0.7, 0.1, 3.2], // (x2, y1, z0), (x2, y1, z1), (x2, y1, z2)
///         [1.4, 1.1, 0.0], // (x2, y2, z0), (x2, y2, z1), (x2, y2, z2)
///     ],
/// ]
/// .into_dyn();
///
/// let point_a = [0.5, 0.5, 0.5];
/// assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 2.0125);
/// let point_b = [1.52, 0.36, 0.5];
/// assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 2.46272);
/// let point_c = [grid[0][2], grid[1][1], grid[2][0]]; // returns value at (x2, y1, z0)
/// assert_eq!(
///     multilinear(&point_c, &grid, &values).unwrap(),
///     values[[2, 1, 0]]
/// );
/// ```
///
pub fn multilinear(point: &[f64], grid: &[Vec<f64>], values: &ArrayD<f64>) -> anyhow::Result<f64> {
    // Dimensionality
    let mut n = values.ndim();

    // Validate inputs
    anyhow::ensure!(
        point.len() == n,
        "Length of supplied `point` must be same as `values` dimensionality: {point:?} is not {n}-dimensional",
    );
    anyhow::ensure!(
        grid.len() == n,
        "Length of supplied `grid` must be same as `values` dimensionality: {grid:?} is not {n}-dimensional",
    );
    for i in 0..n {
        anyhow::ensure!(
            grid[i].len() == values.shape()[i],
            "Supplied `grid` and `values` are not compatible shapes: dimension {i}, lengths {} != {}",
            grid[i].len(),
            values.shape()[i]
        );
        anyhow::ensure!(
            grid[i].windows(2).all(|w| w[0] < w[1]),
            "Supplied `grid` coordinates must be sorted and non-repeating: dimension {i}, {:?}",
            grid[i]
        );
        anyhow::ensure!(
            grid[i][0] <= point[i] && point[i] <= *grid[i].last().unwrap(),
            "Supplied `point` must be within `grid` for dimension {i}: point[{i}] = {:?}, grid[{i}] = {:?}",
            point[i],
            grid[i],
        );
    }

    // Point can share up to N values of a grid point, which reduces the problem dimensionality
    // i.e. the point shares one of three values of a 3-D grid point, then the interpolation becomes 2-D at that slice
    // or   if the point shares two of three values of a 3-D grid point, then the interpolation becomes 1-D
    let mut point = point.to_vec();
    let mut grid = grid.to_vec();
    let mut values_view = values.view();
    for dim in (0..n).rev() {
        // Range is reversed so that removal doesn't affect indexing
        if let Some(pos) = grid[dim]
            .iter()
            .position(|&grid_point| grid_point == point[dim])
        {
            point.remove(dim);
            grid.remove(dim);
            values_view.index_axis_inplace(Axis(dim), pos);
        }
    }
    if values_view.len() == 1 {
        // Supplied point is coincident with a grid point, so just return the value
        return Ok(*values_view.first().unwrap());
    }
    // Simplified dimensionality
    n = values_view.ndim();

    // Extract the lower and upper indices for each dimension,
    // as well as the fraction of how far the supplied point is between the surrounding grid points
    let mut lower_idxs = Vec::with_capacity(n);
    let mut interp_diffs = Vec::with_capacity(n);
    for dim in 0..n {
        let lower_idx = grid[dim]
            .windows(2)
            .position(|w| w[0] < point[dim] && point[dim] < w[1])
            .unwrap();
        let interp_diff =
            (point[dim] - grid[dim][lower_idx]) / (grid[dim][lower_idx + 1] - grid[dim][lower_idx]);
        lower_idxs.push(lower_idx);
        interp_diffs.push(interp_diff);
    }
    // `interp_vals` contains all values surrounding the point of interest, starting with shape (2, 2, ...) in N dimensions
    // this gets mutated and reduces in dimension each iteration, filling with the next values to interpolate with
    // this ends up as a 0-dimensional array containing only the final interpolated value
    let mut interp_vals = values_view
        .slice_each_axis(|ax| {
            let lower = lower_idxs[ax.axis.0];
            Slice::from(lower..=lower + 1)
        })
        .to_owned();
    let mut index_permutations = get_index_permutations(interp_vals.shape());
    // This loop interpolates in each dimension sequentially
    // each outer loop iteration the dimensionality reduces by 1
    // `interp_vals` ends up as a 0-dimensional array containing only the final interpolated value
    for (dim, diff) in interp_diffs.iter().enumerate() {
        let next_dim = n - 1 - dim;
        let next_shape = vec![2; next_dim];
        // Indeces used for saving results of this dimensions interpolation results
        // assigned to `index_permutations` at end of loop to be used for indexing in next iteration
        let next_idxs = get_index_permutations(&next_shape);
        let mut intermediate_arr = Array::default(next_shape);
        for i in 0..next_idxs.len() {
            // `next_idxs` is always half the length of `index_permutations`
            let l = index_permutations[i].as_slice();
            let u = index_permutations[next_idxs.len() + i].as_slice();
            if dim == 0 {
                anyhow::ensure!(
                    !interp_vals[l].is_nan() && !interp_vals[u].is_nan(),
                    "Surrounding value(s) cannot be NaN:\npoint = {point:?},\ngrid = {grid:?},\nvalues = {values:?}"
                );
            }
            // This calculation happens 2^(n-1) times in the first iteration of the outer loop,
            // 2^(n-2) times in the second iteration, etc.
            intermediate_arr[next_idxs[i].as_slice()] =
                interp_vals[l] * (1.0 - diff) + interp_vals[u] * diff;
        }
        index_permutations = next_idxs;
        interp_vals = intermediate_arr;
    }

    // return the only value contained within the 0-dimensional array
    Ok(*interp_vals.first().unwrap())
}

lazy_static! {
    static ref TIRE_CODE_REGEX: Regex = Regex::new(
        r"(?i)[P|LT|ST|T]?((?:[0-9]{2,3}\.)?[0-9]+)/((?:[0-9]{1,2}\.)?[0-9]+) ?[B|D|R]?[x|\-| ]?((?:[0-9]{1,2}\.)?[0-9]+)[A|B|C|D|E|F|G|H|J|L|M|N]?"
    ).unwrap();
}

/// Calculate tire radius (in meters) from an [ISO metric tire code](https://en.wikipedia.org/wiki/Tire_code#ISO_metric_tire_codes)
///
/// # Arguments
/// * `tire_code` - A string containing a parsable ISO metric tire code
///
/// # Examples
/// ## Example 1:
///
/// ```rust
/// // Note the floating point imprecision in the result
/// use fastsim_core::utils::tire_code_to_radius;
/// let tire_code = "225/70Rx19.5G";
/// assert_eq!(tire_code_to_radius(&tire_code).unwrap(), 0.40514999999999995);
/// ```
///
/// ## Example 2:
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

/// Creates/gets an OS-specific data directory and returns the path.
pub fn create_project_subdir<P: AsRef<Path>>(subpath: P) -> anyhow::Result<PathBuf> {
    let proj_dirs = ProjectDirs::from("gov", "NREL", "fastsim").ok_or_else(|| {
        anyhow!("Could not build path to project directory: \"gov.NREL.fastsim\"")
    })?;
    let path = PathBuf::from(proj_dirs.config_dir()).join(subpath);
    std::fs::create_dir_all(path.as_path())?;
    Ok(path)
}

/// Returns the path to the OS-specific data directory, if it exists.
pub fn path_to_cache() -> anyhow::Result<PathBuf> {
    let proj_dirs = ProjectDirs::from("gov", "NREL", "fastsim").ok_or_else(|| {
        anyhow!("Could not build path to project directory: \"gov.NREL.fastsim\"")
    })?;
    Ok(PathBuf::from(proj_dirs.config_dir()))
}

/// Deletes FASTSim data directory, clearing its contents. If subpath is
/// provided, will only delete the subdirectory pointed to by the subpath,
/// rather than deleting the whole data directory. If the subpath is an empty
/// string, deletes the entire FASTSim directory.     
/// USE WITH CAUTION, as this function deletes ALL objects stored in the FASTSim
/// data directory or provided subdirectory.  
/// # Arguments  
/// - subpath: Subpath to a subdirectory within the FASTSim data directory. If
///   an empty string, the function will delete the whole FASTSim data
///   directory, clearing all its contents.  
/// Note: it is not possible to delete single files using this function, only
/// directories. If a single file needs deleting, the path_to_cache() function
/// can be used to find the FASTSim data directory location. The file can then
/// be found and manually deleted.
pub fn clear_cache<P: AsRef<Path>>(subpath: P) -> anyhow::Result<()> {
    let path = path_to_cache()?.join(subpath);
    Ok(std::fs::remove_dir_all(path)?)
}

/// takes an object from a url and saves it in the FASTSim data directory in a
/// rust_objects folder  
/// WARNING: if there is a file already in the data subdirectory with the same
/// name, it will be replaced by the new file   
/// to save to a folder other than rust_objects, define constant CACHE_FOLDER to
/// be the desired folder name  
/// # Arguments  
/// - url: url (either as a string or url type) to object  
/// - subpath: path to subdirectory within FASTSim data directory. Suggested
/// paths are "vehicles" for a RustVehicle, "cycles" for a RustCycle, and
/// "rust_objects" for other Rust objects.  
/// Note: In order for the file to be save in the proper format, the URL needs
/// to be a URL pointing directly to a file, for example a raw github URL.
pub fn url_to_cache<S: AsRef<str>, P: AsRef<Path>>(url: S, subpath: P) -> anyhow::Result<()> {
    let url = Url::parse(url.as_ref())?;
    let file_name = url
        .path_segments()
        .and_then(|segments| segments.last())
        .with_context(|| "Could not parse filename from URL: {url:?}")?;
    let data_subdirectory = create_project_subdir(subpath)
        .with_context(|| "Could not find or build Fastsim data subdirectory.")?;
    let file_path = data_subdirectory.join(file_name);
    download_file_from_url(url.as_ref(), &file_path)?;
    Ok(())
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

    #[test]
    fn test_path_to_cache() {
        let path = path_to_cache().unwrap();
        println!("{:?}", path);
    }

    #[test]
    fn test_clear_cache() {
        let temp_sub_dir = tempfile::TempDir::new_in(create_project_subdir("").unwrap()).unwrap();
        let sub_dir_path = temp_sub_dir.path().to_str().unwrap();
        let still_exists_before = std::fs::metadata(sub_dir_path).is_ok();
        assert_eq!(still_exists_before, true);
        url_to_cache("https://raw.githubusercontent.com/NREL/fastsim-vehicles/main/public/1110_2022_Tesla_Model_Y_RWD_opt45017.yaml", "").unwrap();
        clear_cache(sub_dir_path).unwrap();
        let still_exists = std::fs::metadata(sub_dir_path).is_ok();
        assert_eq!(still_exists, false);
        let path_to_vehicle = path_to_cache()
            .unwrap()
            .join("1110_2022_Tesla_Model_Y_RWD_opt45017.yaml");
        let vehicle_still_exists = std::fs::metadata(&path_to_vehicle).is_ok();
        assert_eq!(vehicle_still_exists, true);
        std::fs::remove_file(path_to_vehicle).unwrap();
    }
}
