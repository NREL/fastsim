use crate::imports::*;
use itertools::Itertools;
use lazy_static::lazy_static;
use paste::paste;
use regex::Regex;

/// Error message for when user attempts to set value in a nested struct.
pub const DIRECT_SET_ERR: &str =
    "Setting field value directly not allowed. Please use fastsim.set_param_from_path() method.";

/// returns true for use with serde default
pub fn return_true() -> bool {
    true
}

/// Function for sorting a slice that implements `std::cmp::PartialOrd`.
/// Remove this once is_sorted is stabilized in std
pub fn is_sorted<T: std::cmp::PartialOrd>(data: &[T]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

/// Download a file to a specified filepath, assuming all necessary parent directories exist.
///
/// If supplied filepath has no file extension,
/// this function will attempt to parse a filename from the last segment of the URL.
#[cfg(feature = "web")]
#[allow(dead_code)]
pub(crate) fn download_file<S: AsRef<str>, P: AsRef<Path>>(
    url: S,
    filepath: P,
) -> anyhow::Result<()> {
    let url = url::Url::parse(url.as_ref())?;
    let filepath = filepath.as_ref();
    let filepath = if filepath.extension().is_none() {
        // No extension in filepath, parse from URL
        let filename = url
            .path_segments()
            .and_then(|segments| segments.last())
            .with_context(|| "Could not parse filename from last URL segment: {url:?}")?;
        filepath.join(filename)
    } else {
        filepath.to_path_buf()
    };
    let mut rdr = ureq::get(url.as_ref()).call()?.into_reader();
    let mut wtr = File::create(filepath)?;
    std::io::copy(&mut rdr, &mut wtr)?;
    Ok(())
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

/// Trilinear interpolation over a structured grid;
/// NOTE: this could be generalized to compute a linear interpolation in N dimensions
/// NOTE: this function assumes the each axis on the grid is sorted and that there
/// are no repeating values on each axis
pub fn interp3d(
    point: &[f64; 3],
    grid: &[Vec<f64>; 3],
    values: &[Vec<Vec<f64>>],
) -> anyhow::Result<f64> {
    let x = point[0];
    let y = point[1];
    let z = point[2];

    let x_points = &grid[0];
    let y_points = &grid[1];
    let z_points = &grid[2];

    let (xi0, xi1) = find_interp_indices(&x, x_points).with_context(|| anyhow!(format_dbg!()))?;
    let (yi0, yi1) = find_interp_indices(&y, y_points).with_context(|| anyhow!(format_dbg!()))?;
    let (zi0, zi1) = find_interp_indices(&z, z_points).with_context(|| anyhow!(format_dbg!()))?;

    let xd = compute_interp_diff(&x, &x_points[xi0], &x_points[xi1]);
    let yd = compute_interp_diff(&x, &x_points[xi0], &x_points[xi1]);
    let zd = compute_interp_diff(&x, &x_points[xi0], &x_points[xi1]);

    let c000 = values[xi0][yi0][zi0];
    let c100 = values[xi1][yi0][zi0];
    let c001 = values[xi0][yi0][zi1];
    let c101 = values[xi1][yi0][zi1];
    let c010 = values[xi0][yi1][zi0];
    let c110 = values[xi1][yi1][zi0];
    let c011 = values[xi0][yi1][zi1];
    let c111 = values[xi1][yi1][zi1];

    let c00 = c000 * (1.0 - xd) + c100 * xd;
    let c01 = c001 * (1.0 - xd) + c101 * xd;
    let c10 = c010 * (1.0 - xd) + c110 * xd;
    let c11 = c011 * (1.0 - xd) + c111 * xd;

    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    let c = c0 * (1.0 - yd) + c1 * zd;

    Ok(c)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub enum Extrapolate {
    /// allow extrapolation
    Yes,
    /// don't allow extrapolation but return result from nearest x-data point
    #[default]
    No,
    /// return an error on attempted extrapolation
    Error,
}

impl SerdeAPI for Extrapolate {}
impl Init for Extrapolate {}

/// interpolation algorithm from <http://www.cplusplus.com/forum/general/216928/>
/// # Arguments:
/// x : value at which to interpolate
pub fn interp1d(
    x: &f64,
    x_data: &[f64],
    y_data: &[f64],
    extrapolate: Extrapolate,
) -> anyhow::Result<f64> {
    let y_first = y_data
        .first()
        .with_context(|| anyhow!("Unable to extract first element"))?;
    if y_data.iter().all(|y| y == y_first) {
        // return first if all data is equal to first
        Ok(*y_first)
    } else {
        let x_mean = x_data.iter().sum::<f64>() / x_data.len() as f64;
        if x_data.iter().all(|&x| x == x_mean) {
            bail!("Cannot interpolate as all values are equal");
        }
        let size = x_data.len();

        let mut i = 0;
        if x >= &x_data[size - 2] {
            i = size - 2;
        } else {
            while i < x_data.len() - 2 && x > &x_data[i + 1] {
                i += 1;
            }
        }
        let xl = &x_data[i];
        let mut yl = &y_data[i];
        let xr = &x_data[i + 1];
        let mut yr = &y_data[i + 1];
        match extrapolate {
            Extrapolate::No => {
                if x < xl {
                    yr = yl;
                }
                if x > xr {
                    yl = yr;
                }
            }
            Extrapolate::Error => {
                if x < xl || x > xr {
                    bail!(
                        "{}\nAttempted extrapolation\n`x_data` first and last: ({}, {})\n`x` input: {}",
                        format_dbg!(),
                        xl,
                        xr,
                        x
                    );
                }
            }
            _ => {}
        }
        let dydx = (yr - yl) / (xr - xl);
        Ok(yl + dydx * (x - xl))
    }
}

/// Returns absolute value of `x_val`
pub fn abs_checked_x_val(x_val: f64, x_data: &[f64]) -> anyhow::Result<f64> {
    if *x_data
        .first()
        .with_context(|| anyhow!("{}\nExpected `first` to return `Some`.", format_dbg!()))?
        == 0.
    {
        Ok(x_val.abs())
    } else {
        Ok(x_val)
    }
}

/// Returns true if `val1` and `val2` are within a relative/absolute `epsilon` of each other,
/// depending on magnitude.
pub fn almost_eq(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(1e-8);
    ((val2 - val1) / (val1 + val2)).abs() < epsilon || (val2 - val1).abs() < epsilon
}

pub fn almost_gt(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(1e-8);
    val1 > val2 * (1.0 + epsilon)
}

pub fn almost_lt(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(1e-8);
    val1 < val2 * (1.0 - epsilon)
}

/// Returns true if `val1` is greater than or equal to `val2` with some error margin, `epsilon`
pub fn almost_ge(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(1e-8);
    val1 > val2 * (1.0 - epsilon) || val1 > val2 - epsilon
}

/// Returns true if `val1` is less than or equal to `val2` with some error margin, `epsilon`
pub fn almost_le(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(1e-8);
    val1 < val2 * (1.0 + epsilon) || val1 < val2 + epsilon
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

make_uom_cmp_fn!(almost_eq);
make_uom_cmp_fn!(almost_gt);
make_uom_cmp_fn!(almost_lt);
make_uom_cmp_fn!(almost_ge);
make_uom_cmp_fn!(almost_le);

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Pyo3VecBoolWrapper(pub Vec<bool>);
impl SerdeAPI for Pyo3VecBoolWrapper {}
impl Init for Pyo3VecBoolWrapper {}

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3VecWrapper(pub Vec<f64>);
impl SerdeAPI for Pyo3VecWrapper {}
impl Init for Pyo3VecWrapper {}

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3Vec2Wrapper(pub Vec<Vec<f64>>);
impl From<Vec<Vec<f64>>> for Pyo3Vec2Wrapper {
    fn from(v: Vec<Vec<f64>>) -> Self {
        Pyo3Vec2Wrapper::new(v)
    }
}
impl SerdeAPI for Pyo3Vec2Wrapper {}
impl Init for Pyo3Vec2Wrapper {}

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3Vec3Wrapper(pub Vec<Vec<Vec<f64>>>);
impl From<Vec<Vec<Vec<f64>>>> for Pyo3Vec3Wrapper {
    fn from(v: Vec<Vec<Vec<f64>>>) -> Self {
        Pyo3Vec3Wrapper::new(v)
    }
}
impl SerdeAPI for Pyo3Vec3Wrapper {}
impl Init for Pyo3Vec3Wrapper {}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub enum Efficiency {
    /// Constant efficiency
    Constant(si::Ratio),
    /// *N*-dimensional efficiency to be used with [multilinear]
    NDim {
        /// A grid containing the coordinates for each dimension, i.e. `[[0.0, 1.0], [-0.5, 1.5]]`
        /// indicates x<sub>0</sub> = 0.0, x<sub>1</sub> = 1.0, y<sub>0</sub> = -0.5, y<sub>1</sub> = 1.5
        grid: Vec<Vec<f64>>,
        /// An *N*-dimensional [`ndarray::ArrayD`] containing the values at given grid coordinates
        values: ArrayD<f64>,
        // NOTE: maybe should figure out a way to provide labels for x-data, y-data, ...
    },
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
/// This function returns an [InterpolationError] if any of the validation checks from [`validate_inputs`] fail,
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
/// ## 2D Example with non-uniform dimension sizes
/// ```rust
/// use ndarray::prelude::*;
/// use fastsim_core::utils::multilinear;
///
/// let grid = [
///     vec![0.0, 1.0, 2.0], // x0, x1, x2
///     vec![0.0, 1.0], // y0, y1
/// ];
/// let values = array![
///     [0.0, 2.0], // f(x0, y0), f(x0, y1)
///     [2.0, 4.0], // f(x1, y0), f(x1, y1)
///     [5.0, 0.0], // f(x2, y0), f(x2, y1)
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

pub(crate) enum InterpRange {
    ZeroThroughOne,
    NegativeOneThroughOne,
    Either,
}

/// Ensures that passed data is between 0 and 1 and monotonically increasing.  
/// # Arguments:
/// - `data`: data used for interpolating efficiency from fraction of peak power
/// - `interp_range`: allowed range
pub(crate) fn check_interp_frac_data(
    data: &[f64],
    interp_range: InterpRange,
) -> anyhow::Result<InterpRange> {
    check_monotonicity(data).with_context(|| anyhow!(format_dbg!()))?;
    let min = data.first().with_context(|| {
        anyhow!(
            "{}\nProblem extracting first element of `data`",
            format_dbg!()
        )
    })?;
    let max = data.last().with_context(|| {
        anyhow!(
            "{}\nProblem extracting first element of `data`",
            format_dbg!()
        )
    })?;
    match interp_range {
        InterpRange::ZeroThroughOne => {
            ensure!(
                *min == 0. && *max == 1.,
                "data min ({}) and max ({}) must be zero and one, respectively.",
                min,
                max
            );
        }
        InterpRange::NegativeOneThroughOne => {
            ensure!(
                *min == -1. && *max == 1.,
                "data min ({}) and max ({}) must be zero and one, respectively.",
                min,
                max
            );
        }
        InterpRange::Either => {
            ensure!(
                (*min == -1. || *min == 0.) && *max == 1.,
                "data min ({}) and max ({}) must be zero or negative one and one, respectively.",
                min,
                max
            );
        }
    }
    if *min == 0. && *max == 1. {
        Ok(InterpRange::ZeroThroughOne)
    } else {
        Ok(InterpRange::NegativeOneThroughOne)
    }
}

/// Verifies that passed `data` is monotonically increasing.
pub fn check_monotonicity(data: &[f64]) -> anyhow::Result<()> {
    ensure!(
        data.windows(2).all(|w| w[0] < w[1]),
        format_dbg!("{}\n`data` must be monotonically increasing")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp3d() {
        let point = [0.5, 0.5, 0.5];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 0.5),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    #[test]
    fn test_interp3d_offset() {
        let point = [0.75, 0.25, 0.5];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 0.75),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    #[test]
    fn test_interp3d_exact_value_lower() {
        let point = [0.0, 0.0, 0.0];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 0.0),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    #[test]
    fn test_interp3d_below_value_lower() {
        let point = [-1.0, -1.0, -1.0];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 0.0),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    #[test]
    fn test_interp3d_above_value_upper() {
        let point = [2.0, 2.0, 2.0];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 1.0),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    #[test]
    fn test_interp3d_exact_value_upper() {
        let point = [1.0, 1.0, 1.0];
        let grid = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let values = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        match interp3d(&point, &grid, &values) {
            Ok(i) => assert!(i == 1.0),
            Err(e) => panic!("test failed with: {e}"),
        };
    }

    // interp1d
    #[test]
    fn test_interp1d_above_value_upper() {
        assert_eq!(
            interp1d(&2.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::Yes).unwrap(),
            2.0
        );
        assert_eq!(
            interp1d(&2.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::No).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_interp1d_exact_value_upper() {
        assert_eq!(
            interp1d(&1.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::Yes).unwrap(),
            1.0
        );
        assert_eq!(
            interp1d(&1.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::No).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_interp1d_exact_value_lower() {
        assert_eq!(
            interp1d(&0.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::Yes).unwrap(),
            0.0
        );
        assert_eq!(
            interp1d(&0.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::No).unwrap(),
            0.0
        );
    }
    #[test]
    fn test_interp1d_below_value_lower() {
        assert_eq!(
            interp1d(&-1.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::Yes).unwrap(),
            -1.0
        );
        assert_eq!(
            interp1d(&-1.0, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::No).unwrap(),
            0.0
        );
    }
    #[test]
    fn test_interp1d_inside_range() {
        assert_eq!(
            interp1d(&0.5, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::Yes).unwrap(),
            0.5
        );
        assert_eq!(
            interp1d(&0.5, &[0.0, 1.0], &[0.0, 1.0], Extrapolate::No).unwrap(),
            0.5
        );
    }

    #[test]
    fn test_interp1d_with_duplicate_y_data() {
        assert_eq!(
            interp1d(&0.5, &[0.0, 1.0], &[1.0, 1.0], Extrapolate::Yes).unwrap(),
            1.0
        );
        assert_eq!(
            interp1d(&0.5, &[0.0, 1.0], &[1.0, 1.0], Extrapolate::No).unwrap(),
            1.0
        );
    }

    #[test]
    fn test_interp1d_with_duplicate_x_data() {
        assert!(interp1d(&0.5, &[0.0, 0.0], &[0.0, 1.0], Extrapolate::Yes).is_err());
    }

    #[test]
    fn test_linspace() {
        assert_eq!(Vec::linspace(0.0, 1.0, 3), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_almost_gt_zero() {
        assert!(almost_gt(1e-9, 0.0, None));
        assert!(!almost_gt(0.0, 1e-9, None));
        assert!(almost_gt(1e-7, 0.0, None));
        assert!(!almost_gt(0.0, 1e-7, None));
    }

    #[test]
    fn test_almost_ge_zero() {
        assert!(almost_ge(1e-9, 0.0, None));
        assert!(almost_ge(0.0, 1e-9, None));
        assert!(almost_ge(1e-7, 0.0, None));
        assert!(!almost_ge(0.0, 1e-7, None));
    }

    #[test]
    fn test_almost_eq_zero() {
        assert!(almost_eq(0.0, 1e-9, None));
        assert!(almost_eq(1e-9, 0.0, None));
        assert!(!almost_eq(0.0, 1e-7, None));
        assert!(!almost_eq(1e-7, 0.0, None));
    }

    #[test]
    fn test_almost_le_zero() {
        assert!(almost_le(1e-9, 0.0, None));
        assert!(almost_le(0.0, 1e-9, None));
        assert!(!almost_le(1e-7, 0.0, None));
        assert!(almost_le(0.0, 1e-7, None));
    }

    #[test]
    fn test_almost_lt_zero() {
        assert!(!almost_lt(1e-9, 0.0, None));
        assert!(almost_lt(0.0, 1e-9, None));
        assert!(!almost_lt(1e-7, 0.0, None));
        assert!(almost_lt(0.0, 1e-7, None));
    }

    #[test]
    fn test_almost_gt_large() {
        assert!(!almost_gt(1e9 * (1.0 + 1e-9), 1e9, None));
        assert!(!almost_gt(1e9, 1e9 * (1.0 + 1e-9), None));
        assert!(almost_gt(1e9 * (1.0 + 1e-7), 1e9, None));
        assert!(!almost_gt(1e9, 1e9 * (1.0 + 1e-7), None));
    }

    #[test]
    fn test_almost_ge_large() {
        assert!(almost_ge(1e9 * (1.0 + 1e-9), 1e9, None));
        assert!(almost_ge(1e9, 1e9 * (1.0 + 1e-9), None));
        assert!(almost_ge(1e9 * (1.0 + 1e-7), 1e9, None));
        assert!(!almost_ge(1e9, 1e9 * (1.0 + 1e-7), None));
    }

    #[test]
    fn test_almost_eq_large() {
        assert!(almost_eq(1e9 * (1.0 + 1e-9), 1e9, None));
        assert!(almost_eq(1e9, 1e9 * (1.0 + 1e-9), None));
        assert!(!almost_eq(1e9 * (1.0 + 1e-7), 1e9, None));
        assert!(!almost_eq(1e9, 1e9 * (1.0 + 1e-7), None));
    }

    #[test]
    fn test_almost_le_large() {
        assert!(almost_le(1e9 * (1.0 + 1e-9), 1e9, None));
        assert!(almost_le(1e9, 1e9 * (1.0 + 1e-9), None));
        assert!(!almost_le(1e9 * (1.0 + 1e-7), 1e9, None));
        assert!(almost_le(1e9, 1e9 * (1.0 + 1e-7), None));
    }

    #[test]
    fn test_almost_lt_large() {
        assert!(!almost_lt(1e9 * (1.0 + 1e-9), 1e9, None));
        assert!(!almost_lt(1e9, 1e9 * (1.0 + 1e-9), None));
        assert!(!almost_lt(1e9 * (1.0 + 1e-7), 1e9, None));
        assert!(almost_lt(1e9, 1e9 * (1.0 + 1e-7), None));
    }
}
