use crate::imports::*;
use itertools::Itertools;
use lazy_static::lazy_static;
use paste::paste;
use regex::Regex;

pub mod interp;

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
