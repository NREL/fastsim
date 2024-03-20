use crate::imports::*;
use paste::paste;
use lazy_static::lazy_static;
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
/// TODO: this could be generalized to compute a linear interpolation in N dimensions  
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

    let (xi0, xi1) = find_interp_indices(&x, x_points)?;
    let (yi0, yi1) = find_interp_indices(&y, y_points)?;
    let (zi0, zi1) = find_interp_indices(&z, z_points)?;

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

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, SerdeAPI)]
pub enum Extrapolate {
    /// allow extrapolation
    Yes,
    /// don't allow extropalaiton but return result from nearest x-data point
    #[default]
    No,
    /// return an error on extrapolation
    Error,
}

/// interpolation algorithm from <http://www.cplusplus.com/forum/general/216928/>  
/// Arguments:
/// x : value at which to interpolate
pub fn interp1d(
    x: &f64,
    x_data: &[f64],
    y_data: &[f64],
    extrapolate: Extrapolate,
) -> anyhow::Result<f64> {
    let y_mean = y_data.iter().sum::<f64>() / y_data.len() as f64;
    if y_data.iter().all(|&y| y == y_mean) {
        // return mean if all data is equal to mean
        Ok(y_mean)
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
            while x > &x_data[i + 1] {
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
                bail!("{}\nAttempted extrapolation", format_dbg!());
            }
            _ => {}
        }
        let dydx = (yr - yl) / (xr - xl);
        Ok(yl + dydx * (x - xl))
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

make_cmp_fns!(almost_eq);
make_cmp_fns!(almost_gt);
make_cmp_fns!(almost_lt);
make_cmp_fns!(almost_ge);
make_cmp_fns!(almost_le);

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, Eq, SerdeAPI)]
pub struct Pyo3VecBoolWrapper(pub Vec<bool>);

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, SerdeAPI)]
pub struct Pyo3VecWrapper(pub Vec<f64>);

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, SerdeAPI)]
pub struct Pyo3Vec2Wrapper(pub Vec<Vec<f64>>);
impl From<Vec<Vec<f64>>> for Pyo3Vec2Wrapper {
    fn from(v: Vec<Vec<f64>>) -> Self {
        Pyo3Vec2Wrapper::new(v)
    }
}

#[pyo3_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, SerdeAPI)]
pub struct Pyo3Vec3Wrapper(pub Vec<Vec<Vec<f64>>>);
impl From<Vec<Vec<Vec<f64>>>> for Pyo3Vec3Wrapper {
    fn from(v: Vec<Vec<Vec<f64>>>) -> Self {
        Pyo3Vec3Wrapper::new(v)
    }
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
