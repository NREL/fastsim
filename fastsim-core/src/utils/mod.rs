use crate::imports::*;
use paste::paste;
use regex::Regex;

pub mod interp;
pub use interp::*;

/// Error message for when user attempts to set value in a nested struct.
pub const DIRECT_SET_ERR: &str = "Setting field value directly not allowed";

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

#[allow(unused)]
/// Helper function to find where a query falls on an axis of discrete values;
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

#[allow(unused)]
/// Helper function to compute the difference between a value and a set of bounds
fn compute_interp_diff(value: &f64, lower: &f64, upper: &f64) -> f64 {
    if lower == upper {
        0.0
    } else {
        (value - lower) / (upper - lower)
    }
}

impl SerdeAPI for Extrapolate {}
impl Init for Extrapolate {}

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

// public to enable exposure in docs
pub const COMP_EPSILON: f64 = 1e-8;

/// Returns true if `val1` and `val2` are within a relative/absolute `epsilon` of each other,
/// depending on magnitude.
pub fn almost_eq(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(COMP_EPSILON);
    ((val2 - val1) / (val1 + val2)).abs() < epsilon || (val2 - val1).abs() < epsilon
}

pub fn almost_gt(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(COMP_EPSILON);
    val1 > val2 * (1.0 + epsilon)
}

pub fn almost_lt(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(COMP_EPSILON);
    val1 < val2 * (1.0 - epsilon)
}

/// Returns true if `val1` is greater than or equal to `val2` with some error margin, `epsilon`
pub fn almost_ge(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(COMP_EPSILON);
    val1 > val2 * (1.0 - epsilon) || val1 > val2 - epsilon
}

/// Returns true if `val1` is less than or equal to `val2` with some error margin, `epsilon`
pub fn almost_le(val1: f64, val2: f64, epsilon: Option<f64>) -> bool {
    let epsilon = epsilon.unwrap_or(COMP_EPSILON);
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

#[fastsim_api]
#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Pyo3VecBoolWrapper(pub Vec<bool>);
impl SerdeAPI for Pyo3VecBoolWrapper {}
impl Init for Pyo3VecBoolWrapper {}

#[fastsim_api]
#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3VecWrapper(pub Vec<f64>);
impl SerdeAPI for Pyo3VecWrapper {}
impl Init for Pyo3VecWrapper {}

#[allow(non_snake_case)]
#[fastsim_api]
#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3Vec2Wrapper(pub Vec<Vec<f64>>);
impl From<Vec<Vec<f64>>> for Pyo3Vec2Wrapper {
    fn from(v: Vec<Vec<f64>>) -> Self {
        Pyo3Vec2Wrapper::new(v)
    }
}
impl SerdeAPI for Pyo3Vec2Wrapper {}
impl Init for Pyo3Vec2Wrapper {}

#[fastsim_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Pyo3Vec3Wrapper(pub Vec<Vec<Vec<f64>>>);
impl From<Vec<Vec<Vec<f64>>>> for Pyo3Vec3Wrapper {
    fn from(v: Vec<Vec<Vec<f64>>>) -> Self {
        Pyo3Vec3Wrapper::new(v)
    }
}
impl SerdeAPI for Pyo3Vec3Wrapper {}
impl Init for Pyo3Vec3Wrapper {}

#[derive(IsVariant, derive_more::From, TryInto)]
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

#[derive(PartialEq, Clone, Debug)]
/// Struct for storing state variable and ensuring one mutation per
/// initialization or reset
pub struct TrackedState<T: std::fmt::Debug + Clone + PartialEq + Default>(Option<T>);

impl<T> TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    // Not that `anyhow::Error` is fine here because this should result only in
    // logic errors and not runtime errors for end users
    /// Update the value of the tracked state
    /// # Arguments
    /// - `value`: new value
    /// - `loc`: file and line number where called
    pub fn update(&mut self, value: T, loc: String) -> anyhow::Result<()> {
        ensure!(
            self.0.is_none(),
            format!("{}\nState variable has not been reset", loc)
        );
        self.0 = Some(value);
        Ok(())
    }

    /// Reset the tracked state for the next update
    pub fn reset(&mut self) {
        self.0 = None;
    }

    /// Verify that the state has been updated
    pub fn check(&self) -> anyhow::Result<()> {
        ensure!(self.0.is_some(), "State variable was not updated!");
        Ok(())
    }

    /// Run [Self::check] and [Self::reset]
    pub fn check_and_reset(&mut self) -> anyhow::Result<()> {
        self.check()?;
        self.reset();
        Ok(())
    }

    /// Check that value has been updated and then return as a result
    /// # Arguments
    /// - `loc`: call site location filename and line number
    pub fn get(&self, loc: String) -> anyhow::Result<&T> {
        self.0
            .as_ref()
            .ok_or(anyhow!("{}\nState variable was not updated!", loc))
    }

    pub fn new(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T> Default for TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    fn default() -> Self {
        Self(Some(Default::default()))
    }
}

// Custom serialization
impl<T> Serialize for TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + for<'de> Deserialize<'de> + Serialize + Default,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Deserialize<'de> + Serialize + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: T = T::deserialize(deserializer)?;

        Ok(Self(Some(value)))
    }
}

#[derive(PartialEq, Clone, Debug)]
/// Struct for storing state variable and ensuring one mutation per
/// initialization or reset
pub struct TrackedStateWithMemory<T: std::fmt::Debug + Clone + PartialEq + Default>(
    Option<T>,
    Option<T>,
);

impl<T> TrackedStateWithMemory<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    // Not that `anyhow::Error` is fine here because this should result only in
    // logic errors and not runtime errors for end users
    /// Update the value of the tracked state
    /// # Arguments
    /// - `value`: new value
    /// - `loc`: file and line number where called
    pub fn update(&mut self, value: T, loc: String) -> anyhow::Result<()> {
        ensure!(
            self.0.is_none(),
            format!("{}\nState variable has not been reset", loc)
        );
        self.0 = Some(value);
        Ok(())
    }

    /// Reset the tracked state for the next update
    pub fn reset(&mut self) {
        self.1 = self.0.clone();
        self.0 = None;
    }

    /// Verify that the state has been updated
    fn check(&self) -> anyhow::Result<()> {
        ensure!(self.0.is_some(), "State variable was not updated!");
        Ok(())
    }

    /// Run [Self::check] and [Self::reset]
    pub fn check_and_reset(&mut self) -> anyhow::Result<()> {
        self.check()?;
        self.reset();
        Ok(())
    }

    /// Check that value has been updated and then return as a result
    /// # Arguments
    /// - `loc`: call site location filename and line number
    pub fn get(&self, loc: String) -> anyhow::Result<&T> {
        self.0
            .as_ref()
            .ok_or(anyhow!("{}\nState variable was not updated!", loc))
    }

    /// Return previous value
    pub fn get_prev(&self) -> Option<T> {
        self.1.clone()
    }

    pub fn new(value: T) -> Self {
        Self(Some(value), None)
    }
}

impl<T> Default for TrackedStateWithMemory<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    fn default() -> Self {
        Self(Some(Default::default()), None)
    }
}

// Custom serialization
impl<T> Serialize for TrackedStateWithMemory<T>
where
    T: std::fmt::Debug + Clone + PartialEq + for<'de> Deserialize<'de> + Serialize + Default,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for TrackedStateWithMemory<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Deserialize<'de> + Serialize + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: T = T::deserialize(deserializer)?;

        Ok(Self(Some(value), None))
    }
}

#[cfg(test)]
mod test_tracked_state {
    use super::*;
    #[test]
    #[should_panic]
    fn test_that_update_can_happen_only_once() {
        let mut pwr = TrackedState::<si::Power>::default();
        let mut energy = TrackedStateWithMemory::<si::Energy>::default();
        let mut dt = TrackedState::<si::Time>::default();

        pwr.update(si::Power::new::<si::watt>(1.0), format_dbg!())
            .unwrap();
        dt.update(si::Time::new::<si::second>(1.0), format_dbg!())
            .unwrap();
        energy
            .update(
                *pwr.get(format_dbg!()).unwrap() * *dt.get(format_dbg!()).unwrap()
                    + energy.get_prev().unwrap_or_default(),
                format_dbg!(),
            )
            .unwrap();

        pwr.update(si::Power::new::<si::watt>(2.0), format_dbg!())
            .unwrap();
    }

    #[test]
    fn test_that_reset_and_check_work() {
        let mut pwr = TrackedState::<si::Power>::default();
        let mut energy = TrackedState::<si::Energy>::default();
        let mut dt = TrackedState::<si::Time>::default();

        pwr.check_and_reset().unwrap();
        dt.check_and_reset().unwrap();
        energy.check_and_reset().unwrap();

        pwr.update(si::Power::new::<si::watt>(1.0), format_dbg!())
            .unwrap();
        dt.update(si::Time::new::<si::second>(1.0), format_dbg!())
            .unwrap();
        energy
            .update(
                *pwr.get(format_dbg!()).unwrap() * *dt.get(format_dbg!()).unwrap(),
                format_dbg!(),
            )
            .unwrap();

        pwr.check_and_reset().unwrap();
        dt.check_and_reset().unwrap();
        energy.check_and_reset().unwrap();

        pwr.update(si::Power::new::<si::watt>(1.0), format_dbg!())
            .unwrap();
        dt.update(si::Time::new::<si::second>(1.0), format_dbg!())
            .unwrap();
        energy
            .update(
                *pwr.get(format_dbg!()).unwrap() * *dt.get(format_dbg!()).unwrap(),
                format_dbg!(),
            )
            .unwrap();

        pwr.check_and_reset().unwrap();
        dt.check_and_reset().unwrap();
        energy.check_and_reset().unwrap();
    }
}
