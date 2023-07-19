use crate::imports::*;

/// A range of values stored as a tuple struct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValRange<T>(
    /// Lower value
    T,
    /// Upper value
    T,
);

impl<T> SerdeAPI for ValRange<T> where T: Serialize + for<'a> Deserialize<'a> {}

/// Some common functionality that should exist when working with a value in
/// a [ValRange] or when comparing to another [ValRange].
trait ValRangeCheck<T> {
    /// Is the value contained in the [ValRange]?
    fn in_range(&self, val: &T) -> bool;

    /// Create a new value constrained by the [ValRange]
    fn constrain(&self, val: &T) -> T;

    /// Update the [ValRange] to include a value.
    fn min_max(&mut self, val: &T);
}

/// Implementation of the [ValRangeCheck] common capabilities for [ValRange]es
/// over most number types.
impl<T> ValRangeCheck<T> for ValRange<T>
where
    T: Ord + PartialEq + Clone,
{
    fn in_range(&self, val: &T) -> bool {
        (&self.0..=&self.1).contains(&val)
    }

    fn constrain(&self, val: &T) -> T {
        cmp::min(self.1.clone(), cmp::max(self.0.clone(), val.clone()))
    }

    fn min_max(&mut self, val: &T) {
        self.0 = cmp::min(self.0.clone(), val.clone());
        self.1 = cmp::max(self.1.clone(), val.clone());
    }
}

impl<T> ValRangeCheck<ValRange<T>> for ValRange<T>
where
    T: Ord + PartialEq + Clone,
{
    /// Is the argument `range` contained within this range?
    fn in_range(&self, range: &ValRange<T>) -> bool {
        self.0 <= range.0 && range.1 <= self.1
    }

    /// Create a new [ValRange] based on `range` but constrained by this
    /// [ValRange].
    fn constrain(&self, range: &ValRange<T>) -> ValRange<T> {
        Self(
            cmp::min(self.1.clone(), cmp::max(self.0.clone(), range.0.clone())),
            cmp::min(self.1.clone(), cmp::max(self.0.clone(), range.1.clone())),
        )
    }

    /// Update this [ValRange] to include the other [ValRange].
    fn min_max(&mut self, other: &ValRange<T>) {
        self.0 = cmp::min(self.0.clone(), other.0.clone());
        self.1 = cmp::max(self.1.clone(), other.1.clone());
    }
}

impl<T> ValRange<T>
where
    T: PartialOrd + PartialEq + Sub + Clone,
{
    pub fn is_single_val(&self) -> bool {
        self.0 == self.1
    }

    pub fn diff(&self) -> <T as Sub>::Output {
        self.1.clone() - self.0.clone()
    }

    pub fn validate(&self) {
        assert!(self.0 < self.1)
    }
}
