use super::*;

#[derive(PartialEq, Clone, Debug)]
/// Struct for storing state variable and ensuring one mutation per
/// initialization or reset -- i.e. one mutation per time step
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
/// initialization or reset -- i.e. one mutation per time step
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
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.1 = Some(self.get(format_dbg!())?.clone());
        self.0 = None;
        Ok(())
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

    /// Return previous value or default
    pub fn get_prev_or_default(&self) -> T {
        self.1.clone().unwrap_or_default()
    }

    /// Return previous value or current value if previous value has not yet been populated
    /// # Arguments
    /// - `loc`: call site location filename and line number
    pub fn get_prev_or_curr(&self, loc: String) -> anyhow::Result<&T> {
        match &self.1 {
            Some(prev) => Ok(&prev),
            None => self.get(loc),
        }
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

#[derive(PartialEq, Clone, Debug)]
pub struct TrackedStateVec<T: std::fmt::Debug + Clone + PartialEq>(pub Vec<T>);

impl<T> TrackedStateVec<T>
where
    T: std::fmt::Debug + Clone + PartialEq,
{
    // This is just here to fit the `TrackedState` pattern
    pub fn check_and_reset(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    pub fn push(&mut self, element: T) {
        self.0.push(element);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<T> Default for TrackedStateVec<T>
where
    T: std::fmt::Debug + Clone + PartialEq,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

// Custom serialization
impl<T> Serialize for TrackedStateVec<T>
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

impl<'de, T> Deserialize<'de> for TrackedStateVec<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Deserialize<'de> + Serialize + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value: Vec<T> = Vec::<T>::deserialize(deserializer)?;

        Ok(Self(value))
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
                    + energy.get_prev_or_default(),
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
