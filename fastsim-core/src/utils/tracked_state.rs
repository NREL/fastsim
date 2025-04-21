use super::*;

pub trait CheckAndResetState {
    /// Ensure [State::Fresh] and reset to [State::Stale]
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()>;
}

impl<T> CheckAndResetState for TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    fn check_and_reset<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.ensure_fresh(loc)?;
        self.mark_stale();
        Ok(())
    }
}

/// Enum for tracking mutation
#[derive(Clone, Default, Debug, PartialEq, IsVariant, derive_more::From, TryInto)]
pub enum State {
    /// Updated in this time step
    #[default]
    Fresh,
    /// Not yet updated in this time step
    Stale,
}

#[derive(Default, PartialEq, Clone, Debug)]
/// Struct for storing state variable and ensuring one mutation per
/// initialization or reset -- i.e. one mutation per time step
pub struct TrackedState<T>(
    /// Value
    T,
    /// Update status
    State,
);

/// Provides methods to guarantee that states are updated once and only once per time step
impl<T> TrackedState<T>
where
    T: std::fmt::Debug + Clone + PartialEq + Default,
{
    pub fn new(value: T) -> Self {
        Self(value, Default::default())
    }

    pub fn is_fresh(&self) -> bool {
        self.1.is_fresh()
    }

    pub fn is_stale(&self) -> bool {
        self.1.is_stale()
    }

    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    pub fn ensure_fresh<F: Fn() -> String>(&self, loc: F) -> anyhow::Result<()> {
        ensure!(
            self.1.is_fresh(),
            format!(
                "{}\nState variable has not been updated. This is a bug in `fastsim-core`",
                loc()
            )
        );
        Ok(())
    }

    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    pub fn ensure_stale<F: Fn() -> String>(&self, loc: F) -> anyhow::Result<()> {
        ensure!(
            self.1.is_stale(),
            format!(
                "{}\nState variable has already been updated. This is a bug in `fastsim-core`",
                loc()
            )
        );
        Ok(())
    }

    /// Reset the tracked state to [State::Stale] for the next update after
    /// verifying that is has been updated
    pub fn mark_stale(&mut self) {
        self.1 = State::Stale;
    }

    // Note that `anyhow::Error` is fine here because this should result only in
    // logic errors and not runtime errors for end users
    /// Update the value of the tracked state after verifying that it has not
    /// already been updated
    /// # Arguments
    /// - `value`: new value
    /// - `loc`: closure that returns file and line number where called
    pub fn update<F: Fn() -> String>(&mut self, value: T, loc: F) -> anyhow::Result<()> {
        ensure!(
            self.1.is_stale(),
            format!(
                "{}\nState variable has not been reset. This is a bug in `fastsim-core`.",
                loc()
            )
        );
        self.0 = value;
        self.1 = State::Fresh;
        Ok(())
    }

    /// Verify that state is [State::Stale] and mark state as [State::Fresh]
    /// without updating
    /// # Arguments
    /// - `loc`: closure that returns file and line number where called
    pub fn mark_fresh<F: Fn() -> String>(&mut self, loc: F) -> anyhow::Result<()> {
        self.ensure_stale(loc)?;
        self.1 = State::Fresh;
        Ok(())
    }

    /// Check that value has been updated and then return as a result
    /// # Arguments
    /// - `loc`: call site location filename and line number
    pub fn get_fresh<F: Fn() -> String>(&self, loc: F) -> anyhow::Result<&T> {
        self.ensure_fresh(loc)?;
        Ok(&self.0)
    }

    /// Check that value has **not** been updated and then return as a result
    /// # Arguments
    /// - `loc`: call site location filename and line number
    pub fn get_stale<F: Fn() -> String>(&self, loc: F) -> anyhow::Result<&T> {
        self.ensure_stale(loc)?;
        Ok(&self.0)
    }
}

/// State methods that allow for `+=`
impl<T: std::fmt::Debug + Clone + PartialEq + Default + std::ops::AddAssign> TrackedState<T> {
    // Note that `anyhow::Error` is fine here because this should result only in
    // logic errors and not runtime errors for end users
    /// Update the value of the tracked state
    /// # Arguments
    /// - `value`: new value
    /// - `loc`: closure that returns file and line number where called
    pub fn increment<F: Fn() -> String>(&mut self, value: T, loc: F) -> anyhow::Result<()> {
        self.ensure_stale(loc)?;
        self.0 += value;
        self.1 = State::Fresh;
        Ok(())
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

        Ok(Self(value, Default::default()))
    }
}

#[cfg(test)]
mod test_tracked_state {
    // use super::*;
    // #[test]
    // #[should_panic]
    // fn test_that_update_can_happen_only_once() {
    //     let mut pwr = TrackedState::<si::Power>::default();
    //     let mut energy = TrackedState::<si::Energy>::default();
    //     let mut dt = TrackedState::<si::Time>::default();

    //     pwr.update(si::Power::new::<si::watt>(1.0), || format_dbg!())
    //         .unwrap();
    //     dt.update(si::Time::new::<si::second>(1.0), || format_dbg!())
    //         .unwrap();
    //     energy
    //         .increment(
    //             *pwr.get_fresh(|| format_dbg!()).unwrap() * *dt.get_fresh(|| format_dbg!()).unwrap(),
    //             format_dbg!(),
    //         )
    //         .unwrap();

    //     // This should trigger a panic
    //     pwr.update(si::Power::new::<si::watt>(2.0), || format_dbg!())
    //         .unwrap();
    // }

    // #[test]
    // fn test_that_reset_and_check_work() {
    //     let mut pwr = TrackedState::<si::Power>::default();
    //     let mut energy = TrackedState::<si::Energy>::default();
    //     let mut dt = TrackedState::<si::Time>::default();

    //     pwr.is_fresh_and_reset(format_dbg!()).unwrap();
    //     dt.is_fresh_and_reset(format_dbg!()).unwrap();
    //     energy.is_fresh_and_reset(format_dbg!()).unwrap();

    //     pwr.update(si::Power::new::<si::watt>(1.0), || format_dbg!())
    //         .unwrap();
    //     dt.update(si::Time::new::<si::second>(1.0), || format_dbg!())
    //         .unwrap();
    //     energy
    //         .increment(
    //             *pwr.get_fresh(|| format_dbg!()).unwrap() * *dt.get_fresh(|| format_dbg!()).unwrap(),
    //             format_dbg!(),
    //         )
    //         .unwrap();

    //     pwr.is_fresh_and_reset(format_dbg!()).unwrap();
    //     dt.is_fresh_and_reset(format_dbg!()).unwrap();
    //     energy.is_fresh_and_reset(format_dbg!()).unwrap();

    //     pwr.update(si::Power::new::<si::watt>(1.0), || format_dbg!())
    //         .unwrap();
    //     dt.update(si::Time::new::<si::second>(1.0), || format_dbg!())
    //         .unwrap();
    //     energy
    //         .increment(
    //             *pwr.get_fresh(|| format_dbg!()).unwrap() * *dt.get_fresh(|| format_dbg!()).unwrap(),
    //             format_dbg!(),
    //         )
    //         .unwrap();

    //     pwr.is_fresh_and_reset(format_dbg!()).unwrap();
    //     dt.is_fresh_and_reset(format_dbg!()).unwrap();
    //     energy.is_fresh_and_reset(format_dbg!()).unwrap();
    // }
}
