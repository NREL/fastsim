//! Module containing structs with parallel and serial iteration methods for batch
//! running of `sim_drive` method
// crate local
use super::RustSimDrive;
use crate::imports::*;
use crate::proc_macros::add_pyo3_api;
#[cfg(feature = "pyo3")]
use crate::pyo3imports::*;
use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[add_pyo3_api(
    #[pyo3(name = "sim_drive")]
    /// Calls `sim_drive` method for each simdrive instance in vec.
    /// # Arguments:
    /// * parallelize: whether to parallelize `sim_drive` calls, defaults to `true`
    fn sim_drive_py(&mut self, parallelize: Option<bool>) -> anyhow::Result<()> {
        self.sim_drive(parallelize)
    }

    #[pyo3(name = "push")]
    fn push_py(&mut self, sd: RustSimDrive) {
        self.push(sd)
    }

    #[pyo3(name = "pop")]
    fn pop_py(&mut self) -> Option<RustSimDrive> {
        self.pop()
    }

    #[pyo3(name = "remove")]
    fn remove_py(&mut self, idx: usize) {
        self.remove(idx);
    }

    #[pyo3(name = "insert")]
    fn insert_py(&mut self, idx: usize, sd: RustSimDrive) {
        self.insert(idx, sd);
    }
)]
pub struct SimDriveVec(pub Vec<RustSimDrive>);

impl SimDriveVec {
    /// Calls `sim_drive` method for each simdrive instance in vec.
    /// # Arguments:
    /// * parallelize: whether to parallelize `sim_drive` calls
    pub fn sim_drive(&mut self, parallelize: Option<bool>) -> anyhow::Result<()> {
        let parallelize = parallelize.unwrap_or(true);
        if parallelize {
            self.0.par_iter_mut().enumerate().try_for_each(|(i, sd)| {
                sd.sim_drive(None, None)
                    .with_context(|| format!("simdrive idx: {}", i))
            })?;
        } else {
            self.0.iter_mut().enumerate().try_for_each(|(i, sd)| {
                sd.sim_drive(None, None)
                    .with_context(|| format!("simdrive idx: {}", i))
            })?;
        }
        Ok(())
    }

    pub fn push(&mut self, sd: RustSimDrive) {
        self.0.push(sd);
    }

    pub fn pop(&mut self) -> Option<RustSimDrive> {
        self.0.pop()
    }

    pub fn remove(&mut self, idx: usize) {
        self.0.remove(idx);
    }

    pub fn insert(&mut self, idx: usize, sd: RustSimDrive) {
        self.0.insert(idx, sd);
    }
}

impl SerdeAPI for SimDriveVec {}
