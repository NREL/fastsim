//! Module for crate-local imports to reduce boilerplate in submodules

#![allow(unused_imports)]

#[cfg(feature = "pyo3")]
pub(crate) use crate::pyo3::*;

pub(crate) use crate::si;
pub(crate) use crate::traits::*;
pub(crate) use crate::uc;
pub(crate) use crate::utils;
pub(crate) use crate::utils::{
    almost_eq, check_interp_frac_data, check_monotonicity, interp1d, interp3d, is_sorted,
    Extrapolate, InterpRange, DIRECT_SET_ERR,
};
pub(crate) use crate::utils::{Pyo3Vec2Wrapper, Pyo3Vec3Wrapper, Pyo3VecWrapper};
pub(crate) use fastsim_proc_macros::{pyo3_api, HistoryMethods, HistoryVec, SetCumulative};

pub(crate) use anyhow::{anyhow, bail, ensure, Context};
pub(crate) use bincode::{deserialize, serialize};
pub(crate) use duplicate::duplicate_item;
pub(crate) use easy_ext::ext;
pub(crate) use ndarray::prelude::*;
pub(crate) use ndarray::Slice;
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use std::cmp::{self, Ordering};
pub(crate) use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
pub(crate) use std::error::Error;
pub(crate) use std::ffi::OsStr;
pub(crate) use std::fmt;
pub(crate) use std::fs::File;
pub(crate) use std::num::{NonZeroU16, NonZeroUsize};
pub(crate) use std::ops::{Deref, DerefMut, IndexMut, Sub};
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use uom::typenum;
pub(crate) use uom::ConstZero;
