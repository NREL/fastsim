//! Provides conversion functions for data structures between major FASTSim versions.

use super::vehicle::*;
use super::vehicle::vehicle_model::*;
use super::vehicle::conv::*;
use super::vehicle::hev::*;
use super::vehicle::bev::*;

pub mod fastsim2;

pub use fastsim2::*;