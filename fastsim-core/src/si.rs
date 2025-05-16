//! Import uom si system and add unit constants
//! Zero values should be created using standard uom syntax ($Quantity::ZERO) after adding "use crate::imports::*"
//! Non-zero values should be created using standard uom syntax ($Quantity::new::<$unit>($value)) or multiplication syntax ($value * $UNIT_CONSTANT)

use uom::si;

pub use si::area::square_meter;
pub use si::available_energy::{joule_per_kilogram, kilojoule_per_kilogram};
pub use si::dynamic_viscosity::pascal_second;
pub use si::energy::{joule, kilowatt_hour, watt_hour};
pub use si::f64::{
    Acceleration, Angle, Area, AvailableEnergy as SpecificEnergy, Curvature, DynamicViscosity,
    Energy, Force, Frequency, HeatCapacity, HeatTransfer as HeatTransferCoeff, InverseVelocity,
    Length, Mass, MassDensity, MomentOfInertia, Power, PowerRate, Pressure, Ratio,
    SpecificHeatCapacity, SpecificPower, TemperatureInterval, ThermalConductance,
    ThermalConductivity, ThermodynamicTemperature as Temperature, Time, Velocity, Volume,
};
pub use si::force::{newton, pound_force};
pub use si::heat_capacity::{joule_per_degree_celsius, joule_per_kelvin};
pub use si::heat_transfer::{watt_per_square_meter_degree_celsius, watt_per_square_meter_kelvin};
pub use si::length::{foot, kilometer, meter};
pub use si::mass::{kilogram, megagram};
pub use si::mass_density::kilogram_per_cubic_meter;
pub use si::moment_of_inertia::kilogram_square_meter;
pub use si::power::{kilowatt, megawatt, watt};
pub use si::power_rate::watt_per_second;
pub use si::pressure::kilopascal;
pub use si::ratio::ratio;
pub use si::specific_heat_capacity::{
    joule_per_kilogram_degree_celsius, joule_per_kilogram_kelvin,
};
pub use si::specific_power::{kilowatt_per_kilogram, watt_per_kilogram};
pub use si::temperature_interval::kelvin;
pub use si::thermal_conductance::watt_per_kelvin;
pub use si::thermal_conductivity::{watt_per_meter_degree_celsius, watt_per_meter_kelvin};
pub use si::thermodynamic_temperature::{degree_celsius, kelvin as kelvin_abs};
pub use si::time::{hour, second};
pub use si::velocity::{meter_per_second, mile_per_hour};
pub use si::volume::cubic_meter;
