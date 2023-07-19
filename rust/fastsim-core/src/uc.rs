//! Module providing unit constants (e.g. 1 kg) for an assortment of
//! dimensional quantities.  

use super::si::*;

use uom::lib::marker::PhantomData;
use uom::si::Quantity;

/// Invoking `unit_const!(R, Ratio, 1.0);` yields:
/// ```ignore
/// pub const R: Ratio = Quantity {
///    dimension: PhantomData,
///    units: PhantomData,
///    value: 1.0,
/// }
/// ```
macro_rules! unit_const {
    ($(#[$docs:meta])* $name:ident, $T:ty, $value:expr) => {
        $(#[$docs])*
        /// defines uom unit const
        pub const $name: $T = Quantity {
            dimension: PhantomData,
            units: PhantomData,
            value: $value,
        };
    };
}

unit_const!(R, Ratio, 1.0);
unit_const!(RAD, Angle, 1.0);
unit_const!(DEG, Angle, 1.745_329_251_994_329_5_E-2);
unit_const!(REV, Angle, 6.283_185_307_179_586_E0);
unit_const!(RADPM, Curvature, 1.0);

unit_const!(KG, Mass, 1.0);
unit_const!(TON, Mass, 9.071_847_E2);
unit_const!(LB, Mass, 4.535_924_E-1);

unit_const!(M, Length, 1.0);
unit_const!(FT, Length, 3.048_E-1);
unit_const!(MI, Length, 1.609_344_E3);
unit_const!(M2, Area, 1.0);
unit_const!(FT2, Area, 9.290_304_E-2);
unit_const!(M3, Volume, 1.0);

unit_const!(S, Time, 1.0);
unit_const!(MIN, Time, 60.0);
unit_const!(TIME_NAN, Time, f64::NAN);
unit_const!(HZ, Frequency, 1.0);

unit_const!(N, Force, 1.0);
unit_const!(LBF, Force, 4.448_222_E0);

unit_const!(W, Power, 1.0);
unit_const!(KW, Power, 1.0E3);
unit_const!(MW, Power, 1.0E6);
unit_const!(J, Energy, 1.0);

unit_const!(KGPM3, MassDensity, 1.0);

unit_const!(MPS, Velocity, 1.0);
unit_const!(MPH, Velocity, 4.470_4_E-1);
unit_const!(MPS2, Acceleration, 1.0);
unit_const!(SPM, InverseVelocity, 1.0);
unit_const!(SPEED_DIFF_JOIN, Velocity, 4.470_4_E-2);

unit_const!(WPS, PowerRate, 1.0);

unit_const!(
    /// Acceleration due to gravity at geographic center of continental US (39.833333, -98.585522) at sea level
    /// <https://en.wikipedia.org/wiki/Geographic_center_of_the_United_States#Contiguous_United_States>
    /// Calculated using the WGS-84 formula <https://en.wikipedia.org/wiki/Theoretical_gravity#Somigliana_equation>
    ACC_GRAV,
    Acceleration,
    9.801_548_494_963_14
);

unit_const!(KELVIN, ThermodynamicTemperature, 1.0);
unit_const!(KELVIN_INT, TemperatureInterval, 1.0);
unit_const!(M2PS2K, SpecificHeatCapacity, 1.0);
unit_const!(PASCAL, Pressure, 1.0);

// TODO: make this variable
pub fn rho_air() -> MassDensity {
    KGPM3 * 1.225
}

pub fn get_rho_air(temperature: ThermodynamicTemperature, elevation: Length) -> MassDensity {
    let cur_elevation_std_temp = (15.04 - 0.00649 * elevation.get::<meter>() + 273.15) * KELVIN;
    let cur_pressure = (101.29e3 * PASCAL)
        * ((cur_elevation_std_temp / (288.08 * KELVIN))
            .get::<ratio>()
            .powf(5.256));
    cur_pressure / (287.0 * M2PS2K) / temperature
}
