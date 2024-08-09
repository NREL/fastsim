pub use super::*;

/// Possible drive wheel configurations for traction limit calculations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DriveTypes {
    /// Rear-wheel drive
    RWD,
    /// Front-wheel drive
    FWD,
    /// All-wheel drive
    AWD,
    /// 4-wheel drive
    FourWD,
}

impl SerdeAPI for DriveTypes {}
impl Init for DriveTypes {}

#[pyo3_api]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, HistoryMethods)]
/// Struct for simulating vehicle
pub struct Chassis {
    /// Aerodynamic drag coefficient
    pub drag_coef: si::Ratio,
    /// Projected frontal area for drag calculations
    pub frontal_area: si::Area,
    /// Wheel rolling resistance coefficient for the vehicle (i.e. all wheels included)
    pub wheel_rr_coef: si::Ratio,
    /// Wheel inertia per wheel
    pub wheel_inertia: si::MomentOfInertia,
    /// Number of wheels
    pub num_wheels: u8,
    /// Wheel radius
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub wheel_radius: Option<si::Length>,
    /// Tire code (optional method of calculating wheel radius)
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    #[api(skip_get, skip_set)]
    pub tire_code: Option<String>,
    /// Vehicle center of mass height
    pub cg_height: si::Length,
    /// Wheel coefficient of friction
    pub wheel_fric_coef: si::Ratio,
    #[api(skip_get, skip_set)]
    /// TODO: make getters and setters for this.
    /// Drive wheel configuration
    pub drive_type: DriveTypes,
    /// Fraction of vehicle weight on drive action when stationary
    pub drive_axle_weight_frac: si::Ratio,
    /// Wheel base length
    pub wheel_base: si::Length,
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mass: Option<si::Mass>,
    /// Vehicle mass excluding cargo, passengers, and powertrain components
    // TODO: make sure setter and getter get written
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) glider_mass: Option<si::Mass>,
    /// Cargo mass including passengers
    #[serde(default)]
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cargo_mass: Option<si::Mass>,
}

impl SerdeAPI for Chassis {}
impl Init for Chassis {}

impl TryFrom<&fastsim_2::vehicle::RustVehicle> for Chassis {
    type Error = anyhow::Error;
    fn try_from(f2veh: &fastsim_2::vehicle::RustVehicle) -> anyhow::Result<Self> {
        let drive_type = if f2veh.veh_cg_m < 0. {
            chassis::DriveTypes::RWD
        } else {
            chassis::DriveTypes::FWD
        };

        Ok(Self {
            drag_coef: f2veh.drag_coef * uc::R,
            frontal_area: f2veh.frontal_area_m2 * uc::M2,
            cg_height: f2veh.veh_cg_m * uc::M,
            wheel_fric_coef: f2veh.wheel_coef_of_fric * uc::R,
            drive_type,
            drive_axle_weight_frac: f2veh.drive_axle_weight_frac * uc::R,
            wheel_base: f2veh.wheel_base_m * uc::M,
            wheel_inertia: f2veh.wheel_inertia_kg_m2 * uc::KGM2,
            wheel_rr_coef: f2veh.wheel_rr_coef * uc::R,
            num_wheels: f2veh.num_wheels as u8,
            wheel_radius: Some(f2veh.wheel_radius_m * uc::M),
            tire_code: None,
            mass: None,
            glider_mass: Some(f2veh.glider_kg * uc::KG),
            cargo_mass: Some(f2veh.cargo_kg * uc::KG),
        })
    }
}

impl Mass for Chassis {
    fn mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        if let (Some(derived_mass), Some(set_mass)) = (derived_mass, self.mass) {
            ensure!(
                utils::almost_eq_uom(&set_mass, &derived_mass, None),
                format!(
                    "{}",
                    format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                )
            );
        }
        Ok(self.mass)
    }

    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        _side_effect: MassSideEffect,
    ) -> anyhow::Result<()> {
        let derived_mass = self
            .derived_mass()
            .with_context(|| anyhow!(format_dbg!()))?;
        if let (Some(derived_mass), Some(new_mass)) = (derived_mass, new_mass) {
            if derived_mass != new_mass {
                #[cfg(feature = "logging")]
                log::warn!("Derived mass does not match provided mass, setting `{}` constituent mass fields to `None`", stringify!(Chassis));
                self.expunge_mass_fields();
            }
        } else if new_mass.is_none() {
            #[cfg(feature = "logging")]
            log::debug!(
                "Provided mass is None, setting `{}` constituent mass fields to `None`",
                stringify!(Chassis)
            );
            self.expunge_mass_fields();
        }
        self.mass = new_mass;
        Ok(())
    }

    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>> {
        let mass =
            if let (Some(glider_mass), Some(cargo_mass)) = (self.glider_mass, self.cargo_mass) {
                Some(glider_mass + cargo_mass)
            } else if let (None, None) = (self.glider_mass, self.cargo_mass) {
                None
            } else {
                bail!(
                    "`{}` field masses are not consistently set to `Some` or `None`",
                    stringify!(Chassis)
                )
            };
        Ok(mass)
    }

    fn expunge_mass_fields(&mut self) {
        self.mass = None;
        self.glider_mass = None;
        self.cargo_mass = None;
    }
}
