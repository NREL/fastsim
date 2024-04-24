pub use super::*;

/// Possible drive wheel configurations for traction limit calculations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
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

#[pyo3_api]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, HistoryMethods, SerdeAPI)]
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
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wheel_radius: Option<si::Length>,
    /// Tire code (optional method of calculating wheel radius)
    #[serde(default)]
    #[api(skip_get, skip_set)]
    #[serde(skip_serializing_if = "Option::is_none")]
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

impl TryFrom<&fastsim_2::vehicle::RustVehicle> for Chassis {
    type Error = anyhow::Error;
    fn try_from(f2veh: &fastsim_2::vehicle::RustVehicle) -> Result<Self, Self::Error> {
        // TODO: check this sign convention
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
    fn mass(&self) -> anyhow::Result<si::Mass> {
        let derived_mass = self.derived_mass()?;
        if let (Some(derived_mass), Some(set_mass)) = (derived_mass, self.mass) {
            ensure!(
                utils::almost_eq_uom(&set_mass, &derived_mass, None),
                format!(
                    "{}",
                    format_dbg!(utils::almost_eq_uom(&set_mass, &derived_mass, None)),
                )
            );
            Ok(set_mass)
        } else if let Some(derived_mass) = derived_mass {
            Ok(derived_mass)
        } else if let Some(set_mass) = self.mass {
            Ok(set_mass)
        } else {
            bail!(
                "Not all mass fields in `{}` are set and mass field is `None`.",
                stringify!(Chassis)
            )
        }
    }

    fn set_mass(&mut self, new_mass: Option<si::Mass>) -> anyhow::Result<()> {
        let derived_mass = self.derived_mass()?;
        match new_mass {
            Some(new_mass) => {
                self.mass = Some(new_mass);
                if let Some(dm) = derived_mass {
                    if dm != new_mass {
                        // set all the field with mass to `None` because their
                        // values are not consistent with new mass
                        self.expunge_mass_fields();
                    }
                }
            }
            None => {
                self.mass = Some(derived_mass.with_context(|| {
                    format!(
                        "Not all mass fields in `{}` are set and no mass was provided.",
                        stringify!(Chassis)
                    )
                })?);
            }
        }
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
        self.glider_mass = None;
        self.cargo_mass = None;
    }
}
