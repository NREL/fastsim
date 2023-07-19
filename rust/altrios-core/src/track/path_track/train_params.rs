use super::super::link::*;
use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct TrainParams {
    pub length: si::Length,
    pub speed_max: si::Velocity,
    pub mass_total: si::Mass,
    pub mass_per_brake: si::Mass,
    pub axle_count: u32,
    pub train_type: TrainType,
    pub curve_coeff_0: si::Ratio,
    pub curve_coeff_1: si::Ratio,
    pub curve_coeff_2: si::Ratio,
}

impl TrainParams {
    pub fn speed_set_applies(&self, speed_set: &SpeedSet) -> bool {
        if self.train_type != speed_set.train_type {
            return false;
        }
        for speed_param in &speed_set.speed_params {
            if !{
                match speed_param.limit_type {
                    LimitType::MassTotal => speed_param
                        .compare_type
                        .applies(self.mass_total, speed_param.limit_val * uc::KG),
                    LimitType::MassPerBrake => speed_param
                        .compare_type
                        .applies(self.mass_per_brake, speed_param.limit_val * uc::KG),
                    LimitType::AxleCount => speed_param
                        .compare_type
                        .applies(self.axle_count, speed_param.limit_val as u32),
                }
            } {
                return false;
            }
        }
        true
    }
}

impl Valid for TrainParams {
    fn valid() -> Self {
        Self {
            length: uc::M * 2000.0,
            speed_max: uc::MPS * 25.0,
            mass_total: uc::TON * 143.0 * 100.0,
            mass_per_brake: uc::TON * 143.0,
            axle_count: 100 * 4,
            train_type: TrainType::Freight,
            curve_coeff_0: si::Ratio::ZERO,
            curve_coeff_1: si::Ratio::ZERO,
            curve_coeff_2: si::Ratio::ZERO,
        }
    }
}

impl ObjState for TrainParams {
    fn is_fake(&self) -> bool {
        self.length == si::Length::ZERO
    }
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            si_chk_num_eqz(&mut errors, &self.length, "Length");
            si_chk_num_eqz(&mut errors, &self.speed_max, "Speed max");
            si_chk_num_eqz(&mut errors, &self.mass_total, "Mass total");
            si_chk_num_eqz(&mut errors, &self.mass_per_brake, "Mass per brake");
            if self.axle_count != 0 {
                errors.push(anyhow!(
                    "Axle count = {:?} must equal zero!",
                    self.axle_count
                ));
            }
            validate_field_fake(&mut errors, &self.train_type, "Train type");
            si_chk_num_eqz(&mut errors, &self.curve_coeff_0, "Curve coeff 0");
            si_chk_num_eqz(&mut errors, &self.curve_coeff_1, "Curve coeff 1");
            si_chk_num_eqz(&mut errors, &self.curve_coeff_2, "Curve coeff 2");
        } else {
            si_chk_num_gtz_fin(&mut errors, &self.length, "Length");
            si_chk_num_gtz_fin(&mut errors, &self.speed_max, "Speed max");
            si_chk_num_gtz_fin(&mut errors, &self.mass_total, "Mass total");
            si_chk_num_gtz_fin(&mut errors, &self.mass_per_brake, "Mass per brake");
            if self.axle_count == 0 {
                errors.push(anyhow!(
                    "Axle count = {:?} must be a number larger than zero!",
                    self.axle_count
                ));
            }
            validate_field_real(&mut errors, &self.train_type, "Train type");
            si_chk_num_fin(&mut errors, &self.curve_coeff_0, "Curve coeff 0");
            si_chk_num_fin(&mut errors, &self.curve_coeff_1, "Curve coeff 1");
            si_chk_num_fin(&mut errors, &self.curve_coeff_2, "Curve coeff 2");
        }

        errors.make_err()
    }
}
