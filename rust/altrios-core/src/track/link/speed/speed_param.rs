use crate::imports::*;

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, SerdeAPI)]
#[repr(u8)]
pub enum LimitType {
    //CivilSpeed = 1,
    //MaxPermissibleSpeed = 2,
    MassTotal = 3,
    #[default]
    MassPerBrake = 4,
    AxleCount = 5,
}

impl Valid for LimitType {}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, SerdeAPI)]
#[repr(u8)]
pub enum CompareType {
    #[default]
    TpEqualRp = 1,
    TpGreaterThanRp = 2,
    TpLessThanRp = 3,
    TpGreaterThanEqualRp = 4,
    TpLessThanEqualRp = 5,
}

impl CompareType {
    pub fn applies<T>(&self, train_param: T, limit_param: T) -> bool
    where
        T: PartialEq + PartialOrd,
    {
        match self {
            Self::TpEqualRp => train_param == limit_param,
            Self::TpGreaterThanRp => train_param > limit_param,
            Self::TpLessThanRp => train_param < limit_param,
            Self::TpGreaterThanEqualRp => train_param >= limit_param,
            Self::TpLessThanEqualRp => train_param <= limit_param,
        }
    }
}

impl Valid for CompareType {
    fn valid() -> Self {
        Self::TpGreaterThanRp
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct SpeedParam {
    pub limit_val: f64,
    pub limit_type: LimitType,
    pub compare_type: CompareType,
}

impl Valid for SpeedParam {
    fn valid() -> Self {
        Self {
            limit_val: (100.0 * uc::TON).value,
            limit_type: LimitType::valid(),
            compare_type: CompareType::valid(),
        }
    }
}

impl ObjState for SpeedParam {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if let None | Some(Ordering::Less) = self.limit_val.partial_cmp(&0.0) {
            errors.push(anyhow!(
                "Limit val for {:?} = {:?} must be a positive number",
                self.limit_type,
                self.limit_val,
            ));
        }
        if self.limit_type == LimitType::AxleCount && self.limit_val.trunc() != self.limit_val {
            errors.push(anyhow!(
                "Limit val for {:?} = {:?} must also be an integer!",
                self.limit_type,
                self.limit_val
            ));
        }
        errors.make_err()
    }
}

impl Valid for Vec<SpeedParam> {}

impl ObjState for [SpeedParam] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Speed param");
        early_err!(errors, "Speed params");
        if self.windows(2).any(|w| w[0] == w[1]) {
            errors.push(anyhow!("Speed params must be unique!"));
        }
        errors.make_err()
    }
}

#[cfg(test)]
mod test_speed_param {
    use super::*;
    use crate::testing::*;

    impl Cases for SpeedParam {}
    check_cases!(SpeedParam);
}

#[cfg(test)]
mod test_speed_params {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<SpeedParam> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid(), vec![SpeedParam::valid()]]
        }
    }
    check_cases!(Vec<SpeedParam>);
    check_vec_elems!(SpeedParam);
    check_vec_duplicates!(SpeedParam);
}
