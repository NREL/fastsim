use crate::imports::*;

pub fn min_speed(speed_old: si::Velocity, speed_new: si::Velocity) -> si::Velocity {
    if speed_old.is_sign_positive() & speed_new.is_sign_positive() {
        speed_old.min(speed_new)
    } else {
        -speed_old.abs().min(speed_new.abs())
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, SerdeAPI)]
#[altrios_api]
pub struct SpeedLimit {
    pub offset_start: si::Length,
    pub offset_end: si::Length,
    pub speed: si::Velocity,
}

impl Valid for SpeedLimit {
    fn valid() -> Self {
        Self {
            offset_start: si::Length::ZERO,
            offset_end: uc::M * 10000.0,
            speed: uc::MPS * 20.0,
        }
    }
}

impl ObjState for SpeedLimit {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset_start, "Offset start");
        si_chk_num_gez(&mut errors, &self.offset_end, "Offset end");
        si_chk_num(&mut errors, &self.speed, "Speed");
        if self.offset_start > self.offset_end {
            errors.push(anyhow!(
                "Offset end = {:?} must be at least equal to offset start = {:?}!",
                self.offset_end,
                self.offset_start
            ));
        }
        errors.make_err()
    }
}

impl Valid for Vec<SpeedLimit> {
    fn valid() -> Self {
        let speed_limit = SpeedLimit::valid();
        vec![
            speed_limit,
            SpeedLimit {
                offset_start: speed_limit.offset_end * 0.5,
                offset_end: speed_limit.offset_end,
                speed: speed_limit.speed * 0.5,
            },
        ]
    }
}

impl ObjState for [SpeedLimit] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }

    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Speed limit");
        early_err!(errors, "Speed limits");

        if self
            .windows(2)
            .any(|w| w[0].offset_start == w[1].offset_start && w[0].offset_end == w[1].offset_end)
        {
            errors.push(anyhow!("Speed limit offset pairs must be unique!"));
        }
        if !utils::is_sorted(self) {
            errors.push(anyhow!("Speed limits must be sorted!"));
        }
        errors.make_err()
    }
}

#[cfg(test)]
mod test_speed_limit {
    use super::*;
    use crate::testing::*;

    impl Cases for SpeedLimit {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    offset_start: Self::valid().offset_end,
                    ..Self::valid()
                },
                Self {
                    offset_end: Self::valid().offset_start,
                    ..Self::valid()
                },
                Self {
                    offset_end: uc::M * f64::INFINITY,
                    ..Self::valid()
                },
                Self {
                    speed: uc::MPS * f64::INFINITY,
                    ..Self::valid()
                },
                Self {
                    speed: si::Velocity::ZERO,
                    ..Self::valid()
                },
                Self {
                    speed: uc::MPS * f64::NEG_INFINITY,
                    ..Self::valid()
                },
            ]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    offset_start: uc::M * f64::NEG_INFINITY,
                    ..Self::valid()
                },
                Self {
                    offset_start: -uc::M,
                    ..Self::valid()
                },
                Self {
                    offset_start: Self::valid().offset_end + uc::M,
                    ..Self::valid()
                },
                Self {
                    offset_start: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    offset_end: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    speed: uc::MPS * f64::NAN,
                    ..Self::valid()
                },
            ]
        }
    }
    check_cases!(SpeedLimit);
}

#[cfg(test)]
mod test_speed_limits {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<SpeedLimit> {
        fn fake_cases() -> Vec<Self> {
            vec![vec![]]
        }
        fn real_cases() -> Vec<Self> {
            vec![Self::valid(), vec![SpeedLimit::valid()]]
        }
    }
    check_cases!(Vec<SpeedLimit>);
    check_vec_elems!(SpeedLimit);
    check_vec_sorted!(SpeedLimit);
}
