use crate::imports::*;

#[derive(Clone, Default, Debug, PartialEq, PartialOrd, Deserialize, Serialize, SerdeAPI)]
#[altrios_api]
/// Struct representing local train-level power limits for catenary charging
pub struct CatPowerLimit {
    /// start of current power limit
    pub offset_start: si::Length,
    /// end of current power limit
    pub offset_end: si::Length,
    /// maximum possible catenary charging rate  
    /// assumed to be identical for charging and discharging
    pub power_limit: si::Power,
    /// Optional user-defined catenary district
    pub district_id: Option<String>,
}

impl Valid for CatPowerLimit {
    fn valid() -> Self {
        Self {
            offset_start: si::Length::ZERO,
            offset_end: uc::M * 10000.0,
            power_limit: uc::W * 5.0e6,
            district_id: None,
        }
    }
}

impl ObjState for CatPowerLimit {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset_start, "Offset start");
        si_chk_num_gez(&mut errors, &self.offset_end, "Offset end");
        si_chk_num_gez(&mut errors, &self.power_limit, "Power limit");
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

impl Valid for Vec<CatPowerLimit> {
    fn valid() -> Self {
        vec![CatPowerLimit::valid()]
    }
}

impl ObjState for [CatPowerLimit] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Catenary power limit");
        early_err!(errors, "Catenary power limits");

        if self
            .windows(2)
            .any(|w| w[0].offset_end <= w[1].offset_start)
        {
            errors.push(anyhow!(
                "Catenary power limit offset pairs must be non-overlapping!"
            ));
        }
        errors.make_err()
    }
}
