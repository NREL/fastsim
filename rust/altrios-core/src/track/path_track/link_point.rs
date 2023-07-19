use super::super::LinkIdx;
use crate::imports::*;

/// Specifies the relative location of a link within the PathTpc
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd, SerdeAPI)]
#[altrios_api]
pub struct LinkPoint {
    #[api(skip_set)]
    pub offset: si::Length,
    #[api(skip_set)]
    pub grade_count: usize,
    #[api(skip_set)]
    pub curve_count: usize,
    #[api(skip_set)]
    pub cat_power_count: usize,
    #[api(skip_set)]
    pub link_idx: LinkIdx,
}

impl LinkPoint {
    pub fn add_counts(&mut self, other: &Self) {
        self.grade_count += other.grade_count;
        self.curve_count += other.curve_count;
        self.cat_power_count += other.cat_power_count;
    }
}

impl Valid for LinkPoint {
    fn valid() -> Self {
        Self {
            offset: uc::M * 10000.0,
            grade_count: 2,
            curve_count: 2,
            cat_power_count: 0,
            link_idx: LinkIdx::default(),
        }
    }
}

impl ObjState for LinkPoint {
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        errors.make_err()
    }
}

impl GetOffset for LinkPoint {
    fn get_offset(&self) -> si::Length {
        self.offset
    }
}

impl Valid for Vec<LinkPoint> {
    fn valid() -> Self {
        vec![
            LinkPoint {
                link_idx: LinkIdx::valid(),
                ..LinkPoint::default()
            },
            LinkPoint::valid(),
        ]
    }
}

impl ObjState for [LinkPoint] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }
    fn validate(&self) -> Result<(), crate::validate::ValidationErrors> {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Link point");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two link points!"));
        }
        early_err!(errors, "Link points");

        for link_point in &self[..(self.len() - 1)] {
            if link_point.link_idx.is_fake() {
                errors.push(anyhow!(
                    "All link point link indices (except for the last one) must be real!"
                ));
            }
        }
        if self.last().unwrap().link_idx.is_real() {
            errors.push(anyhow!("Last link point link index must be fake!"));
        }

        if !self.windows(2).all(|w| w[0].offset < w[1].offset) {
            errors.push(anyhow!("Link point offsets must be sorted and unique!"));
        }

        errors.make_err()
    }
}

#[cfg(test)]
mod test_link_point {
    use super::*;
    use crate::testing::*;

    impl Cases for LinkPoint {
        fn real_cases() -> Vec<Self> {
            vec![Self::default(), Self::valid()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    offset: -uc::M,
                    ..Self::default()
                },
                Self {
                    offset: uc::M * f64::NAN,
                    ..Self::default()
                },
                Self {
                    offset: -uc::M,
                    ..Self::valid()
                },
                Self {
                    offset: uc::M * f64::NAN,
                    ..Self::valid()
                },
            ]
        }
    }

    check_cases!(LinkPoint);
}

#[cfg(test)]
mod test_link_points {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<LinkPoint> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid(), {
                let mut base = Self::valid();
                base.push(*base.last().unwrap());
                base.last_mut().unwrap().offset += uc::M;
                let base_len = base.len();
                base[base_len - 2].link_idx = LinkIdx::valid();
                base
            }]
        }
        fn fake_cases() -> Vec<Self> {
            vec![vec![]]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                vec![LinkPoint::default()],
                vec![LinkPoint::valid()],
                vec![LinkPoint::valid(), LinkPoint::valid()],
                vec![LinkPoint::valid(), LinkPoint::default()],
                vec![LinkPoint::default(), LinkPoint::valid()],
                vec![LinkPoint::default(), LinkPoint::default()],
                Self::valid().into_iter().rev().collect::<Self>(),
                {
                    let mut base = Self::valid();
                    base.push(*base.last().unwrap());
                    base.last_mut().unwrap().offset += uc::M;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.last_mut().unwrap().offset = base.first().unwrap().offset;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.first_mut().unwrap().offset = base.last().unwrap().offset;
                    base
                },
                {
                    let mut base = Self::valid();
                    base.first_mut().unwrap().offset = base.last().unwrap().offset + uc::M;
                    base
                },
            ]
        }
    }
    check_cases!(Vec<LinkPoint>);
    check_vec_elems!(LinkPoint);
    check_vec_sorted!(LinkPoint);
    check_vec_duplicates!(LinkPoint);
}
