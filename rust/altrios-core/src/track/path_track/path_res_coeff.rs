use crate::imports::*;

/// Struct containing linear resistance coefficients for a particular offset w.r.t. `PathTpc`
#[derive(Clone, Copy, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize, SerdeAPI)]
#[altrios_api]
pub struct PathResCoeff {
    #[api(skip_set)]
    pub offset: si::Length,
    #[api(skip_set)]
    pub res_coeff: si::Ratio,
    #[api(skip_set)]
    pub res_net: si::Length,
}

impl PathResCoeff {
    pub fn calc_res_val(&self, offset: si::Length) -> si::Length {
        self.res_net + self.res_coeff * (offset - self.offset)
    }
}

impl GetOffset for PathResCoeff {
    fn get_offset(&self) -> si::Length {
        self.offset
    }
}

impl Valid for PathResCoeff {}

impl ObjState for PathResCoeff {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gez(&mut errors, &self.offset, "Offset");
        si_chk_num_fin(&mut errors, &self.res_net, "Res net");
        si_chk_num_fin(&mut errors, &self.res_coeff, "Res coeff");
        errors.make_err()
    }
}

impl Valid for Vec<PathResCoeff> {
    fn valid() -> Self {
        let offset_end = uc::M * 10000.0;
        let coeff_max = uc::M * 50.0;
        vec![
            PathResCoeff {
                res_coeff: coeff_max / offset_end,
                ..PathResCoeff::valid()
            },
            PathResCoeff {
                offset: offset_end,
                res_net: coeff_max,
                ..PathResCoeff::valid()
            },
        ]
    }
}

impl ObjState for [PathResCoeff] {
    fn is_fake(&self) -> bool {
        self.is_empty()
    }

    fn validate(&self) -> ValidationResults {
        early_fake_ok!(self);
        let mut errors = ValidationErrors::new();
        validate_slice_real(&mut errors, self, "Path res coeff");
        if self.len() < 2 {
            errors.push(anyhow!("There must be at least two path res coeffs!"));
        }
        if !self.windows(2).all(|w| w[0].offset < w[1].offset) {
            errors.push(anyhow!("Offsets must be sorted and unique!"));
        }
        if !self
            .windows(2)
            .all(|w| w[0].res_coeff == (w[1].res_net - w[0].res_net) / (w[1].offset - w[0].offset))
        {
            errors.push(anyhow!(
                "Res coeff must equal change in res net over change in offset!"
            ));
        }
        si_chk_num_eqz(
            &mut errors,
            &self.last().unwrap().res_coeff,
            "Last res coeff",
        );
        errors.make_err()
    }
}
