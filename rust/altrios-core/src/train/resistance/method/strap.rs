use super::super::kind::*;
use super::super::ResMethod;
use crate::imports::*;
use crate::track::{LinkPoint, PathResCoeff, PathTpc};
use crate::train::TrainState;

#[altrios_api]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, SerdeAPI)]
pub struct Strap {
    bearing: bearing::Basic,
    rolling: rolling::Basic,
    davis_b: davis_b::Basic,
    aerodynamic: aerodynamic::Basic,
    grade: path_res::Strap,
    curve: path_res::Strap,
}

impl Strap {
    pub fn new(
        bearing: bearing::Basic,
        rolling: rolling::Basic,
        davis_b: davis_b::Basic,
        aerodynamic: aerodynamic::Basic,
        grade: path_res::Strap,
        curve: path_res::Strap,
    ) -> Self {
        Self {
            bearing,
            rolling,
            davis_b,
            aerodynamic,
            grade,
            curve,
        }
    }
}
impl ResMethod for Strap {
    fn update_res<const DIR: DirT>(
        &mut self,
        state: &mut TrainState,
        path_tpc: &PathTpc,
    ) -> anyhow::Result<()> {
        state.offset_back = state.offset - state.length;
        state.weight_static = state.mass_static * uc::ACC_GRAV;
        state.res_bearing = self.bearing.calc_res();
        state.res_rolling = self.rolling.calc_res(state);
        state.res_davis_b = self.davis_b.calc_res(state);
        state.res_aero = self.aerodynamic.calc_res(state);
        state.res_grade = self.grade.calc_res::<DIR>(path_tpc.grades(), state)?;
        state.res_curve = self.curve.calc_res::<DIR>(path_tpc.curves(), state)?;
        state.grade_front = self.grade.res_coeff_front(path_tpc.grades());
        state.elev_front = self.grade.res_net_front(path_tpc.grades(), state);
        Ok(())
    }
    fn fix_cache(&mut self, link_point_del: &LinkPoint) {
        self.grade.fix_cache(link_point_del.grade_count);
        self.curve.fix_cache(link_point_del.curve_count);
    }
}
impl Valid for Strap {
    fn valid() -> Self {
        Self {
            bearing: bearing::Basic::new(40.0 * 100.0 * uc::LBF),
            rolling: rolling::Basic::new(1.5 * uc::LB / uc::TON),
            davis_b: davis_b::Basic::new(0.03 / uc::MPH * uc::LB / uc::TON),
            aerodynamic: aerodynamic::Basic::new(
                5.0 * 100.0 * uc::FT2 / 10000.0 / 1.225 * uc::MPS / uc::MPH * uc::MPS / uc::MPH
                    * uc::LBF
                    / uc::N
                    * 100.0,
            ),
            grade: path_res::Strap::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
            curve: path_res::Strap::new(&Vec::<PathResCoeff>::valid(), &TrainState::valid())
                .unwrap(),
        }
    }
}
