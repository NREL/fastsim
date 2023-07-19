use crate::imports::*;
use crate::track::PathResCoeff;
use crate::train::TrainState;

fn calc_res_val(res_coeff: si::Ratio, state: &TrainState) -> si::Force {
    res_coeff * state.weight_static
}

#[altrios_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, SerdeAPI)]
pub struct Point {
    idx: usize,
}
impl Point {
    pub fn new(vals: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        Ok(Self {
            idx: vals.calc_idx::<{ Dir::Fwd }>(state.offset - state.length * 0.5, 0)?,
        })
    }
    pub fn calc_res<const DIR: DirT>(
        &mut self,
        vals: &[PathResCoeff],
        state: &TrainState,
    ) -> anyhow::Result<si::Force> {
        self.idx = vals.calc_idx::<DIR>(state.offset - state.length * 0.5, self.idx)?;
        Ok(calc_res_val(vals[self.idx].res_coeff, state))
    }
    pub fn res_coeff_front(&self, vals: &[PathResCoeff]) -> si::Ratio {
        vals[self.idx].res_coeff
    }
    pub fn res_net_front(&self, vals: &[PathResCoeff], state: &TrainState) -> si::Length {
        vals[self.idx].calc_res_val(state.offset)
    }
    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx -= idx_sub;
    }
}

#[ext(CalcResStrap)]
impl [PathResCoeff] {
    fn calc_res_strap(&self, idx_front: usize, idx_back: usize, state: &TrainState) -> si::Ratio {
        debug_assert!(state.length > si::Length::ZERO);
        (self[idx_front].calc_res_val(state.offset)
            - self[idx_back].calc_res_val(state.offset_back))
            / state.length
    }
}

#[altrios_api]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq, SerdeAPI)]
pub struct Strap {
    idx_front: usize,
    idx_back: usize,
}

impl Strap {
    pub fn new(vals: &[PathResCoeff], state: &TrainState) -> anyhow::Result<Self> {
        if vals.len() <= 1 {
            Ok(Self {
                idx_front: 0,
                idx_back: 0,
            })
        } else {
            let idx_back = vals.calc_idx::<{ Dir::Fwd }>(state.offset - state.length, 0)?;
            Ok(Self {
                idx_back,
                idx_front: vals.calc_idx::<{ Dir::Fwd }>(state.offset, idx_back)?,
            })
        }
    }
    pub fn calc_res<const DIR: DirT>(
        &mut self,
        vals: &[PathResCoeff],
        state: &TrainState,
    ) -> anyhow::Result<si::Force> {
        if DIR == Dir::Fwd || DIR == Dir::Unk {
            self.idx_front = vals.calc_idx::<DIR>(state.offset, self.idx_front)?;
        }
        if DIR == Dir::Bwd || DIR == Dir::Unk {
            self.idx_back = vals.calc_idx::<DIR>(state.offset_back, self.idx_back)?;
        }
        let res_coeff = if self.idx_front == self.idx_back {
            vals[self.idx_front].res_coeff
        } else {
            if DIR == Dir::Fwd {
                self.idx_back = vals.calc_idx::<DIR>(state.offset_back, self.idx_back)?;
            } else if DIR == Dir::Bwd {
                self.idx_front = vals.calc_idx::<DIR>(state.offset, self.idx_front)?;
            }
            vals.calc_res_strap(self.idx_front, self.idx_back, state)
        };

        Ok(calc_res_val(res_coeff, state))
    }
    pub fn res_coeff_front(&self, vals: &[PathResCoeff]) -> si::Ratio {
        vals[self.idx_front].res_coeff
    }
    pub fn res_net_front(&self, vals: &[PathResCoeff], state: &TrainState) -> si::Length {
        vals[self.idx_front].calc_res_val(state.offset)
    }
    pub fn fix_cache(&mut self, idx_sub: usize) {
        self.idx_back -= idx_sub;
        self.idx_front -= idx_sub;
    }
}
