use super::locomotive::{BatteryElectricLoco, ConventionalLoco, Dummy, HybridLoco};
use super::*;

/// Trait for ensuring consistency among locomotives and consists
#[enum_dispatch]
pub trait LocoTrait {
    /// returns current max power, current max power rate, and current max regen
    /// power that can be absorbed by the RES/battery
    fn set_cur_pwr_max_out(
        &mut self,
        pwr_aux: Option<si::Power>,
        dt: si::Time,
    ) -> anyhow::Result<()>;
    /// Save current state
    fn save_state(&mut self) {
        unimplemented!();
    }
    /// Step counter
    fn step(&mut self) {
        unimplemented!();
    }
    /// Get energy loss in components
    fn get_energy_loss(&self) -> si::Energy;
}

#[altrios_api]
#[derive(Default, Serialize, Deserialize, Clone, PartialEq, SerdeAPI)]
/// Wrapper struct for `Vec<Locomotive>` to expose various methods to Python.
pub struct Pyo3VecLocoWrapper(pub Vec<Locomotive>);

#[enum_dispatch]
pub trait SolvePower {
    /// Returns vector of locomotive tractive powers during positive traction events
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>>;
    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>>;
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
/// Similar to [self::Proportional], but positive traction conditions use locomotives with
/// ReversibleEnergyStorage preferentially, within their power limits.  Recharge is same as
/// `Proportional` variant.
pub struct RESGreedy;
impl SolvePower for RESGreedy {
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        let loco_pwr_out_vec: Vec<si::Power> = if state.pwr_out_deficit == si::Power::ZERO {
            // draw all power from RES-equipped locomotives
            loco_vec
                .iter()
                .map(|loco| match &loco.loco_type {
                    LocoType::ConventionalLoco(_) => si::Power::ZERO,
                    LocoType::HybridLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_reves * state.pwr_out_req
                    }
                    LocoType::BatteryElectricLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_reves * state.pwr_out_req
                    }
                    // if the Dummy is present in the consist, it should be the only locomotive
                    // and pwr_out_deficit should be 0.0
                    LocoType::Dummy(_) => state.pwr_out_req,
                })
                .collect()
        } else {
            // draw deficit power from conventional locomotives
            loco_vec
                .iter()
                .map(|loco| match &loco.loco_type {
                    LocoType::ConventionalLoco(_) => {
                        loco.state.pwr_out_max / state.pwr_out_max_non_reves
                            * state.pwr_out_deficit
                    }
                    LocoType::HybridLoco(_) => loco.state.pwr_out_max,
                    LocoType::BatteryElectricLoco(_) => loco.state.pwr_out_max,
                    LocoType::Dummy(_) => {
                        si::Power::ZERO /* this else branch should not happen when Dummy is present */
                    }
                })
                .collect()
        };
        utils::assert_almost_eq_uom(
            &loco_pwr_out_vec.iter().copied().sum(),
            &state.pwr_out_req,
            None,
        );
        Ok(loco_pwr_out_vec)
    }

    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        solve_negative_traction(loco_vec, state)
    }
}

fn get_pwr_regen_vec(loco_vec: &[Locomotive], regen_frac: si::Ratio) -> Vec<si::Power> {
    loco_vec
        .iter()
        .map(|loco| match &loco.loco_type {
            // no braking power from conventional locos if there is capacity to regen all power
            LocoType::ConventionalLoco(_) => si::Power::ZERO,
            LocoType::HybridLoco(_) => loco.state.pwr_regen_max * regen_frac,
            LocoType::BatteryElectricLoco(_) => loco.state.pwr_regen_max * regen_frac,
            // if the Dummy is present in the consist, it should be the only locomotive
            // and pwr_regen_deficit should be 0.0
            LocoType::Dummy(_) => si::Power::ZERO,
        })
        .collect()
}

/// Used for apportioning negative tractive power throughout consist for several
/// [PowerDistributionControlType] variants
fn solve_negative_traction(
    loco_vec: &[Locomotive],
    consist_state: &ConsistState,
) -> anyhow::Result<Vec<si::Power>> {
    // positive during any kind of negative traction event
    let pwr_brake_req = -consist_state.pwr_out_req;

    // fraction of consist-level max regen required to fulfill required braking power
    let regen_frac = if consist_state.pwr_regen_max == si::Power::ZERO {
        // TODO: think carefully about whether this branch is correct
        si::Ratio::ZERO
    } else {
        (pwr_brake_req / consist_state.pwr_regen_max).min(uc::R * 1.)
    };
    let pwr_out_vec: Vec<si::Power> = if consist_state.pwr_regen_deficit == si::Power::ZERO {
        get_pwr_regen_vec(loco_vec, regen_frac)
    } else {
        // In this block, we know that all of the regen capability will be used so the goal is to spread
        // dynamic braking effort among the non-RES-equipped and then all locomotives up until they're doing
        // the same dynmamic braking effort
        let pwr_regen_vec = get_pwr_regen_vec(loco_vec, regen_frac);
        // extra dynamic braking power after regen has been subtracted off
        let pwr_surplus_vec: Vec<si::Power> = loco_vec
            .iter()
            .zip(&pwr_regen_vec)
            // this `unwrap` might cause problems for Dummy
            .map(|(loco, pwr_regen)| loco.electric_drivetrain().unwrap().pwr_out_max - *pwr_regen)
            .collect();
        let pwr_surplus_sum = pwr_surplus_vec
            .iter()
            .fold(0.0 * uc::W, |acc, &curr| acc + curr);

        // needed braking power not including regen per total available braking power not including regen
        let surplus_frac = consist_state.pwr_regen_deficit / pwr_surplus_sum;
        ensure!(
            surplus_frac >= si::Ratio::ZERO && surplus_frac <= uc::R,
            format_dbg!(surplus_frac),
        );
        // total dynamic braking, including regen
        let pwr_dyn_brake_vec: Vec<si::Power> = pwr_surplus_vec
            .iter()
            .zip(pwr_regen_vec)
            .map(|(pwr_surplus, pwr_regen)| *pwr_surplus * surplus_frac + pwr_regen)
            .collect();
        pwr_dyn_brake_vec
    };
    // negate it to be consistent with sign convention
    let pwr_out_vec: Vec<si::Power> = pwr_out_vec.iter().map(|x| -*x).collect();
    Ok(pwr_out_vec)
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
/// During positive traction, power is proportional to each locomotive's current max
/// available power.  During negative traction, any power that's less negative than the total
/// sum of the regen capacity is distributed to each locomotive with regen capacity, proportionally
/// to it's current max regen ability.
pub struct Proportional;
impl SolvePower for Proportional {
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        Ok(loco_vec
            .iter()
            .map(|loco| {
                // loco.state.pwr_out_max already accounts for rate
                loco.state.pwr_out_max / state.pwr_out_max * state.pwr_out_req
            })
            .collect())
    }

    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        solve_negative_traction(loco_vec, state)
    }
}

#[derive(PartialEq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
/// Similar to `Proportional`, regenerates greedily, but during positive traction, minimizes
/// cost function of `fuel_res_ratio` * fuel use + battery use at every `gss_interval`
/// time step.
pub struct GoldenSectionSearch {
    /// Ratio of fuel cost (abstract cost for solver -- could be dollars, MJ, etc.) to battery energy cost.
    pub fuel_res_ratio: f64,
    /// Time step interval for exercising GoldenSectionSearch. A value of `1` means it is solved at every time step.
    /// Solving the objective used by GoldenSectionSearch is computationlly expensive so care should be given when
    /// selecting this value.
    pub gss_interval: usize,
}
impl SolvePower for GoldenSectionSearch {
    fn solve_positive_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        if state.i == 1 || state.i % self.gss_interval == 0 {
            todo!() // not needed urgently
        } else {
            // use the previous iteration
            Ok(loco_vec.iter().map(|loco| loco.state.pwr_out).collect())
        }
    }

    fn solve_negative_traction(
        &mut self,
        loco_vec: &[Locomotive],
        state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        if state.i == 1 || state.i % self.gss_interval == 0 {
            todo!() // not needed urgently
        } else {
            // use the previous iteration
            Ok(loco_vec.iter().map(|loco| loco.state.pwr_out).collect())
        }
    }
}

#[derive(PartialEq, Eq, Clone, Deserialize, Serialize, Debug)]
/// Control strategy for when locomotives are located at both the front and back of the train.
pub struct FrontAndBack;
impl SerdeAPI for FrontAndBack {}
impl SolvePower for FrontAndBack {
    fn solve_positive_traction(
        &mut self,
        _loco_vec: &[Locomotive],
        _state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!() // not needed urgently
    }

    fn solve_negative_traction(
        &mut self,
        _loco_vec: &[Locomotive],
        _state: &ConsistState,
    ) -> anyhow::Result<Vec<si::Power>> {
        todo!() // not needed urgently
    }
}
/// Variants of this enum are used to determine what control strategy gets used for distributing
/// power required from or delivered to during negative tractive power each locomotive.
#[enum_dispatch(SolvePower)]
#[derive(PartialEq, Clone, Deserialize, Serialize, Debug, SerdeAPI)]
pub enum PowerDistributionControlType {
    RESGreedy,
    Proportional,
    GoldenSectionSearch,
    FrontAndBack,
}
impl Default for PowerDistributionControlType {
    fn default() -> Self {
        Self::RESGreedy(RESGreedy)
    }
}
