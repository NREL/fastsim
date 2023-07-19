use crate::imports::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, HistoryVec)]
#[altrios_api(
    #[new]
    fn __new__(
        time_seconds: Option<f64>,
        offset_meters: Option<f64>,
        velocity_meters_per_second: Option<f64>,
        dt_seconds: Option<f64>,
    ) -> Self {
        Self::new(
            time_seconds.map(|x| x * uc::S),
            offset_meters.map(|x| x * uc::M),
            velocity_meters_per_second.map(|x| x * uc::MPS),
            dt_seconds.map(|x| x * uc::S),
        )
    }
)]
pub struct InitTrainState {
    pub time: si::Time,
    pub offset: si::Length,
    pub velocity: si::Velocity,
    pub dt: si::Time,
}

impl Default for InitTrainState {
    fn default() -> Self {
        Self {
            time: si::Time::ZERO,
            offset: f64::NAN * uc::M,
            velocity: si::Velocity::ZERO,
            dt: uc::S,
        }
    }
}

impl InitTrainState {
    pub fn new(
        time: Option<si::Time>,
        offset: Option<si::Length>,
        velocity: Option<si::Velocity>,
        dt: Option<si::Time>,
    ) -> Self {
        let base = InitTrainState::default();
        Self {
            time: time.unwrap_or(base.time),
            offset: offset.unwrap_or(base.offset),
            velocity: velocity.unwrap_or(base.velocity),
            dt: dt.unwrap_or(base.dt),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, HistoryVec, PartialEq)]
#[altrios_api(
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn __new__(
        offset_meters: f64,
        length_meters: f64,
        mass_static_kilograms: f64,
        mass_adj_kilograms: f64,
        mass_freight_kilograms: f64,
        time_seconds: Option<f64>,
        i: Option<usize>,
        velocity_meters_per_second: Option<f64>,
        dt_seconds: Option<f64>,
    ) -> Self {
        Self::new(
            time_seconds.map(|x| x * uc::S),
            i,
            offset_meters * uc::M,
            velocity_meters_per_second.map(|x| x * uc::MPS),
            dt_seconds.map(|x| x * uc::S),
            length_meters * uc::M,
            mass_static_kilograms * uc::KG,
            mass_adj_kilograms * uc::KG,
            mass_freight_kilograms * uc::KG,
        )
    }
)]
pub struct TrainState {
    pub time: si::Time,
    pub i: usize,
    pub offset: si::Length,
    pub offset_back: si::Length,
    pub total_dist: si::Length,
    pub velocity: si::Velocity,
    pub speed_limit: si::Velocity,
    pub speed_target: si::Velocity,

    pub dt: si::Time,
    pub length: si::Length,
    pub mass_static: si::Mass,
    pub mass_adj: si::Mass,
    pub mass_freight: si::Mass,

    pub weight_static: si::Force,
    pub res_rolling: si::Force,
    pub res_bearing: si::Force,
    pub res_davis_b: si::Force,
    pub res_aero: si::Force,
    pub res_grade: si::Force,
    pub res_curve: si::Force,

    /// Grade at front of train
    pub grade_front: si::Ratio,
    /// Elevation at front of train
    pub elev_front: si::Length,

    /// Power to overcome train resistance forces
    pub pwr_res: si::Power,
    /// Power to overcome inertial forces
    pub pwr_accel: si::Power,

    pub pwr_whl_out: si::Power,
    pub energy_whl_out: si::Energy,
    /// Energy out during positive or zero traction
    pub energy_whl_out_pos: si::Energy,
    /// Energy out during negative traction (positive value means negative traction)
    pub energy_whl_out_neg: si::Energy,
}

impl Default for TrainState {
    fn default() -> Self {
        Self {
            time: Default::default(),
            i: 1,
            offset: Default::default(),
            offset_back: Default::default(),
            total_dist: si::Length::ZERO,
            velocity: Default::default(),
            speed_limit: Default::default(),
            dt: uc::S,
            length: Default::default(),
            mass_static: Default::default(),
            mass_adj: Default::default(),
            mass_freight: Default::default(),
            elev_front: Default::default(),
            energy_whl_out: Default::default(),
            grade_front: Default::default(),
            speed_target: Default::default(),
            weight_static: Default::default(),
            res_rolling: Default::default(),
            res_bearing: Default::default(),
            res_davis_b: Default::default(),
            res_aero: Default::default(),
            res_grade: Default::default(),
            res_curve: Default::default(),
            pwr_res: Default::default(),
            pwr_accel: Default::default(),
            pwr_whl_out: Default::default(),
            energy_whl_out_pos: Default::default(),
            energy_whl_out_neg: Default::default(),
        }
    }
}

impl TrainState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time: Option<si::Time>,
        i: Option<usize>,
        offset: si::Length,
        velocity: Option<si::Velocity>,
        dt: Option<si::Time>,
        // the following variables probably won't ever change so it'd be good to have a separate train params object
        length: si::Length,
        mass_static: si::Mass,
        mass_adj: si::Mass,
        mass_freight: si::Mass,
    ) -> Self {
        Self {
            time: time.unwrap_or_default(),
            i: i.unwrap_or(1),
            offset,
            offset_back: offset - length,
            total_dist: si::Length::ZERO,
            velocity: velocity.unwrap_or_default(),
            speed_limit: velocity.unwrap_or_default(),
            dt: dt.unwrap_or(uc::S),
            length,
            mass_static,
            mass_adj,
            mass_freight,
            ..Self::default()
        }
    }

    pub fn res_net(&self) -> si::Force {
        self.res_rolling
            + self.res_bearing
            + self.res_davis_b
            + self.res_aero
            + self.res_grade
            + self.res_curve
    }
}

impl Valid for TrainState {
    fn valid() -> Self {
        Self {
            length: 2000.0 * uc::M,
            offset: 2000.0 * uc::M,
            offset_back: si::Length::ZERO,
            mass_static: 6000.0 * uc::TON,
            mass_adj: 6200.0 * uc::TON,

            dt: uc::S,
            ..Self::default()
        }
    }
}

///TODO: Add new values!
impl ObjState for TrainState {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        si_chk_num_gtz_fin(&mut errors, &self.mass_static, "Mass static");
        si_chk_num_gtz_fin(&mut errors, &self.length, "Length");
        // si_chk_num_gtz_fin(&mut errors, &self.res_bearing, "Resistance bearing");
        // si_chk_num_fin(&mut errors, &self.res_davis_b, "Resistance Davis B");
        // si_chk_num_gtz_fin(&mut errors, &self.drag_area, "Drag area");
        errors.make_err()
    }
}
