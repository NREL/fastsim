use super::resistance::kind as res_kind;
use super::resistance::method as res_method;
use crate::consist::locomotive::locomotive_model::LocoType;
use crate::consist::Mass;

use super::{
    friction_brakes::*, rail_vehicle::RailVehicleMap, train_imports::*, InitTrainState,
    LinkIdxTime, SetSpeedTrainSim, SpeedLimitTrainSim, SpeedTrace, TrainState,
};
use crate::track::LocationMap;

use polars::prelude::*;
use polars_lazy::prelude::*;
use pyo3_polars::PyDataFrame;

#[altrios_api(
    #[new]
    fn __new__(
        rail_vehicle_type: String,
        cars_empty: u32,
        cars_loaded: u32,
        train_type: Option<TrainType>,
        train_length_meters: Option<f64>,
        train_mass_kilograms: Option<f64>,
    ) -> Self {
        Self::new(
            rail_vehicle_type,
            cars_empty,
            cars_loaded,
            train_type.unwrap_or(TrainType::Freight),
            train_length_meters.map(|v| v * uc::M),
            train_mass_kilograms.map(|v| v * uc::KG),
        )
    }

    #[pyo3(name = "make_train_params")]
    fn make_train_params_py(&self, rail_vehicle_map: RailVehicleMap) -> PyResult<TrainParams> {
        Ok(self.make_train_params(&rail_vehicle_map))
    }
)]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, SerdeAPI)]
pub struct TrainSummary {
    /// User-defined identifier for the car type on this train
    pub rail_vehicle_type: String,
    /// Train type matching one of the PTC types
    pub train_type: TrainType,
    /// Number of empty railcars on the train
    pub cars_empty: u32,
    /// Number of loaded railcars on the train
    pub cars_loaded: u32,
    /// Train length that overrides the railcar specific value
    #[api(skip_set, skip_get)]
    pub train_length: Option<si::Length>,
    /// Total train mass that overrides the railcar specific values
    #[api(skip_set, skip_get)]
    pub train_mass: Option<si::Mass>,
}

impl TrainSummary {
    pub fn new(
        rail_vehicle_type: String,
        cars_empty: u32,
        cars_loaded: u32,
        train_type: TrainType,
        train_length: Option<si::Length>,
        train_mass: Option<si::Mass>,
    ) -> Self {
        Self {
            rail_vehicle_type,
            train_type,
            cars_empty,
            cars_loaded,
            train_length,
            train_mass,
        }
    }

    pub fn cars_total(&self) -> u32 {
        self.cars_empty + self.cars_loaded
    }

    pub fn make_train_params(&self, rail_vehicle_map: &RailVehicleMap) -> TrainParams {
        let rail_vehicle = &rail_vehicle_map[&self.rail_vehicle_type];
        let mass_static = self.train_mass.unwrap_or(
            rail_vehicle.mass_static_empty * self.cars_empty as f64
                + rail_vehicle.mass_static_loaded * self.cars_loaded as f64,
        );

        let length = self
            .train_length
            .unwrap_or(rail_vehicle.length * self.cars_total() as f64);

        TrainParams {
            length,
            speed_max: rail_vehicle.speed_max_loaded.max(if self.cars_empty > 0 {
                rail_vehicle.speed_max_empty
            } else {
                uc::MPS * f64::INFINITY
            }),
            mass_total: mass_static,
            mass_per_brake: mass_static
                / (self.cars_total() * rail_vehicle.brake_count as u32) as f64,
            axle_count: self.cars_total() * rail_vehicle.axle_count as u32,
            train_type: self.train_type,
            curve_coeff_0: rail_vehicle.curve_coeff_0,
            curve_coeff_1: rail_vehicle.curve_coeff_1,
            curve_coeff_2: rail_vehicle.curve_coeff_2,
        }
    }
}

impl Valid for TrainSummary {
    fn valid() -> Self {
        Self {
            rail_vehicle_type: "Bulk".to_string(),
            train_type: TrainType::Freight,
            cars_empty: 0,
            cars_loaded: 100,
            train_length: None,
            train_mass: None,
        }
    }
}

#[altrios_api(
    #[new]
    fn __new__(
        train_id: String,
        origin_id: String,
        destination_id: String,
        train_summary: TrainSummary,
        loco_con: Consist,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self::new(
            train_id,
            origin_id,
            destination_id,
            train_summary,
            loco_con,
            init_train_state,
        )
    }

    #[pyo3(name = "make_set_speed_train_sim")]
    fn make_set_speed_train_sim_py(
        &self,
        rail_vehicle_map: RailVehicleMap,
        network: Vec<Link>,
        link_path: Vec<LinkIdx>,
        speed_trace: SpeedTrace,
        save_interval: Option<usize>
    ) -> anyhow::Result<SetSpeedTrainSim> {
        self.make_set_speed_train_sim(
            &rail_vehicle_map,
            &network,
            &link_path,
            speed_trace,
            save_interval
        )
    }

    #[pyo3(name = "make_speed_limit_train_sim")]
    fn make_speed_limit_train_sim_py(
        &self,
        rail_vehicle_map: RailVehicleMap,
        location_map: LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> PyResult<SpeedLimitTrainSim> {
        Ok(self.make_speed_limit_train_sim(
            &rail_vehicle_map,
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
        )?)
    }

)]
#[derive(Debug, Default, Clone, Deserialize, Serialize, PartialEq, SerdeAPI)]
pub struct TrainSimBuilder {
    /// Unique identifier for the train starting from 0
    pub train_id: String,
    /// Origin ID from train planner to map to track network locations
    pub origin_id: String,
    /// Destination ID from train planner to map to track network locations
    pub destination_id: String,
    pub train_summary: TrainSummary,
    pub loco_con: Consist,
    pub init_train_state: InitTrainState,
}

impl TrainSimBuilder {
    pub fn new(
        train_id: String,
        origin_id: String,
        destination_id: String,
        train_summary: TrainSummary,
        loco_con: Consist,
        init_train_state: Option<InitTrainState>,
    ) -> Self {
        Self {
            train_id,
            origin_id,
            destination_id,
            train_summary,
            loco_con,
            init_train_state: init_train_state.unwrap_or_default(),
        }
    }

    fn make_train_sim_parts(
        &self,
        rail_vehicle_map: &RailVehicleMap,
        save_interval: Option<usize>,
    ) -> anyhow::Result<(TrainState, PathTpc, TrainRes, FricBrake)> {
        let veh = &rail_vehicle_map[&self.train_summary.rail_vehicle_type];
        let train_params = self.train_summary.make_train_params(rail_vehicle_map);

        let length = train_params.length;
        // TODO: figure out what to do about rotational mass of locomotive components (e.g. axles, gearboxes, motor shafts)
        let mass_static = train_params.mass_total + self.loco_con.mass()?.unwrap();
        let cars_total = self.train_summary.cars_total() as f64;
        let mass_adj = mass_static + veh.mass_extra_per_axle * train_params.axle_count as f64;
        let mass_freight =
            si::Mass::ZERO.max(train_params.mass_total - veh.mass_static_empty * cars_total);
        let max_fric_braking = uc::ACC_GRAV
            * train_params.mass_total
            * (veh.braking_ratio_empty * self.train_summary.cars_empty as f64
                + veh.braking_ratio_loaded * self.train_summary.cars_loaded as f64)
            / cars_total;

        let start_offset = self.init_train_state.offset.max(length);
        let state = TrainState::new(
            Some(self.init_train_state.time),
            None,
            start_offset,
            Some(self.init_train_state.velocity),
            Some(self.init_train_state.dt),
            length,
            mass_static,
            mass_adj,
            mass_freight,
        );

        let path_tpc = PathTpc::new(train_params);

        let train_res = {
            let res_bearing = res_kind::bearing::Basic::new(
                veh.bearing_res_per_axle * train_params.axle_count as f64,
            );
            let res_rolling = res_kind::rolling::Basic::new(veh.rolling_ratio);
            let davis_b = res_kind::davis_b::Basic::new(veh.davis_b);
            let res_aero = res_kind::aerodynamic::Basic::new(
                veh.drag_area_empty * self.train_summary.cars_empty as f64
                    + veh.drag_area_loaded * self.train_summary.cars_loaded as f64,
            );
            let res_grade = res_kind::path_res::Strap::new(path_tpc.grades(), &state)?;
            let res_curve = res_kind::path_res::Strap::new(path_tpc.curves(), &state)?;
            TrainRes::Strap(res_method::Strap::new(
                res_bearing,
                res_rolling,
                davis_b,
                res_aero,
                res_grade,
                res_curve,
            ))
        };

        // brake propagation rate is 800 ft/s (about speed of sound)
        // ramp up duration is ~30 s
        // TODO: make this not hard coded!
        // TODO: remove save_interval from new function!
        let ramp_up_time = 0.0 * uc::S;
        let ramp_up_coeff = 0.6 * uc::R;

        let fric_brake = FricBrake::new(
            max_fric_braking,
            ramp_up_time,
            ramp_up_coeff,
            None,
            save_interval,
        );

        Ok((state, path_tpc, train_res, fric_brake))
    }

    pub fn make_set_speed_train_sim(
        &self,
        rail_vehicle_map: &RailVehicleMap,
        network: &[Link],
        link_path: &[LinkIdx],
        speed_trace: SpeedTrace,
        save_interval: Option<usize>,
    ) -> anyhow::Result<SetSpeedTrainSim> {
        let (state, mut path_tpc, train_res, _fric_brake) =
            self.make_train_sim_parts(rail_vehicle_map, save_interval)?;

        path_tpc.extend(network, link_path)?;
        Ok(SetSpeedTrainSim::new(
            self.loco_con.clone(),
            state,
            speed_trace,
            train_res,
            path_tpc,
            save_interval,
        ))
    }

    pub fn make_speed_limit_train_sim(
        &self,
        rail_vehicle_map: &RailVehicleMap,
        location_map: &LocationMap,
        save_interval: Option<usize>,
        simulation_days: Option<i32>,
        scenario_year: Option<i32>,
    ) -> anyhow::Result<SpeedLimitTrainSim> {
        let (state, path_tpc, train_res, fric_brake) =
            self.make_train_sim_parts(rail_vehicle_map, save_interval)?;

        Ok(SpeedLimitTrainSim::new(
            self.train_id.clone(),
            location_map.get(&self.origin_id).unwrap(),
            location_map.get(&self.destination_id).unwrap(),
            self.loco_con.clone(),
            state,
            train_res,
            path_tpc,
            fric_brake,
            save_interval,
            simulation_days,
            scenario_year,
        ))
    }
}

/// This may be deprecated soon! Slts building occurs in train planner.
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn build_speed_limit_train_sims(
    train_sim_builders: Vec<TrainSimBuilder>,
    rail_veh_map: RailVehicleMap,
    location_map: LocationMap,
    save_interval: Option<usize>,
    simulation_days: Option<i32>,
    scenario_year: Option<i32>,
) -> anyhow::Result<SpeedLimitTrainSimVec> {
    let mut speed_limit_train_sims = Vec::with_capacity(train_sim_builders.len());
    for tsb in train_sim_builders.iter() {
        speed_limit_train_sims.push(tsb.make_speed_limit_train_sim(
            &rail_veh_map,
            &location_map,
            save_interval,
            simulation_days,
            scenario_year,
        )?);
    }
    Ok(SpeedLimitTrainSimVec(speed_limit_train_sims))
}

#[allow(unused_variables)]
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn run_speed_limit_train_sims(
    mut speed_limit_train_sims: SpeedLimitTrainSimVec,
    network: Vec<Link>,
    train_consist_plan_py: PyDataFrame,
    loco_pool_py: PyDataFrame,
    refuel_facilities_py: PyDataFrame,
    timed_paths: Vec<Vec<LinkIdxTime>>,
) -> anyhow::Result<(SpeedLimitTrainSimVec, PyDataFrame)> {
    let train_consist_plan: DataFrame = train_consist_plan_py.into();
    let mut loco_pool: DataFrame = loco_pool_py.into();
    let refuel_facilities: DataFrame = refuel_facilities_py.into();

    loco_pool = loco_pool
        .lazy()
        .with_columns(vec![
            lit(f64::INFINITY).alias("Ready_Time_Min").to_owned(),
            lit(f64::INFINITY).alias("Ready_Time_Est").to_owned(),
            lit("Ready").alias("Status").to_owned(),
            col("SOC_Max_J").alias("SOC_J").to_owned(),
        ])
        .collect()
        .unwrap();

    let mut arrival_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![
            col("Actual Arrival Time(hr)"),
            col("Locomotive ID"),
            col("Destination ID"),
            col("TrainSimVec Index"),
        ])
        .sort_by_exprs(
            vec![col("Actual Arrival Time(hr)"), col("Locomotive ID")],
            vec![false, false],
            false,
        )
        .collect()
        .unwrap();

    let departure_times = train_consist_plan
        .clone()
        .lazy()
        .select(vec![col("Actual Departure Time(hr)"), col("Locomotive ID")])
        .sort_by_exprs(
            vec![col("Locomotive ID"), col("Actual Departure Time(hr)")],
            vec![false, false],
            false,
        )
        .collect()
        .unwrap();

    let mut charge_sessions = DataFrame::default();

    let active_loco_statuses =
        Series::from_iter(vec!["Refueling".to_string(), "Dispatched".to_string()]);
    let mut current_time: f64 = (&arrival_times)
        .column("Actual Arrival Time(hr)")?
        .min()
        .unwrap();
    let mut done = false;
    while !done {
        let arrivals_mask = (&arrival_times)
            .column("Actual Arrival Time(hr)")?
            .equal(current_time)?;
        let arrivals = arrival_times.clone().filter(&arrivals_mask)?;
        let arrivals_merged =
            loco_pool
                .clone()
                .left_join(&arrivals, &["Locomotive_ID"], &["Locomotive ID"])?;
        let arrival_locations = arrivals_merged.column("Destination ID")?;
        if arrivals.height() > 0 {
            let arrival_ids = arrivals.column("Locomotive ID")?;
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit("Queued"))
                        .otherwise(col("Status"))
                        .alias("Status"),
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit(current_time))
                        .otherwise(col("Ready_Time_Est"))
                        .alias("Ready_Time_Est"),
                    when(col("Locomotive_ID").is_in(lit(arrival_ids.clone())))
                        .then(lit(arrival_locations.clone()))
                        .otherwise(col("Node"))
                        .alias("Node"),
                ])
                .drop_columns(vec!["Charger_J_Per_Hr", "Queue_Size"])
                .join(
                    refuel_facilities.clone().lazy(),
                    [col("Node"), col("Type")],
                    [col("Node"), col("Type")],
                    JoinType::Left,
                )
                .sort("Ready_Time_Est", SortOptions::default())
                .collect()
                .unwrap();

            let indices = arrivals.column("TrainSimVec Index")?.u32()?.unique()?;
            for index in indices.into_iter() {
                let idx = index.unwrap() as usize;
                let sim = &mut speed_limit_train_sims.0[idx];
                let _ = sim
                    .walk_timed_path(&network, &timed_paths[idx])
                    .map_err(|err| err.context(format!("train sim idx: {}", idx)));

                let new_soc_vec: Vec<f64> = sim
                    .loco_con
                    .loco_vec
                    .iter()
                    .map(|loco| match loco.loco_type {
                        LocoType::BatteryElectricLoco(_) => {
                            (loco.reversible_energy_storage().unwrap().state.soc
                                * loco.reversible_energy_storage().unwrap().energy_capacity)
                                .get::<si::joule>()
                        }
                        _ => f64::ZERO,
                    })
                    .collect();
                let mut all_current_socs: Vec<f64> = loco_pool
                    .column("SOC_J")?
                    .f64()?
                    .into_no_null_iter()
                    .collect();
                let idx_mask = (&arrival_times)
                    .column("TrainSimVec Index")?
                    .equal(idx as u32)?;
                let arrival_locos = arrival_times.filter(&idx_mask)?;
                let arrival_loco_ids = arrival_locos.column("Locomotive ID")?.u32()?;
                let arrival_loco_mask = loco_pool
                    .column("Locomotive_ID")?
                    .is_in(&(arrival_loco_ids.clone().into_series()))
                    .unwrap();
                // Get the indices of true values in the boolean ChunkedArray
                let arrival_loco_indices: Vec<usize> = arrival_loco_mask
                    .into_iter()
                    .enumerate()
                    .filter(|(_, val)| val.unwrap_or_default())
                    .map(|(i, _)| i)
                    .collect();

                for (index, value) in arrival_loco_indices.iter().zip(new_soc_vec) {
                    all_current_socs[*index] = value;
                }
                loco_pool = loco_pool
                    .lazy()
                    .with_columns(vec![when(lit(arrival_loco_mask.into_series()))
                        .then(lit(Series::new("SOC_J", all_current_socs)))
                        .otherwise(col("SOC_J"))
                        .alias("SOC_J")])
                    .collect()
                    .unwrap();
            }
        }

        let refueling_mask = (&loco_pool).column("Status")?.equal("Refueling")?;
        let refueling_finished_mask =
            refueling_mask & (&loco_pool).column("Ready_Time_Est")?.equal(current_time)?;
        let refueling_finished = loco_pool.clone().filter(&refueling_finished_mask)?;
        if refueling_finished_mask.sum().unwrap_or_default() > 0 {
            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![when(lit(refueling_finished_mask.into_series()))
                    .then(lit("Ready"))
                    .otherwise(col("Status"))
                    .alias("Status")])
                .collect()
                .unwrap();
        }

        if (arrivals.height() > 0) || (refueling_finished.height() > 0) {
            // update queue
            let place_in_queue = loco_pool
                .clone()
                .lazy()
                .select(&[((col("Status")
                    .eq(lit("Refueling"))
                    .sum()
                    .over(["Node", "Type"]))
                    + (col("Status")
                        .eq(lit("Queued"))
                        .cumsum(false)
                        .over(["Node", "Type"])))
                .alias("place_in_queue")])
                .collect()?
                .column("place_in_queue")?
                .clone()
                .into_series();
            let future_times_mask = (&departure_times)
                .column("Actual Departure Time(hr)")?
                .f64()?
                .gt(current_time);

            let next_departure_time = departure_times
                .clone()
                .lazy()
                .filter(col("Actual Departure Time(hr)").gt(lit(current_time)))
                .groupby(&["Locomotive ID"])
                .agg([col("Actual Departure Time(hr)").min()])
                .collect()
                .unwrap();

            let departures_merged = loco_pool.clone().left_join(
                &next_departure_time,
                &["Locomotive_ID"],
                &["Locomotive ID"],
            )?;
            let departure_times = departures_merged
                .column("Actual Departure Time(hr)")?
                .f64()?;

            let charge_end_time_ideal = loco_pool
                .clone()
                .lazy()
                .select(&[(lit(current_time)
                    + (col("SOC_Max_J") - col("SOC_J")) / col("Charger_J_Per_Hr"))
                .alias("Charge_End_Time")])
                .collect()?
                .column("Charge_End_Time")?
                .clone()
                .into_series();

            let charge_end_time: Vec<f64> = departure_times
                .into_iter()
                .zip(charge_end_time_ideal.f64()?.into_iter())
                .map(|(b, v)| b.unwrap_or(f64::INFINITY).min(v.unwrap_or(f64::INFINITY)))
                .collect::<Vec<_>>();

            let mut charge_duration: Vec<f64> = charge_end_time.clone();
            for element in charge_duration.iter_mut() {
                *element -= current_time;
            }
            let charge_duration_series = Series::new("charge_duration", charge_duration);
            let charge_end_series = Series::new("charge_end_time", charge_end_time);

            loco_pool = loco_pool
                .lazy()
                .with_columns(vec![
                    lit(place_in_queue),
                    lit(charge_duration_series),
                    lit(charge_end_series),
                ])
                .collect()
                .unwrap();

            // store the filter as an Expr
            let charge_starting = loco_pool
                .clone()
                .lazy()
                .filter(
                    col("Status")
                        .eq(lit("Queued"))
                        .and(col("Queue_Size").gt_eq(col("place_in_queue"))),
                )
                .collect()
                .unwrap();

            // Make a Polars DataFrame of:
            // Node from filtered loco_pool
            // Type from filtered loco_pool
            // Locomotive_ID from filtered loco_pool
            // Charge_J_Per_Hr from filtered loco_pool
            // charge_duration_series * Charger_J_Per_Hr (total J of charging)
            // current_time (Charge_Start_Time)
            // Ready_Time_Est from filtered loco_pool (Charge_End_Time)
            //
            let these_charge_sessions = df![
                "Node" => charge_starting.column("Node").unwrap(),
                "Type" => charge_starting.column("Type").unwrap(),
                "Locomotive_ID" => charge_starting.column("Locomotive_ID").unwrap(),
                "Charger_J_Per_Hr" => charge_starting.column("Charger_J_Per_Hr").unwrap(),
                "Charge_Total_J" => charge_starting.column("charge_duration").unwrap() *
                    charge_starting.column("Charger_J_Per_Hr").unwrap(),
                "Charge_Start_Time_Hr" => charge_starting.column("charge_end_time").unwrap() -
                    charge_starting.column("charge_duration").unwrap(),
                "Charge_End_Time_Hr" => charge_starting.column("charge_end_time").unwrap()
            ]?;
            charge_sessions.vstack_mut(&these_charge_sessions)?;
            // set finishedCharging times to min(max soc OR departure time)
            loco_pool = loco_pool
                .clone()
                .lazy()
                .with_columns(vec![
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Queue_Size").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("SOC_J") + col("charge_duration") * col("Charger_J_Per_Hr"))
                    .otherwise(col("SOC_J"))
                    .alias("SOC_J"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Queue_Size").gt_eq(col("place_in_queue"))),
                    )
                    .then(col("charge_end_time"))
                    .otherwise(col("Ready_Time_Est"))
                    .alias("Ready_Time_Est"),
                    when(
                        col("Status")
                            .eq(lit("Queued"))
                            .and(col("Queue_Size").gt_eq(col("place_in_queue"))),
                    )
                    .then(lit("Refueling"))
                    .otherwise(col("Status"))
                    .alias("Status"),
                ])
                .collect()
                .unwrap();

            loco_pool = loco_pool.drop("place_in_queue")?;
            loco_pool = loco_pool.drop("charge_duration")?;
            loco_pool = loco_pool.drop("charge_end_time")?;
        }

        let active_loco_ready_times = loco_pool
            .clone()
            .lazy()
            .filter(col("Status").is_in(lit(active_loco_statuses.clone())))
            .select(vec![col("Ready_Time_Est")])
            .collect()?
            .column("Ready_Time_Est")?
            .clone()
            .into_series();
        arrival_times = arrival_times
            .lazy()
            .filter(col("Actual Arrival Time(hr)").gt(current_time))
            .collect()?;
        let arrival_times_remaining = arrival_times
            .clone()
            .lazy()
            .select(vec![
                col("Actual Arrival Time(hr)").alias("Actual Arrival Time(hr)")
            ])
            .collect()?
            .column("Actual Arrival Time(hr)")?
            .clone()
            .into_series();
        if (arrival_times_remaining.len() == 0) & (active_loco_ready_times.len() == 0) {
            done = true;
        } else {
            let min1 = active_loco_ready_times
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            let min2 = arrival_times_remaining
                .f64()?
                .min()
                .unwrap_or(f64::INFINITY);
            current_time = f64::min(min1, min2);
        }
    }

    Ok((speed_limit_train_sims, PyDataFrame(charge_sessions)))
}

// This MUST remain a unit struct to trigger correct tolist() behavior
#[altrios_api(
    #[pyo3(name = "get_energy_fuel_joules")]
    pub fn get_energy_fuel_py(&self, annualize: bool) -> f64 {
        self.get_energy_fuel(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_net_energy_res_joules")]
    pub fn get_net_energy_res_py(&self, annualize: bool) -> f64 {
        self.get_net_energy_res(annualize).get::<si::joule>()
    }

    #[pyo3(name = "get_megagram_kilometers")]
    pub fn get_megagram_kilometers_py(&self, annualize: bool) -> f64 {
        self.get_megagram_kilometers(annualize)
    }

    #[pyo3(name = "set_save_interval")]
    pub fn set_save_interval_py(&mut self, save_interval: Option<usize>) {
        self.set_save_interval(save_interval);
    }
)]
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct SpeedLimitTrainSimVec(pub Vec<SpeedLimitTrainSim>);

impl SpeedLimitTrainSimVec {
    pub fn get_energy_fuel(&self, annualize: bool) -> si::Energy {
        self.0
            .iter()
            .map(|sim| sim.get_energy_fuel(annualize))
            .sum()
    }

    pub fn get_net_energy_res(&self, annualize: bool) -> si::Energy {
        self.0
            .iter()
            .map(|sim| sim.get_net_energy_res(annualize))
            .sum()
    }

    pub fn get_megagram_kilometers(&self, annualize: bool) -> f64 {
        self.0
            .iter()
            .map(|sim| sim.get_megagram_kilometers(annualize))
            .sum()
    }

    pub fn set_save_interval(&mut self, save_interval: Option<usize>) {
        self.0
            .iter_mut()
            .for_each(|slts| slts.set_save_interval(save_interval));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::rail_vehicle::import_rail_vehicles;

    #[test]
    fn test_make_train_params() {
        let train_summaries = vec![TrainSummary::valid()];
        let rail_vehicle_map =
            import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap();
        for train_summary in train_summaries {
            train_summary.make_train_params(&rail_vehicle_map);
        }
    }

    #[test]
    fn test_make_speed_limit_train_sims() {
        let train_summaries = vec![TrainSummary::valid()];
        let mut rvm_file = project_root::get_project_root().unwrap();
        rvm_file.push("altrios-core/src/train/test_rail_vehicles.csv");
        let rail_vehicle_map = match import_rail_vehicles(&rvm_file) {
            Ok(rvm) => rvm,
            Err(_) => {
                import_rail_vehicles(Path::new("./src/train/test_rail_vehicles.csv")).unwrap()
            }
        };
        let mut location_map = LocationMap::new();
        location_map.insert("dummy".to_string(), vec![]);

        let consist = Consist::default();

        for train_summary in train_summaries {
            let tsb = TrainSimBuilder::new(
                "".to_string(),
                "dummy".to_string(),
                "dummy".to_string(),
                train_summary,
                consist.clone(),
                None,
            );
            tsb.make_speed_limit_train_sim(&rail_vehicle_map, &location_map, None, None, None)
                .unwrap();
        }
    }
}
