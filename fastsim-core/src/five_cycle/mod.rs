use super::imports::*;
use super::prelude::*;

use log;
use log::{Level, Metadata, Record};
use log::{LevelFilter, SetLoggerError};

static LOGGER: SimpleLogger = SimpleLogger;

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

pub fn init_logger(level: LevelFilter) -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(level))
}

pub fn pct_error(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        f64::INFINITY
    } else {
        100. * (actual - expected) / expected
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "five_cycle")]
#[cfg_attr(
    feature = "pyo3",
    pyo3(signature = (
        veh, logging=false))
)]
pub fn five_cycle(veh: &Vehicle, logging: bool) -> (f64, f64, f64) {
    if logging {
        init_logger(LevelFilter::Info).unwrap();
    }

    // Pacifica FE targets
    // https://fueleconomy.gov/feg/Find.do?action=sbs&id=49434
    // label city FE
    let target_city_fe = 18.9954;
    // label highway FE
    let target_highway_fe = 28.1266;
    // label combined FE
    let target_combined_fe = 22.2452;

    log::debug!("HFET");
    let hfet_fe = hfet(veh.clone());
    log::info!("hfet_fe: {} MPG", hfet_fe);
    log::debug!("US06");
    let (us06_city_fe, us06_highway_fe) = us06(veh.clone());
    log::info!("us06_city_fe: {} MPG", us06_city_fe);
    log::info!("us06_highway_fe: {} MPG", us06_highway_fe);
    log::debug!("SC03");
    let sc03_fe = sc03(veh.clone());
    log::info!("sc03_fe: {} MPG", sc03_fe);
    log::debug!("FTP Cold");
    let (bag_1_fe_20, bag_2_fe_20, bag_3_fe_20) = ftp(
        veh.clone(),
        (-6.66666667 + uc::CELSIUS_TO_KELVIN) * uc::KELVIN,
    );
    log::info!("bag_1_fe_20: {} MPG", bag_1_fe_20);
    log::info!("bag_2_fe_20: {} MPG", bag_2_fe_20);
    log::info!("bag_3_fe_20: {} MPG", bag_3_fe_20);
    log::debug!("FTP");
    let (bag_1_fe_75, bag_2_fe_75, bag_3_fe_75) =
        ftp(veh.clone(), (23.8889 + uc::CELSIUS_TO_KELVIN) * uc::KELVIN);
    log::info!("bag_1_fe_75: {} MPG", bag_1_fe_75);
    log::info!("bag_2_fe_75: {} MPG", bag_2_fe_75);
    log::info!("bag_3_fe_75: {} MPG", bag_3_fe_75);

    // City label FE
    // https://www.law.cornell.edu/cfr/text/40/600.114-12#a
    let running_fc = 0.82 * (0.48 / bag_2_fe_75 + 0.41 / bag_3_fe_75 + 0.11 / us06_city_fe)
        + 0.18 * (0.5 / bag_2_fe_20 + 0.5 / bag_3_fe_20)
        + 0.133 * 1.083 * (1. / sc03_fe - (0.61 / bag_3_fe_75 + 0.39 / bag_2_fe_75));
    let start_fuel_20 = 3.6 * (1. / bag_1_fe_20 + 1. / bag_3_fe_20);
    let start_fuel_75 = 3.6 * (1. / bag_1_fe_75 + 1. / bag_3_fe_75);
    let start_fc = 0.33 * (0.76 * start_fuel_75 + 0.24 * start_fuel_20) / 4.1;
    let city_fe = 0.905 / (start_fc + running_fc);
    log::info!("city_fe: {} MPG", city_fe);

    // Highway label FE
    // https://www.law.cornell.edu/cfr/text/40/600.114-12#b
    let running_fc = 1.007 * (0.79 / us06_highway_fe + 0.21 / hfet_fe)
        + 0.133 * 0.377 * (1. / sc03_fe - (0.61 / bag_3_fe_75 + 0.39 / bag_2_fe_75));
    let start_fuel_20 = 3.6 * (1. / bag_1_fe_20 + 1. / bag_3_fe_20);
    let start_fuel_75 = 3.6 * (1. / bag_1_fe_75 + 1. / bag_3_fe_75);
    let start_fc = 0.33 * (0.76 * start_fuel_75 + 0.24 * start_fuel_20) / 60.;
    let highway_fe = 0.905 / (start_fc + running_fc);
    log::info!("highway_fe: {} MPG", highway_fe);

    let combined_fe = 1.0 / (0.55 / city_fe + 0.45 / highway_fe);
    log::info!("combined_fe: {} MPG", combined_fe);

    let city_pct_error = pct_error(1. / city_fe, 1. / target_city_fe);
    let highway_pct_error = pct_error(1. / highway_fe, 1. / target_highway_fe);
    let combined_pct_error = pct_error(1. / combined_fe, 1. / target_combined_fe);

    log::info!("Five cycle results:");
    log::info!(
        "  City:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        city_fe,
        1. / city_fe,
        target_city_fe,
        1. / target_city_fe,
        city_pct_error
    );
    log::info!(
        "  Highway:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        highway_fe,
        1. / highway_fe,
        target_highway_fe,
        1. / target_highway_fe,
        highway_pct_error
    );
    log::info!(
        "  Combined:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        combined_fe,
        1. / combined_fe,
        target_combined_fe,
        1. / target_combined_fe,
        combined_pct_error
    );

    let mut veh_no_thrml =
        Vehicle::from_resource("2026_Chrysler_Pacifica_Select_DFCO_StopStart.yaml", false).unwrap();
    let (sdlabel_results, _) =
        crate::simdrivelabel::get_label_fe(&mut veh_no_thrml, None, false, None, None, false)
            .unwrap();

    let city_pct_error = pct_error(1. / sdlabel_results.adj_udds_mpgge, 1. / target_city_fe);
    let highway_pct_error = pct_error(1. / sdlabel_results.adj_hwy_mpgge, 1. / target_highway_fe);
    let combined_pct_error =
        pct_error(1. / sdlabel_results.adj_comb_mpgge, 1. / target_combined_fe);

    log::info!("Simdrivelabel results:");
    log::info!(
        "  City:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        sdlabel_results.adj_udds_mpgge,
        1. / sdlabel_results.adj_udds_mpgge,
        target_city_fe,
        1. / target_city_fe,
        city_pct_error
    );
    log::info!(
        "  Highway:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        sdlabel_results.adj_hwy_mpgge,
        1. / sdlabel_results.adj_hwy_mpgge,
        target_highway_fe,
        1. / target_highway_fe,
        highway_pct_error
    );
    log::info!(
        "  Combined:\n    sim:  {:.2} MPG ({:.4} gal/mi)\n    meas: {:.2} MPG ({:.4} gal/mi)\n    error: {:+.2}%",
        sdlabel_results.adj_comb_mpgge,
        1. / sdlabel_results.adj_comb_mpgge,
        target_combined_fe,
        1. / target_combined_fe,
        combined_pct_error
    );

    // TODO:
    // - [x] merge in start-stop and DFCO
    // - [x] run simdrivelabel on base (non-thermal) model
    // - [ ] manual tweaking of thermal parameters - likely increase engine thermal mass

    (city_fe, highway_fe, combined_fe)
}

pub fn hfet(mut veh: Vehicle) -> f64 {
    let ambient_temp = (25. + uc::CELSIUS_TO_KELVIN) * uc::KELVIN;
    // Set initial temperatures to ambient
    set_cabin_temperature(&mut veh, ambient_temp).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut veh, ambient_temp).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut veh, ambient_temp).unwrap();
    }
    let hfet = Cycle::from_resource("hwfet.csv", false).unwrap();

    // Set up first simulation (cold start)
    let mut sd0 = SimDrive::new(veh.clone(), hfet.clone(), None);
    log::debug!("Initial temperatures:");
    log_temperatures(&sd0.veh);
    sd0.walk().unwrap(); // simulate
    log::debug!("Temperatures after preconditioning:");
    log_temperatures(&sd0.veh);

    // Set up idle simulation (15 seconds idle)
    let idle = Cycle::try_from(crate::drive_cycle::CycleBuilder {
        name: String::from("idle"),
        time: (0..15)
            .map(|t| (t as f64) * uc::S)
            .collect::<Vec<si::Time>>(),
        speed: vec![0.0 * uc::MPS; 15],
    })
    .unwrap();
    let mut sd_idle = SimDrive::new(veh.clone(), idle, None);
    // Set temperatures
    set_cabin_temperature(&mut sd_idle.veh, get_cabin_temperature(&sd0.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd_idle.veh, get_fc_temperature(&sd0.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd_idle.veh, get_res_temperature(&sd0.veh).unwrap()).unwrap();
    }
    sd_idle.walk().unwrap(); // simulate
    log::debug!("Post-idle results:");
    log_temperatures(&sd_idle.veh);
    output_energy(&sd_idle.veh);

    // Set up second simulation
    let mut sd1 = SimDrive::new(veh.clone(), hfet, None);
    // Set temperatures
    set_cabin_temperature(&mut sd1.veh, get_cabin_temperature(&sd_idle.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd1.veh, get_fc_temperature(&sd_idle.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd1.veh, get_res_temperature(&sd_idle.veh).unwrap()).unwrap();
    }
    sd1.walk().unwrap(); // simulate
    log::debug!("Final results:");
    log_temperatures(&sd1.veh);
    output_energy(&sd1.veh);

    let hfet_fe = get_mpg(&sd1.veh);

    hfet_fe
}

pub fn us06(mut veh: Vehicle) -> (f64, f64) {
    let ambient_temp = (25. + uc::CELSIUS_TO_KELVIN) * uc::KELVIN;
    // Set initial temperatures to ambient
    set_cabin_temperature(&mut veh, ambient_temp).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut veh, ambient_temp).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut veh, ambient_temp).unwrap();
    }

    let us06 = Cycle::from_resource("us06.csv", false).unwrap();
    let us06_city_1 = Cycle::from_resource("us06_city_1.csv", false).unwrap();
    let us06_highway = Cycle::from_resource("us06_highway.csv", false).unwrap();
    let us06_city_2 = Cycle::from_resource("us06_city_2.csv", false).unwrap();

    // Set up first simulation (cold start)
    let mut sd0 = SimDrive::new(veh.clone(), us06.clone(), None);
    log::debug!("Initial temperatures:");
    log_temperatures(&sd0.veh);
    sd0.walk().unwrap(); // simulate
    log::debug!("Temperatures after preconditioning:");
    log_temperatures(&sd0.veh);

    // Set up idle simulation (1 to 2 minutes idle)
    let idle = Cycle::try_from(crate::drive_cycle::CycleBuilder {
        name: String::from("idle"),
        time: (0..90)
            .map(|t| (t as f64) * uc::S)
            .collect::<Vec<si::Time>>(),
        speed: vec![0.0 * uc::MPS; 90],
    })
    .unwrap();
    let mut sd_idle = SimDrive::new(veh.clone(), idle, None);
    // Set temperatures
    set_cabin_temperature(&mut sd_idle.veh, get_cabin_temperature(&sd0.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd_idle.veh, get_fc_temperature(&sd0.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd_idle.veh, get_res_temperature(&sd0.veh).unwrap()).unwrap();
    }
    sd_idle.walk().unwrap(); // simulate
    log::debug!("Post-idle results:");
    log_temperatures(&sd_idle.veh);
    output_energy(&sd_idle.veh);

    // Set up second simulation, US06 first 'city' portion (1-130 sec)
    let mut sd1 = SimDrive::new(veh.clone(), us06_city_1, None);
    // Set temperatures
    set_cabin_temperature(&mut sd1.veh, get_cabin_temperature(&sd_idle.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd1.veh, get_fc_temperature(&sd_idle.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd1.veh, get_res_temperature(&sd_idle.veh).unwrap()).unwrap();
    }
    sd1.walk().unwrap(); // simulate
    log::debug!("Phase 1 results (city 1):");
    log_temperatures(&sd1.veh);
    output_energy(&sd1.veh);

    // Set up third simulation, US06 'highway' portion (130-495 sec)
    let mut sd2 = SimDrive::new(veh.clone(), us06_highway, None);
    // Set temperatures
    set_cabin_temperature(&mut sd2.veh, get_cabin_temperature(&sd1.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd2.veh, get_fc_temperature(&sd1.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd2.veh, get_res_temperature(&sd1.veh).unwrap()).unwrap();
    }
    sd2.walk().unwrap(); // simulate
    log::debug!("Phase 2 results (highway):");
    log_temperatures(&sd2.veh);
    output_energy(&sd2.veh);

    // Set up fourth simulation, US06 'city' portion (495-600 sec)
    let mut sd3 = SimDrive::new(veh.clone(), us06_city_2, None);
    // Set temperatures
    set_cabin_temperature(&mut sd3.veh, get_cabin_temperature(&sd2.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd3.veh, get_fc_temperature(&sd2.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd3.veh, get_res_temperature(&sd2.veh).unwrap()).unwrap();
    }
    sd3.walk().unwrap(); // simulate
    log::debug!("Phase 3 results (city 2):");
    log_temperatures(&sd3.veh);
    output_energy(&sd3.veh);

    let us06_city_distance = *sd1.veh.state.dist.get_fresh(|| format_dbg!()).unwrap()
        + *sd3.veh.state.dist.get_fresh(|| format_dbg!()).unwrap();
    let us06_city_fuel = get_fuel_used(&sd1.veh) + get_fuel_used(&sd3.veh);
    let us06_city_mpg = us06_city_distance.get::<si::mile>() / us06_city_fuel.get::<si::gallon>();

    let us06_highway_mpg = get_mpg(&sd2.veh);

    (us06_city_mpg, us06_highway_mpg)
}

pub fn sc03(mut veh: Vehicle) -> f64 {
    let ambient_temp = (35. + uc::CELSIUS_TO_KELVIN) * uc::KELVIN;
    // Set initial temperatures to ambient
    set_cabin_temperature(&mut veh, ambient_temp).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut veh, ambient_temp).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut veh, ambient_temp).unwrap();
    }

    let udds = Cycle::from_resource("udds.csv", false).unwrap();
    let sc03 = Cycle::from_resource("sc03.csv", false).unwrap();

    // First simulation (UDDS cold start)
    let mut sd0 = SimDrive::new(veh.clone(), udds, None);
    log::debug!("Initial temperatures:");
    log_temperatures(&sd0.veh);
    sd0.walk().unwrap(); // simulate
    log::debug!("Temperatures after preconditioning:");
    log_temperatures(&sd0.veh);

    // Engine-off soak (10 minutes)
    let soak = Cycle::try_from(crate::drive_cycle::CycleBuilder {
        name: String::from("soak"),
        time: (0..600)
            .map(|t| (t as f64) * uc::S)
            .collect::<Vec<si::Time>>(),
        speed: vec![0.0 * uc::MPS; 600],
    })
    .unwrap();
    let mut sd_soak = SimDrive::new(veh.clone(), soak, None);
    // Zero out aux load
    sd_soak.veh.pwr_aux_base = 0.0 * uc::W;
    // Set temperatures
    set_cabin_temperature(&mut sd_soak.veh, get_cabin_temperature(&sd0.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd_soak.veh, get_fc_temperature(&sd0.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd_soak.veh, get_res_temperature(&sd0.veh).unwrap()).unwrap();
    }
    sd_soak.walk().unwrap(); // simulate
    assert!(output_energy(&sd_soak.veh).get::<si::joule>() == 0.0);
    log::debug!("Post-soak temperatures:");
    log_temperatures(&sd_soak.veh);

    // Second simulation
    let mut sd1 = SimDrive::new(veh.clone(), sc03.clone(), None);
    // Set temperatures
    set_cabin_temperature(&mut sd1.veh, get_cabin_temperature(&sd_soak.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd1.veh, get_fc_temperature(&sd_soak.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd1.veh, get_res_temperature(&sd_soak.veh).unwrap()).unwrap();
    }
    sd1.walk().unwrap(); // simulate
    log::debug!("Final results:");
    log_temperatures(&sd1.veh);
    output_energy(&sd1.veh);

    get_mpg(&sd1.veh)
}

pub fn ftp(mut veh: Vehicle, ambient_temp: si::Temperature) -> (f64, f64, f64) {
    // Set initial temperatures to ambient
    set_cabin_temperature(&mut veh, ambient_temp).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut veh, ambient_temp).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut veh, ambient_temp).unwrap();
    }

    let mut ftp_transient = Cycle::from_resource("ftp_transient.csv", false).unwrap();
    ftp_transient.temp_amb_air = vec![ambient_temp; ftp_transient.time.len()];
    let mut ftp_stabilized = Cycle::from_resource("ftp_stabilized.csv", false).unwrap();
    ftp_stabilized.temp_amb_air = vec![ambient_temp; ftp_stabilized.time.len()];

    // First simulation (cold start transient)
    let mut sd0 = SimDrive::new(veh.clone(), ftp_transient.clone(), None);
    log::debug!("Initial temperatures:");
    log_temperatures(&sd0.veh);
    sd0.walk().unwrap(); // simulate
    log::debug!("Temperatures after cold start transient segment:");
    log_temperatures(&sd0.veh);

    // Second simulation (stabilized)
    let mut sd1 = SimDrive::new(veh.clone(), ftp_stabilized.clone(), None);
    // Set temperatures
    set_cabin_temperature(&mut sd1.veh, get_cabin_temperature(&sd0.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd1.veh, get_fc_temperature(&sd0.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd1.veh, get_res_temperature(&sd0.veh).unwrap()).unwrap();
    }
    sd1.walk().unwrap(); // simulate
    log::debug!("Temperatures after stabilized segment:");
    log_temperatures(&sd1.veh);

    // Engine-off soak (10 minutes)
    let soak = Cycle::try_from(crate::drive_cycle::CycleBuilder {
        name: String::from("soak"),
        time: (0..600)
            .map(|t| (t as f64) * uc::S)
            .collect::<Vec<si::Time>>(),
        speed: vec![0.0 * uc::MPS; 600],
    })
    .unwrap();
    let mut sd_soak = SimDrive::new(veh.clone(), soak, None);
    // Zero out aux load
    sd_soak.veh.pwr_aux_base = 0.0 * uc::W;
    // Set temperatures
    set_cabin_temperature(&mut sd_soak.veh, get_cabin_temperature(&sd1.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd_soak.veh, get_fc_temperature(&sd1.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd_soak.veh, get_res_temperature(&sd1.veh).unwrap()).unwrap();
    }
    sd_soak.walk().unwrap(); // simulate
    assert!(output_energy(&sd_soak.veh).get::<si::joule>() == 0.0);
    log::debug!("Temperatures after 10 minute soak:");
    log_temperatures(&sd_soak.veh);

    // Second simulation (transient)
    let mut sd2 = SimDrive::new(veh.clone(), ftp_transient.clone(), None);
    // Set temperatures
    set_cabin_temperature(&mut sd2.veh, get_cabin_temperature(&sd_soak.veh).unwrap()).unwrap();
    if veh.fc().is_some() {
        set_fc_temperature(&mut sd2.veh, get_fc_temperature(&sd_soak.veh).unwrap()).unwrap();
    }
    if veh.res().is_some() {
        set_res_temperature(&mut sd2.veh, get_res_temperature(&sd_soak.veh).unwrap()).unwrap();
    }
    sd2.walk().unwrap(); // simulate
    log::debug!("Temperatures after hot start transient segment:");
    log_temperatures(&sd2.veh);
    output_energy(&sd2.veh);

    let bag_1_fe_25 = get_mpg(&sd0.veh);
    let bag_2_fe_25 = get_mpg(&sd1.veh);
    let bag_3_fe_25 = get_mpg(&sd2.veh);

    (bag_1_fe_25, bag_2_fe_25, bag_3_fe_25)
}

fn get_cabin_temperature(veh: &Vehicle) -> Option<si::Temperature> {
    if let CabinOption::LumpedCabin(lc) = &veh.cabin {
        return Some(
            lc.state
                .temperature
                .get_fresh(|| format_dbg!())
                .unwrap()
                .clone(),
        );
    }
    None
}

fn get_fc_temperature(veh: &Vehicle) -> Option<si::Temperature> {
    if let Some(fc) = veh.fc() {
        if let FuelConverterThermalOption::FuelConverterThermal(thrml) = &fc.thrml {
            return Some(
                thrml
                    .state
                    .temperature
                    .get_fresh(|| format_dbg!())
                    .unwrap()
                    .clone(),
            );
        }
    }
    None
}

// fn get_fc_temperature(veh: &Vehicle) -> Option<si::Temperature> {
//     Some(
//         *veh.fc()?
//             .temperature()?
//             .get_fresh(|| format_dbg!())
//             .unwrap(),
//     )
// }

fn get_res_temperature(veh: &Vehicle) -> Option<si::Temperature> {
    if let Some(res) = veh.res() {
        if let RESThermalOption::RESLumpedThermal(thrml) = &res.thrml {
            return Some(
                thrml
                    .state
                    .temperature
                    .get_fresh(|| format_dbg!())
                    .unwrap()
                    .clone(),
            );
        }
    }
    None
}

fn set_cabin_temperature(
    veh: &mut Vehicle,
    new_temperature: si::Temperature,
) -> anyhow::Result<()> {
    match &mut veh.cabin {
        CabinOption::LumpedCabin(thrml) => {
            thrml
                .state
                .temperature
                .update_unchecked(new_temperature, || format_dbg!())?;
            thrml
                .state
                .temp_prev
                .update_unchecked(new_temperature, || format_dbg!())?;
            Ok(())
        }
        opt @ _ => bail!("Unsupported cabin thermal option {opt:?}"),
    }
}

fn set_fc_temperature(veh: &mut Vehicle, new_temperature: si::Temperature) -> anyhow::Result<()> {
    match &mut veh
        .fc_mut()
        .with_context(|| anyhow!("Vehicle has no fuel converter"))?
        .thrml
    {
        FuelConverterThermalOption::FuelConverterThermal(thrml) => {
            thrml
                .state
                .temperature
                .update_unchecked(new_temperature, || format_dbg!())?;
            Ok(())
        }
        opt @ _ => bail!("Unsupported FC thermal option {opt:?}"),
    }
}

fn set_res_temperature(veh: &mut Vehicle, new_temperature: si::Temperature) -> anyhow::Result<()> {
    match &mut veh
        .res_mut()
        .with_context(|| anyhow!("Vehicle has no RES"))?
        .thrml
    {
        RESThermalOption::RESLumpedThermal(thrml) => {
            thrml
                .state
                .temperature
                .update_unchecked(new_temperature, || format_dbg!())?;
            thrml
                .state
                .temp_prev
                .update_unchecked(new_temperature, || format_dbg!())?;
            Ok(())
        }
        opt @ _ => bail!("Unsupported RES thermal option {opt:?}"),
    }
}

// Preconditioning:
// HFET:
// 1. Simulate HFET from cold start once
// 2. Simulate 15 seconds idle, starting with temperatures from end of step 1
// 3. Simulate HFET again, starting with temperatures from end of step 2
// US06:
// 1. Simulate US06 from cold start once
// 2. Simulate 1 to 2 minutes idle, starting with temperatures from end of step 1
// 3. Simulate US06 again, starting with temperatures from end of step 2
// SC03:
// 1. Simulate UDDS from cold start once
// 2. Simulate 10 minute engine-off soak, starting with temperatures from end of step 1
// 3. Simulate SC03 again, starting with temperatures from end of step 2
// FTP:
// 1. Simulate FTP from cold start once
// 2. Simulate 10 minute engine-off soak, starting with temperatures from end of step 1
// 3. Simulate FTP again, starting with temperatures from end of step 2

fn log_temperatures(veh: &Vehicle) {
    if let Some(temp) = get_cabin_temperature(veh) {
        log::debug!(
            "  cabin temperature: {} °C",
            temp.get::<si::degree_celsius>()
        );
    }
    if let Some(temp) = get_fc_temperature(veh) {
        log::debug!(
            "  fuel converter temperature: {} °C",
            temp.get::<si::degree_celsius>()
        );
    }
    if let Some(temp) = get_res_temperature(veh) {
        log::debug!(
            "  reversible energy storage temperature: {} °C",
            temp.get::<si::degree_celsius>()
        );
    }
}

fn output_energy(veh: &Vehicle) -> si::Energy {
    let fc_energy_opt = veh
        .fc()
        .map(|fc| fc.state.energy_fuel.get_fresh(|| format_dbg!()).unwrap());
    if let Some(fc_energy) = fc_energy_opt {
        log::debug!(
            "  total expended fuel converter energy: {} kWh",
            fc_energy.get::<si::joule>() / 3.6e6
        );
    }
    let res_energy_opt = veh.res().map(|res| {
        res.state
            .energy_out_chemical
            .get_fresh(|| format_dbg!())
            .unwrap()
    });
    if let Some(res_energy) = res_energy_opt {
        log::debug!(
            "  total expended battery energy (not including charger losses): {} kWh",
            res_energy.get::<si::joule>() / 3.6e6
        );
    }
    *fc_energy_opt.unwrap_or(&(0. * uc::J)) + *res_energy_opt.unwrap_or(&(0. * uc::J))
}

fn get_fuel_used(veh: &Vehicle) -> si::Volume {
    let fc_energy_opt = veh
        .fc()
        .map(|fc| fc.state.energy_fuel.get_fresh(|| format_dbg!()).unwrap());
    let res_energy_opt = veh.res().map(|res| {
        res.state
            .energy_out_chemical
            .get_fresh(|| format_dbg!())
            .unwrap()
    });
    let energy = *fc_energy_opt.unwrap_or(&(0. * uc::J)) + *res_energy_opt.unwrap_or(&(0. * uc::J));
    energy / crate::simdrivelabel::FuelProperties::default().energy_density
}

fn get_mpg(veh: &Vehicle) -> f64 {
    // Distance
    let distance = veh.state.dist.get_fresh(|| format_dbg!()).unwrap();
    let fuel_used = get_fuel_used(veh);
    let mpg = distance.get::<si::mile>() / fuel_used.get::<si::gallon>();
    mpg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        five_cycle(&veh, true);
    }

    #[test]
    fn test_hfet() {
        init_logger(LevelFilter::Info).unwrap();
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        hfet(veh);
    }

    #[test]
    fn test_us06() {
        init_logger(LevelFilter::Info).unwrap();
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        us06(veh);
    }

    #[test]
    fn test_sc03() {
        init_logger(LevelFilter::Info).unwrap();
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        sc03(veh);
    }

    #[test]
    fn test_ftp_cold() {
        init_logger(LevelFilter::Info).unwrap();
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        ftp(veh, (-6.66666667 + uc::CELSIUS_TO_KELVIN) * uc::KELVIN);
    }

    #[test]
    fn test_ftp() {
        init_logger(LevelFilter::Info).unwrap();
        let veh = Vehicle::from_resource(
            "2026_Chrysler_Pacifica_Select_Thermal_DFCO_StopStart.yaml",
            false,
        )
        .unwrap();
        ftp(veh, (23.8889 + uc::CELSIUS_TO_KELVIN) * uc::KELVIN);
    }
}
