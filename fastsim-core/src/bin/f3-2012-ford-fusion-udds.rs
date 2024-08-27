/// Binary for profiling the 2012 Ford Fusion in fastsim-3 with `cargo build
/// --profile profiling  --bin f3-2012-ford-fusion-udds`.  To run this for
/// profiling, install [samply] (https://github.com/mstange/samply) and then run
/// `samply record ./target/profiling/f3-2012-ford-fusion-udds`
use fastsim_core::{prelude::*, traits::SerdeAPI};

fn main() {
    run_sd_2012_ford_fusion_udds();
}

fn run_sd_2012_ford_fusion_udds() {
    let veh = Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
    let cyc = Cycle::from_resource("udds.csv", false).unwrap();
    let mut sd = SimDrive::new(veh, cyc, None);
    sd.walk().unwrap();
}

#[cfg(test)]
#[test]
fn test_run_sd_2012_ford_fusion_udds() {
    run_sd_2012_ford_fusion_udds();
}
