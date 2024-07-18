/// binary for running the 2012 Ford Fusion in fastsim-3
use fastsim_core::{prelude::*, traits::SerdeAPI};

fn main() {
    let veh = Vehicle::from_resource("2012_Ford_Fusion.yaml", false).unwrap();
    let cyc = Cycle::from_resource("udds.csv", false).unwrap();
    let mut sd = SimDrive::new(veh, cyc, None);
    sd.walk().unwrap();
}
