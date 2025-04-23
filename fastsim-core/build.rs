use std::path::PathBuf;

fn main() {
    let bolt_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cal_and_val/thermal/f3-vehicles/2020 Chevrolet Bolt EV.yaml");
    let bolt_target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/vehicles/2020 Chevrolet Bolt EV.yaml");
    std::fs::copy(bolt_source, bolt_target).unwrap();

    let sonata_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cal_and_val/thermal/f3-vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml");
    let sonata_target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml");
    std::fs::copy(sonata_source, sonata_target).unwrap();
}
