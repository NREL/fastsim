use std::path::PathBuf;

fn main() {
    // copy calibrated models to resources folder
    let bolt_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cal_and_val/thermal/f3-vehicles/2020 Chevrolet Bolt EV.yaml");
    assert!(bolt_source.exists());
    let bolt_target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/vehicles/2020 Chevrolet Bolt EV thrml.yaml");
    println!(
        "copying {} to {}",
        bolt_source.display(),
        bolt_target.display()
    );
    std::fs::copy(bolt_source, &bolt_target).unwrap();
    assert!(bolt_target.exists());

    let sonata_source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cal_and_val/thermal/f3-vehicles/2021_Hyundai_Sonata_Hybrid_Blue.yaml");
    assert!(sonata_source.exists());
    let sonata_target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources/vehicles/2021_Hyundai_Sonata_Hybrid_Blue_thrml.yaml");
    println!(
        "copying {} to {}",
        sonata_source.display(),
        sonata_target.display()
    );
    std::fs::copy(sonata_source, &sonata_target).unwrap();
    assert!(sonata_target.exists());
}
