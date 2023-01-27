use std::path::Path;
use std::process::Command;

use assert_cmd::prelude::{CommandCargoExt, OutputAssertExt};
use predicates::prelude::predicate;

#[test]
fn test_that_cli_app_produces_result() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("fastsim-cli")?;
    let mut cyc_file = project_root::get_project_root().unwrap();
    cyc_file.push(Path::new("../fastsim/resources/cycles/udds.csv"));
    cyc_file = cyc_file.canonicalize().unwrap();
    assert!(cyc_file.exists());
    let mut veh_file = project_root::get_project_root().unwrap();
    veh_file.push(Path::new(
        "../fastsim/resources/vehdb/2012_Ford_Fusion.yaml",
    ));
    veh_file = veh_file.canonicalize().unwrap();
    assert!(veh_file.exists());

    cmd.args([
        "--cyc-file",
        cyc_file.to_str().unwrap(),
        "--veh-file",
        veh_file.to_str().unwrap(),
    ]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("33.8"));

    Ok(())
}
