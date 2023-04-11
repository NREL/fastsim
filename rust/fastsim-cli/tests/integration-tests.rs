use std::{path::Path, str::FromStr};
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
    // compare against expected value for mpg
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("33.8"));
    Ok(())
}

#[test]
fn test_that_adopt_hd_option_works_as_expected() -> Result<(), Box<dyn std::error::Error>> {
    let expected_results = vec![
        ("adoptstring.json",  "0.245"), // 0.245 kWh/mile
        ("adoptstring2.json", "7.906"), // 7.906 mpgge
        ("adoptstring3.json", "6.882"), // 6.882 mpgge
    ];
    let mut cmd = Command::cargo_bin("fastsim-cli")?;
    for (veh_file, expected_result) in expected_results.iter() {
        let mut adopt_veh_file = project_root::get_project_root().unwrap();
        let mut adopt_str_path = String::from("../rust/fastsim-cli/tests/assets/");
        adopt_str_path.push_str(veh_file);
        adopt_veh_file.push(Path::new(&adopt_str_path));
        adopt_veh_file = adopt_veh_file.canonicalize().unwrap();
        assert!(adopt_veh_file.exists());

        cmd.args([
            "--adopt-hd",
            "true",
            "--veh-file",
            adopt_veh_file.to_str().unwrap(),
        ]);
        // compare against expected value for mpg
        let expected_mpg = String::from_str(expected_result).unwrap();
        cmd.assert()
            .success()
            .stdout(predicate::str::contains(expected_mpg));
    }
    Ok(())
}
