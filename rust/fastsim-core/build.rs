//! Ensures that files that are duplicated in Python resources folder
//! and locally in this crate are identical

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // path when running `cargo publish` in fastsim-core/
    let publish_path = "../../../../python/fastsim/resources".to_string();
    // path when building using build_and_test.sh
    let build_path = "../../python/fastsim/resources".to_string();

    let prepath = match PathBuf::from(&publish_path).exists() {
        true => publish_path,
        false => build_path,
    };

    if !PathBuf::from(&prepath).exists() {
        // no need for further checks since this indicates that it's
        // likely that python fastsim is not available and thus
        // fastsim-core is likely being compiled as a dependency
        return;
    }

    check_files_consistent(&prepath);
}

/// Checks if rust and python resource files are consistent
fn check_files_consistent(prepath: &String) {
    let truth_files = [
        format!(
            "{}/{}/vehdb/2017_Toyota_Highlander_3.5_L.yaml",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
        format!(
            "{}/{}/longparams.json",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
        format!(
            "{}/{}/cycles/udds.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
        format!(
            "{}/{}/cycles/hwfet.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
    ];

    let compare_files = [
        format!(
            "{}/resources/vehicles/2017_Toyota_Highlander_3.5_L.yaml",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
        format!(
            "{}/resources/longparams.json",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
        format!(
            "{}/resources/cycles/udds.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
        format!(
            "{}/resources/cycles/hwfet.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
    ];

    for (tf, cf) in truth_files.iter().zip(compare_files) {
        let tfc = fs::read_to_string(tf).unwrap_or_else(|_| panic!("{tf} does not exist."));

        let cfc = fs::read_to_string(&cf).unwrap_or_else(|_| panic!("{cf} does not exist."));

        if tfc != cfc {
            panic!("Reference file {tf} does not match file being compared: {cf}.  Copy {tf} to {cf} to fix this.")
        }
    }
}
