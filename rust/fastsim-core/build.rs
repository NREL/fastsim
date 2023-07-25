//! Ensures that files that are duplicated in Python resources folder
//! and locally in this crate are identical

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let prepath: String = "../../../../python/fastsim/resources".into();

    let thinger = PathBuf::from(format!(
        "{}/../../../../python",
        env::current_dir().unwrap().as_os_str().to_str().unwrap()
    ))
    .canonicalize()
    .unwrap();
    assert!(&thinger.exists());
    dbg!(thinger);

    let truth_files = [
        format!(
            "{}/{}/longparams.json",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
        format!(
            "{}/{}/udds.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
        format!(
            "{}/{}/hwfet.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap(),
            prepath
        ),
    ];

    let compare_files = [
        format!(
            "{}/resources/longparams.json",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
        format!(
            "{}/resources/hwfet.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
        format!(
            "{}/resources/udds.csv",
            env::current_dir().unwrap().as_os_str().to_str().unwrap()
        ),
    ];

    for (tf, cf) in truth_files.iter().zip(compare_files) {
        let tfc = fs::read_to_string(tf).unwrap_or_else(|_| panic!("{tf} does not exist."));

        let cfc = fs::read_to_string(cf.clone()).unwrap_or_else(|_| panic!("{cf} does not exist."));

        if tfc != cfc {
            panic!("Reference file {tf} does not match file being compared: {cf}.  Copy {tf} to {cf} to fix this.")
        }
    }
}
