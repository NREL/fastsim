use anyhow::{self, Context};
use clap::Parser;
use fastsim_core::utils::create_project_subdir;
use fastsim_core::vehicle_utils::{get_default_cache_url, import_and_save_all_vehicles_from_file};
use std::fs;
use std::path::{Path, PathBuf};

// eliminate db path items; instead, assume we have a config directory on the OS which we'll get via the directories crate
// - if the config directory (fastsim_cache) doesn't exist:
//   - create the directory
//   - download the data -- single zip? multiple zips? zips by year?
//   - unzip/expand the archive
// - if the config directory does exist and a "check_for_updates" flag is true
//   - check the remote for a "latest.txt" file which has the sha256 hash of the latest zip
//   - compare that with the sha256 of the current zip; if different, re-download
// once we have this sucessfully set up, use that data directory internally and proceed with import

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    input_file_path: String,
    output_dir_path: String,
    data_dir_path: Option<String>,
    cache_url: Option<String>,
}

fn run_import(args: &Args) -> anyhow::Result<()> {
    // confirm paths exist for all input files
    let input_file_path = Path::new(&args.input_file_path);
    anyhow::ensure!(
        input_file_path.exists(),
        "input file path does not exist: {}",
        args.input_file_path
    );
    let data_dir_path = match &args.data_dir_path {
        Some(data_dir_str) => {
            let dd_path = PathBuf::from(data_dir_str);
            anyhow::ensure!(dd_path.exists(), "No data directory at {}", data_dir_str);
            dd_path
        }
        None => create_project_subdir("fe_label_data")?,
    };
    let output_dir_path = Path::new(&args.output_dir_path);
    if !output_dir_path.exists() {
        // create output directory if it doesn't exist
        fs::create_dir(output_dir_path)?;
    } else if !output_dir_path.is_dir() {
        anyhow::bail!(
            "Output dir exists but is not a directory: {}",
            args.output_dir_path
        );
    }
    let cache_url = {
        if let Some(url) = &args.cache_url {
            url.clone()
        } else {
            get_default_cache_url()
        }
    };
    import_and_save_all_vehicles_from_file(
        input_file_path,
        data_dir_path.as_path(),
        output_dir_path,
        Some(cache_url),
    )
    .with_context(|| "Error with importing and saving all vehicles from file")?;
    println!("Successfully ran vehicle import");
    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run_import(&args)
}
