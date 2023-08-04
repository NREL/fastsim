use std::path::Path;
use clap::Parser;
use std::fs;
use fastsim_core::vehicle_utils::import_and_save_all_vehicles_from_file;


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
    data_dir_path: Option<String>,
    output_dir_path: String,
}

fn run_import(args: &Args) {
    // confirm paths exist for all input files
    let input_file_path = Path::new(&args.input_file_path);
    if !input_file_path.exists() {
        panic!("input file path does not exist: {}", args.input_file_path);
    }
    let data_dir_path = match &args.data_dir_path {
        Some(data_dir_str) => {
            let dd_path = Path::new(data_dir_str);
            if !dd_path.exists() {
                panic!("No data directory at {}", data_dir_str);
            }
            dd_path
        },
        None => {
            // TODO: use directories crate to find the fastsim_cache directory
            // for now, just hard-code it.
            let fastsim_cache_dir = "C:/Users/mokeefe/AppData/Roaming/fastsim_cache";
            let dd_path = Path::new(fastsim_cache_dir);
            if !dd_path.exists() {
                let result = fs::create_dir_all(dd_path);
                if result.is_err() {
                    panic!("Could not create cache directory at {}; {:?}", fastsim_cache_dir, result.err());
                }
            }
            dd_path
        }
    };
    let output_dir_path = Path::new(&args.output_dir_path);
    if !output_dir_path.exists() {
        // create output directory if it doesn't exist
        let r = fs::create_dir(output_dir_path);
        if r.is_err() {
            panic!("Could not create directory {}", args.output_dir_path);
        }
    } else if !output_dir_path.is_dir() {
        panic!("Output dir exists but is not a directory: {}", args.output_dir_path);
    }
    let result =
        import_and_save_all_vehicles_from_file(
            input_file_path,
            data_dir_path,
            output_dir_path);
    if result.is_err() {
        println!("Error with importing and saving all vehicles from file: {:?}", result.err());
    } else {
        println!("Successfully ran vehicle import");
    }
}

pub fn main() {
    let args = Args::parse();
    run_import(&args);
}