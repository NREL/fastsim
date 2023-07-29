use std::path::Path;
use clap::Parser;
use std::fs;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    input_file_path: String,
    fegov_db_path: String,
    epatest_db_path: String,
    output_dir_path: String,
}

fn run_import(args: &Args) {
    // confirm paths exist for all input files
    let input_file_path = Path::new(&args.input_file_path);
    if !input_file_path.exists() {
        panic!("input file path does not exist: {}", args.input_file_path);
    }
    let fegov_db_path = Path::new(&args.fegov_db_path);
    if !fegov_db_path.exists() {
        panic!("fueleconomy.gov data does not exist at: {}", args.fegov_db_path);
    }
    let epatest_db_path = Path::new(&args.epatest_db_path);
    if !epatest_db_path.exists() {
        panic!("EPA test data does not exist at: {}", args.epatest_db_path);
    }
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
}

pub fn main() {
    let args = Args::parse();
    run_import(&args);
}