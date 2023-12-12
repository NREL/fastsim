"""
Module for pre-processing data from fueleconomy.gov and EPA vehicle testing that is used for "vehicle import" functionality.
Vehicle import allows FASTSim to import vehicles by specifying make, model, and year. 
See fastsim.demos.vehicle_import_demo for usage.

In order to run this pre-processing script, the data from the sources below should be placed in the "input_dir" (see the run function).

fueleconomy.gov data:
https://www.fueleconomy.gov/feg/download.shtml
- vehicles.csv
- emissions.csv

EPA Test data:
https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
- the data for emissions by year; e.g., 20tstcar-2021-03-02.xlsx
- note: there are multiple formats in use
"""
from pathlib import Path
import csv
import re
import shutil
import zipfile
import argparse

import openpyxl


def process_csv(path: Path, fn):
    """
    """
    header = None
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        header_row = True
        for row in reader:
            if header_row:
                header = [h.strip() for h in row]
                header_row = False
            else:
                data = {h:v for (h, v) in zip(header, row)}
                fn(data, row)
    return header


def write_csvs_for_each_year(output_data_dir, basename, rows_by_year, header):
    """
    """
    for year in rows_by_year.keys():
        output_path = output_data_dir / (str(year) + "-" + basename)
        with open(output_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for row in rows_by_year[year]:
                writer.writerow(row)


def sort_fueleconomygov_data_by_year(input_data_dir: Path, output_data_dir: Path):
    """
    Opens up the vehicles.csv and emissions.csv and breaks them up to be by year and saves them again.
    """
    veh_data_path = input_data_dir / 'vehicles.csv'
    emissions_data_path = input_data_dir / 'emissions.csv'
    if not output_data_dir.exists():
        output_data_dir.mkdir(parents=True)
    vehs_by_year = {}
    veh_ids_by_year = {}
    emissions_by_year = {}
    def process_vehicles(data, row):
        year = int(data["year"])
        id = int(data["id"])
        if year not in vehs_by_year:
            vehs_by_year[year] = []
        vehs_by_year[year].append(row)
        if year not in veh_ids_by_year:
            veh_ids_by_year[year] = set()
        veh_ids_by_year[year].add(id)
    def process_emissions(data, row):
        id = int(data["id"])
        for (year, ids) in veh_ids_by_year.items():
            if id in ids:
                if year not in emissions_by_year:
                    emissions_by_year[year] = []
                emissions_by_year[year].append(row)
    veh_header = process_csv(veh_data_path, process_vehicles)
    emiss_header = process_csv(emissions_data_path, process_emissions)
    write_csvs_for_each_year(output_data_dir, basename="vehicles.csv", rows_by_year=vehs_by_year, header=veh_header)
    write_csvs_for_each_year(output_data_dir, basename="emissions.csv", rows_by_year=emissions_by_year, header=emiss_header)


def xlsx_to_csv(xlsx_path, csv_path):
    """
    """
    workbook = openpyxl.load_workbook(xlsx_path)
    worksheet = workbook.active
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in worksheet.rows:
            writer.writerow([cell.value for cell in row])


EPA_TEST_FILE_RX = re.compile(r"^(\d\d)[^\d].*$")


def process_epa_test_data(input_dir, output_dir):
    """
    """
    for path in Path(input_dir).iterdir():
        if not path.is_file():
            continue
        m = EPA_TEST_FILE_RX.search(path.stem)
        if m is not None:
            year2 = int(m.group(1))
            year = year2 + 1900 if year2 > 70 else year2 + 2000
            new_path = output_dir / (str(year) + "-testcar.csv")
            if path.suffix == ".csv":
                shutil.copyfile(path, new_path)
            elif path.suffix == ".xlsx":
                xlsx_to_csv(path, new_path)
            else:
                print(f"Skipping file {path.name} ({path.suffix})")


YEAR_PREFIX_CSV = re.compile(r"(\d\d\d\d)-.*\.csv")


def create_zip_archives_by_year(files_dir, zip_dir):
    """
    Takes files in the files_dir that start with \d\d\d\d-*.csv
    and adds them to a \d\d\d\d.zip in the zip_dir
    """
    if not zip_dir.exists():
        zip_dir.mkdir(parents=True)
    files_by_year = {}
    for path in Path(files_dir).iterdir():
        if not path.is_file():
            continue
        m = YEAR_PREFIX_CSV.match(path.name)
        if m is not None:
            year = int(m.group(1))
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(path)
    for (year, file_paths) in files_by_year.items():
        archive_path = zip_dir / f"{year}.zip"
        with zipfile.ZipFile(archive_path, mode='w') as zf:
            for fp in file_paths:
                zf.write(fp, arcname=fp.name)


def parseargs(default_input_dir, default_output_dir, default_zip_dir):
    parser = argparse.ArgumentParser(description="Programmatically copy Rust docstrings into .pyi file")
    parser.add_argument(
        "input_dir",
        type=str,
        default=default_input_dir,
        help="Input directory containing vehicles.csv and emissions.csv from fueleconomy.gov as well as data files from EPA",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default=default_output_dir,
        help="Output directory for interum CSV files",
    )
    parser.add_argument(
        "zip_dir",
        type=str,
        default=default_zip_dir,
        help="The directory holding all of the data zipped by model year",
    )
    return parser.parse_args()


def get_default_paths() -> dict:
    parent_dir = Path(__file__).parent.resolve()
    input_dir = (parent_dir / "incoming").resolve()
    output_dir = (parent_dir / "output").resolve()
    zip_dir = (parent_dir / "output_zips").resolve()
    return {
        "parent_dir": parent_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "zip_dir": zip_dir,
    }


def run(input_dir=None, output_dir=None, zip_dir=None):
    default_paths = get_default_paths()
    input_dir = input_dir or default_paths["input_dir"]
    output_dir = output_dir or default_paths["output_dir"]
    zip_dir = zip_dir or default_paths["zip_dir"]
    sort_fueleconomygov_data_by_year(input_dir, output_dir)
    process_epa_test_data(input_dir, output_dir)
    create_zip_archives_by_year(output_dir, zip_dir)


if __name__ == "__main__":
    default_paths = get_default_paths()
    args = parseargs(default_paths["input_dir"], default_paths["output_dir"], default_paths["zip_dir"])
    print(f"Input Directory     : {args.input_dir}")
    print(f"Output Directory    : {args.output_dir}")
    print(f"Output Zip Directory: {args.zip_dir}")
    run(args.input_dir, args.output_dir, args.zip_dir)
    print("Done!")