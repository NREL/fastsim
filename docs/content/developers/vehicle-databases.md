# Vehicle Databases

## Structure

FASTSim vehicle databases,
such as the public database available at https://github.com/NatLabRockies/fastsim-vehicles,
enforce a strict organizational structure.

### Database Organizational Schema (v1)

FASTSim accepts the following organizational schema for vehicle databases:

| Level | Description | Example |
| --- | --- | --- |
| Schema version | Describes the format of subsequent organizational levels. | `v1` |
| FASTSim version | Major FASTSim version. | `fastsim-3` |
| Powertrain | Powertrain type (e.g. `conv`/`hev`/`phev`/`bev`/`fcev`) | `conv` |
| Make | Vehicle make/manufacturer name. | `ford` |
| Model | Vehicle model name and any applicable trim information. | `fusion` |
| Year | Model year or year range. | `2012` |
| Variant | Description of active modeling feature set. | `base` |
| Revision | Model revision number for updates and corrections. | `r1` |

Example full filepath:

`v1/fastsim-3/conv/ford/fusion/2012/base/r1.yaml`


### Path Segment Formatting 

Vehicle databases have strict path formatting restrictions to maintain organization, prevent duplicated directories (e.g. `Toyota` and `toyota`), and make programmatic searching/parsing simpler.

Allowed characters:

- Lowercase ASCII letters: `a-z` 
- Numbers: `0-9`
- Hyphens: `-`
- Periods: `.`

Additionally, repeated separators `.` and `-` are disallowed in path segments. Notably, this prevents `".."` from being supplied in a path segment.

When trying to construct an instance of `DatabaseSchemaV1` with invalid strings, it will fail and notify you of a proper, normalized alternative.

## Creating a FASTSim Vehicle Database

In addition to the public database, FASTSim can read from local filesystem databases and databases hosted remotely.

### Pointing to an Alternate Remote Database

Remote databases at `https://raw.githubusercontent.com/...` and the NLR GitHub enterprise equivalent `https://raw.github.nrel.gov/...`
are tested and confirmed to work.

For example, the base URL used by default is `https://raw.githubusercontent.com/NatLabRockies/fastsim-vehicles/main`

### Path Segment Formatting

In Rust, `DatabaseSchemaV1::normalize_identifier` and `DatabaseSchemaV1::validate_identifier` can help with formatting and checking path segments for errors. Otherwise, you can try loading from the database with FASTSim and it will provide descriptive error messaging.

# Uploading to the FASTSim Vehicle Database

All changes must be done via pull request so that CI/CD can run and perform checks on the new vehicle files:
- `db_path` must be provided for all vehicle files, and must match the actual locations in the database
- All FASTSim 3 vehicles must be loadable (using FASTSim cloned and compiled from the `fastsim-3` branch)
  - Enforces path segment formatting rules as described above
