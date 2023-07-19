use crate::imports::*;
use serde_this_or_that::as_bool;

use super::link_idx::*;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
#[altrios_api]
pub struct Location {
    /// User-defined name of the location/terminal
    #[serde(rename = "Location ID")]
    pub location_id: String,
    #[serde(rename = "Offset (m)")]
    pub offset: si::Length,
    #[serde(rename = "Link Index")]
    pub link_idx: LinkIdx,
    #[serde(rename = "Is Front End")]
    #[serde(deserialize_with = "as_bool")]
    pub is_front_end: bool,
    #[serde(rename = "Grid Region")]
    pub grid_region: String,
}
pub type LocationMap = HashMap<String, Vec<Location>>;

#[cfg_attr(feature = "pyo3", pyfunction(name = "import_locations"))]
pub fn import_locations_py(filename: String) -> anyhow::Result<LocationMap> {
    import_locations(&PathBuf::from(filename))
}

pub fn import_locations(filename: &Path) -> anyhow::Result<LocationMap> {
    let file_read = File::open(filename)?;
    let mut reader = csv::Reader::from_reader(file_read);
    let mut location_map = LocationMap::default();
    for result in reader.deserialize() {
        let location: Location = result?;
        if !location_map.contains_key(&location.location_id) {
            location_map.insert(location.location_id.clone(), vec![]);
        }
        location_map
            .get_mut(&location.location_id)
            .unwrap()
            .push(location);
    }
    Ok(location_map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locations_import() {
        import_locations(Path::new("./src/track/link/locations.csv")).unwrap();
    }
}
