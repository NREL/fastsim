#![cfg(feature = "resources")]

use include_dir::{include_dir, Dir};
pub const RESOURCES_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/resources");

/// List the available resources in the resources directory
/// - subdir: &str, a subdirectory to choose from the resources directory
///   if it cannot be resolved, then the top level is used to list resources
/// NOTE: if you want the top level, a good way to get that is to pass "".
/// RETURNS: a vector of strings for resources that can be loaded
pub fn list_resources(subdir: &str) -> Vec<String> {
    let resources_path = if let Some(rp) = RESOURCES_DIR.get_dir(subdir) {
        rp
    } else {
        &RESOURCES_DIR
    };
    let mut file_names: Vec<String> = resources_path
        .files()
        .filter_map(|entry| entry.path().file_name()?.to_str().map(String::from))
        .collect();
    file_names.sort();
    file_names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_resources() {
        let result = list_resources("cycles");
        assert!(result.len() == 3);
        assert!(result[0] == "HHDDTCruiseSmooth.csv");
    }
}
