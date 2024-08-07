#![cfg(feature = "resources")]

use include_dir::{include_dir, Dir};
pub const RESOURCES_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/resources");

/// List the available resources in the resources directory
/// - subdir: &str, a subdirectory to choose from the resources directory
/// NOTE: if subdir cannot be resolved, returns an empty list
/// RETURNS: a vector of strings for resources that can be loaded
pub fn list_resources(subdir: &str) -> Vec<String> {
    if subdir.is_empty() {
        Vec::<String>::new()
    } else if let Some(resources_path) = RESOURCES_DIR.get_dir(subdir) {
        let mut file_names: Vec<String> = resources_path
            .files()
            .filter_map(|entry| entry.path().file_name()?.to_str().map(String::from))
            .collect();
        file_names.sort();
        file_names
    } else {
        Vec::<String>::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_resources() {
        let result = list_resources("cycles");
        assert!(result.len() == 3);
        assert!(result[0] == "HHDDTCruiseSmooth.csv");
        // NOTE: at the time of writing this test, there is no
        // vehicles subdirectory. The agreed-upon behavior in
        // that case is that list_resources should return an
        // empty vector of string.
        let another_result = list_resources("vehicles");
        assert!(another_result.len() == 0);
    }
}
