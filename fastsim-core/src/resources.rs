use include_dir::{include_dir, Dir};
pub const RESOURCES_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/resources");
