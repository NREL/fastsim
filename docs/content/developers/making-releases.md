# Making Releases

1. Determine new version number according to [semantic versioning conventions](https://semver.org/)

    ## Semantic Versioning in FASTSim

    FASTSim uses the following semantic versioning rules:

    - All vehicles serialized within a major version must be loadable within the same *major version*

        - **Example:** A vehicle made in v3.0.5 must be loadable in FASTSim versions up to (not including) v4.0.0
        - Vehicle models specify a minimum FASTSim version they are compatible with (within a major version)
          - Loading a vehicle model designed for a more recent version of FASTSim will throw a warning, but still be attempted
        - New fields introduced within a major version must be optional, or set a sensible default value

    - Breaking changes:
      - *Major* version increments (e.g. 3.x.y -> 4.0.0) may include breaking changes to the Python API
        - Removal of functions, function naming, Python-exposed object naming/structure, etc.
      - *Minor* version increments (e.g. 3.0.x -> 3.1.0) may include breaking changes to the Rust API
        - Alterations to Rust-only functions, function naming, structures, etc.
        - Downstream Rust projects should use [tilde requirements](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#tilde-requirements)
          - Example `Cargo.toml` dependency on FASTSim:
            ```toml
            [dependencies]
            fastsim-core = "~3.1" 
            # - allows 3.1.0, 3.1.1, etc.,
            # - disallows >= 3.2.0
            ```
            Or:
            ```toml
            [dependencies]
            fastsim-core = "~3.1.1"
            # - allows 3.1.1, 3.1.2, etc.,
            # - disallows < 3.1.1, >= 3.2.0
            ```
      - *Patch* version increments (e.g. 3.0.0 -> 3.0.1) are for bug fixes only, no intentional breaking changes

    In summary:

    | Version Increment | Vehicle Serialization Format | Rust API (`fastsim-core`) | Python API (`fastsim`) | Intended Change Scope |
    | --- | --- | --- | --- | --- |
    | Patch (`X.Y.Z -> X.Y.(Z+1)`) | Backward-compatible within major version | No intentional breaking changes | No intentional breaking changes | Bug fixes and small improvements |
    | Minor (`X.Y.Z -> X.(Y+1).0`) | Backward-compatible within major version | May include breaking changes | No intentional breaking changes | New features and Rust-side evolution |
    | Major (`X.Y.Z -> (X+1).0.0`) | May include breaking changes to serialization format | May include breaking changes | May include breaking changes | Large model/API changes |

1. Update FASTSim version in the following locations:
    - `pyproject.toml`
    - `fastsim-core/Cargo.toml`
      - package version
      - `fastsim-proc-macros` dependency version
    - `fastsim-core/fastsim-proc-macros/Cargo.toml`
    - `fastsim-py/Cargo.toml`

    Commit these changes via git

1. Tag the latest commit with your version number and push

    ```
    git tag vX.Y.Z
    ```

    ```
    git push origin tag vX.Y.Z
    ```

1. Draft a new release at https://github.com/NatLabRockies/fastsim/releases/new
    - Select the newly created tag
    - Autogenerate release notes
    - Add extra description of changes, new features, bugfixes, etc.

1. Publish release
    - GitHub Actions will take care of testing, building wheels, and releasing to [PyPI](https://pypi.org/project/fastsim/) and [crates.io](https://crates.io/crates/fastsim-core)
