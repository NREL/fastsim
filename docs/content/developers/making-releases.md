# Making Releases

1. Determine new version number according to [semantic versioning conventions](https://semver.org/)

1. Update FASTSim version in the following locations:
    - `pyproject.toml`
    - `fastsim-core/Cargo.toml`
      - package version
      - `fastsim-proc-macros` dependency version
    - `fastsim-core/fastsim-proc-macros/Cargo.toml`
    - `fastsim-py/Cargo.toml`

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
