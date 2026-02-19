# How to compile/test Rust code

## cargo build
`cargo build` will not compile when run in `/rust` due to problems compiling `/rust/fastsim-py`.
`cargo build` should compile when run in the `/rust/fastsim-core`.

## cargo test
`cargo test` should compile when run in /rust because there are no tests in `/rust/fastsim-py`.

## build_and_test.sh
Running `sh build_and_test.sh` from the root fastsim directory compile/tests the Rust code, and tests the Python code. It should compile without errors.


# Releasing
## Incrementing the Version Number
Increment the 3rd decimal place in the version number for small changes (e.g.
minor bug fixes, new variables), the 2nd for medium changes (e.g. new methods
or classes), and the 1st for large changes (e.g. changes to the interface that
might affect backwards compatibility / the API interface).

## Instructions
1. Create and check out a new branch, e.g. for version X.X.X:
    ```
    git checkout -b fastsim-X.X.X
    ```
1. Update the version number in the `pyproject.toml` file
1. If changes have happened in `rust/`, increment the Rust crate version numbers in `rust/fastsim-core/Cargo.toml` and `rust/fastsim-core/fastsim-proc-macros/Cargo.toml`
1. Commit changes, as appropriate:
    ```
    git add pyproject.toml README.md rust/fastsim-core/Cargo.toml rust/fastsim-core/fastsim-proc-macros/Cargo.toml
    ```
    ```
    git commit -m "vX.X.X"
    ```
1. Tag the commit with the new version number, prepended with a `v`:
    ```
    git tag vX.X.X
    ```
    Or, optionally, you can also add a tag message with the `-m` flag, for example:
    ```
    git tag vX.X.X -m "release version X.X.X"
    ```
1. Push the commit to the GitHub.com repository (for Git remote setup instructions, see [this page](https://github.nrel.gov/MBAP/fastsim/wiki/Setting-up-FASTSim-Git-remotes-for-development)):
    ```
    git push -u external fastsim-X.X.X
    ```
1. Push the tag:
    ```
    git push external vX.X.X
    ```
    This will start the `wheels.yaml` GitHub Actions workflow and run all tests
1. Create a PR for the new version in the external repository, using the `release` label for organization
1. When all tests pass, and a review has been completed, merge the PR
1. If changes were made in `rust/`, publish the crates (you must be listed as an owner of both crates on crates.io):  
    If necessary, log into crates.io first after adding and verifying your email at https://crates.io/settings/profile:
    ```sh
    cargo login
    ```
    Then, run these commands to update the crates (order matters):
    ```sh
    (cd rust/fastsim-core/fastsim-proc-macros && cargo publish)
    (cd rust/fastsim-core && cargo publish)
    ```
1. Start a new release at https://github.com/NREL/fastsim/releases/new, selecting `vX.X.X` as both the tag and the release name. Click "Generate release notes" to automatically create a detailed change log.
1. Click "Publish release". Wheels will then be built for various platforms and automatically uploaded to the PyPI at https://pypi.org/project/fastsim/. **Check that the release workflow finished properly at https://github.com/NREL/fastsim/actions/workflows/release.yaml!**
1. Synchronize changes to the internal GitHub repository:
    ```
    git pull external fastsim-2
    git push origin fastsim-2
    ```
