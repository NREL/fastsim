# How to compile/test Rust code

## cargo build
`cargo build` will not compile when run in `/rust` due to problems compiling `/rust/fastsim-py`.
`cargo build` should compile when run in the `/rust/fastsim-core`.

## cargo test
`cargo test` should compile when run in /rust because there are no tests in `/rust/fastsim-py`.

## build_and_test.sh
Running `sh build_and_test.sh` from the root fastsim directory compile/tests the Rust code, and tests the Python code. It should compile without errors.