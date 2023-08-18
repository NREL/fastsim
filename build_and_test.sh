# build and test with local version of `fastsim-proc-macros`
(cd rust/fastsim-core/ && cargo test --features dev-proc-macros) && \
(cd rust/fastsim-cli/ && cargo test) && \
pip install -qe ".[dev]" && \
pytest -v python/fastsim/tests/