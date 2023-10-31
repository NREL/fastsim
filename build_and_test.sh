# build and test with local version of `fastsim-proc-macros`
echo "Testing rust" && \
(cd rust/fastsim-core/ && cargo test) && \
(cd rust/fastsim-cli/ && cargo test) && \
pip install -qe ".[dev]" && \
echo "Running python tests" && \
pytest -v python/fastsim/tests/ && \
echo "Verifying that demos run" && \
pytest -v python/fastsim/demos/ && \
echo "Complete success!"