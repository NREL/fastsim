echo "Testing rust" &&
(cd rust && cargo test) &&
pip install -qe ".[dev]" &&
echo "Running python tests" &&
pytest -v python/fastsim/tests/ &&
echo "Verifying that demos run" &&
pytest -v python/fastsim/demos/ &&
echo "Complete success!"
