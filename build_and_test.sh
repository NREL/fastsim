(cd rust/ && cargo test) && \
pip install -qe ".[dev]" && \
# pytest -v python/fastsim/tests/ &&
pytest -v python/fastsim/demos
