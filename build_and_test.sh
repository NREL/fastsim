(cd rust/ && cargo test) && \
pip install -e ".[dev]" && \
# pytest -v python/fastsim/tests/ &&
pytest -v python/fastsim/demos
