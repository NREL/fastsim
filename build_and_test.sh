cd rust/ && cargo test && cd - && \
pip install -e .[dev] && \
# pytest -v python/fastsim/tests/ &&
pytest -v
