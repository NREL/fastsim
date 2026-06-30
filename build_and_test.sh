python python/fastsim/check_versions.py && \
# ruff check . && \
cargo test && \
pip install --group test --group lint -e . && \
pytest -v
