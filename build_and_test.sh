python python/fastsim/check_versions.py && \
cargo fmt --check && \
cargo test && \
pip install --group test -e . && \
# ruff check . && \
pytest -v
