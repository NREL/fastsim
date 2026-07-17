python python/fastsim/check_versions.py && \
cargo fmt --check && \
# cargo clippy -- -D warnings && \
cargo test && \
pip install --group test -e . && \
# ruff check . && \
pytest -v
