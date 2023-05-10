(cd rust/ && cargo test --workspace --exclude=uom) && \
(cd rust/fastsim-py/ && maturin develop --release) && \
DEVELOP_MODE=True pip install -e ".[dev]" && \
pytest -v fastsim/tests/