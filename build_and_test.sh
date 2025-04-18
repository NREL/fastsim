cargo test && \
maturin develop --release && \
# pytest -v python/fastsim/tests/ &&
(pytest -v || (pip install -e '.[dev]' && pytest -v) ) && \
echo "Complete success!"
