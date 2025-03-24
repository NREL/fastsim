cargo test && \
pip install -e .[dev] && \
# pytest -v python/fastsim/tests/ &&
pytest -v && \
echo "Complete success!"
