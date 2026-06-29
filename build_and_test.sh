cargo test && \
pip install --group dev . && \
# pytest -v python/fastsim/tests/ &&
pytest -v  && \
echo "Complete success!"
