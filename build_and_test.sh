cargo test && \
pip install --group test . && \
# pytest -v python/fastsim/tests/ &&
pytest -v  && \
echo "Complete success!"
