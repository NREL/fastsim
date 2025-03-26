{
  pixi run build_and_test && echo "Complete success with pixi!"
} || {
  cargo test && \
  pip install -e .[dev] && \
  # pytest -v python/fastsim/tests/ &&
  pytest -v && \
  echo "Complete success!"
}
