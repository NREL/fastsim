# Requires `just` (https://github.com/casey/just): a command runner, install with:
#   macOS:   brew install just
#   Linux:   sudo apt install just   (or see https://just.systems/man/en/packages.html)
#   Windows: winget install --id Casey.Just
# Then run `just --list` from the repo root to see available recipes.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# These recipes (other than `setup`) call cargo/python/pip/pytest/jupyter-book directly rather
# than through a specific environment manager (pixi, uv, conda, ...) — bring your own active
# environment, same as build_and_test.sh did.

# One-time setup after cloning: install the pixi-managed dev environment and approve .envrc (macOS/Linux)
setup: _setup-direnv
    pixi install -e dev

[unix]
_setup-direnv:
    #!/usr/bin/env sh
    if ! command -v direnv >/dev/null 2>&1; then
        echo ""
        echo "Note: direnv not found (optional — install it for automatic environment activation on cd)."
        echo "  macOS:  brew install direnv"
        echo "  Linux:  sudo apt install direnv"
        echo "  Then add the shell hook: https://direnv.net/docs/hook.html"
        echo ""
    else
        grep -q 'direnv hook' ~/.zshrc 2>/dev/null || grep -q 'direnv hook' ~/.bashrc 2>/dev/null || {
            echo ""
            echo "Note: direnv is installed but not hooked into your shell."
            echo "Add one of the following (as appropriate) to your shell rc file and restart your shell:"
            echo "  ~/.zshrc:   eval \"\$(direnv hook zsh)\""
            echo "  ~/.bashrc:  eval \"\$(direnv hook bash)\""
            echo "  See: https://direnv.net/docs/hook.html"
            echo ""
        }
        direnv allow . || {
            echo ""
            echo "Warning: 'direnv allow .' failed — run it manually after checking your .envrc."
            echo ""
        }
    fi

[windows]
_setup-direnv:
    @echo "Note: direnv is not available on Windows — skipping."

# Full local dev-loop check: version consistency → rust fmt/test → python build+install → python tests
check: check-versions rust-check py-build py-test

# Check consistency of versions across the project (Cargo.toml, pyproject.toml, etc.)
check-versions:
    python python/fastsim/check_versions.py

# Format-check and test the Rust workspace
rust-check:
    cargo fmt --check
    cargo test --workspace --all-features

# Editable-install the package (triggers the Rust build via the maturin PEP 517 backend) with test deps
py-build:
    pip install --group test -e .

# Run the Python test suite (including notebooks via nbmake), serially
py-test:
    pytest -v

# Run the Python test suite in parallel via pytest-xdist
py-test-xdist:
    pytest -v -n auto --dist=loadscope

# Start a local live-reloading docs server
docs:
    cd docs && jupyter book start --execute

# Build the docs strictly (fails on warnings), executing notebooks
build-docs:
    cd docs && jupyter book build --strict --html --execute
