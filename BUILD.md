# Basic Building Instructions
To build, create and activate a python3.10 environment, [install Rust](https://www.rust-lang.org/tools/install), and run `sh build_and_test.sh` or equivalent commandss as appropriate for your OS. 

# Using GitHub Actions
If you have push access to https://github.com/NREL/fastsim, you can push tags to trigger github actions.

## For Building Wheels for Python 3.8 - 3.10 for Linux, Mac, and Windows
Push a tag that matches the regex `'v[0-9]+.[0-9]+.[0-9]+'` , e.g. `v2.0.8`.  This will run tests and build wheels in a zip file on the GitHub repo.  

## For PyPI Release
Push tags that match the regex `'r[0-9]+.[0-9]+.[0-9]+'` , e.g. `r2.0.8`.  This will run tests, build wheels in a zip file on the GitHub repo, and release the new version on [PyPI](https://pypi.org/project/fastsim/).  Make sure that `version` in [pyproject.toml](pyproject.toml) and the section Release Notes in [README.md](README.md) have versions that match the git tag.  