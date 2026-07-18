# Installation

FASTSim supports Python 3.10 through 3.15. For the quickest start, install from [PyPI](https://pypi.org/project/fastsim/). Install from source if you are developing FASTSim or need unreleased changes.

## Install from PyPI (recommended)

In an active Python environment, run:

```bash
pip install fastsim
```

## Install from Source

FASTSim's backend is written entirely in Rust. Install the
[Rust toolchain](https://www.rust-lang.org/tools/install) first.

Then install FASTSim from source:

1. Clone the repository:

   ```bash
   git clone https://github.com/NatLabRockies/fastsim.git
   cd fastsim
   ```

1. Install from the repository root:

   ```bash
   pip install .
   ```

Optional:

- Add `-e` for an editable install so source changes are picked up without
  reinstalling.
- Add `--group dev` to install optional developer dependencies.

For more detail on developer builds, see [](developers/compiling-from-source.md).

## Next Steps

With FASTSim installed, continue to [](getting-started.ipynb) to load a vehicle,
run a simulation, and inspect results.
