# Installation

FASTSim is available as a package for use in [Python 3.10 through 3.15](https://www.python.org).

For the quickest start, install from [PyPI](https://pypi.org/project/fastsim/). Install from source if you are developing FASTSim or need unreleased changes.

## Install from PyPI (recommended)

In an active Python environment, run:

```bash
pip install fastsim
```

This will install the latest version of FASTSim as a Python package.

See [](#next-steps) for further resources.

## Install from Source (developers)

FASTSim's backend is written entirely in Rust. Install the
[Rust toolchain](https://www.rust-lang.org/tools/install) first (or
[set up a developer environment](developers/environment-setup.md)).

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

(next-steps)=
## Next Steps

See the following resources:

:::{card} [](getting-started.ipynb)
:link: getting-started.ipynb

A high-level overview showing how to load pre-defined vehicles, run simulations, and inspect results.

:::

:::::{card}

**User Guide**:

::::{grid} 1 1 2 2 3 3

:::{grid-item-card} [Vehicle Models](user-guide/vehicle-models/vehicle.md)
:link: user-guide/vehicle-models/vehicle.md
:link-type: doc
Define and configure vehicle models
:::

:::{grid-item-card} [Drive Cycles](user-guide/drive-cycles/drive-cycle.ipynb)
:link: user-guide/drive-cycles/drive-cycle.ipynb
:link-type: doc
Work with built-in and custom cycles
:::

:::{grid-item-card} [Running Simulations](user-guide/running-simulations/simdrive.ipynb)
:link: user-guide/running-simulations/simdrive.ipynb
:link-type: doc
Execute simulations and inspect results
:::

::::

:::::