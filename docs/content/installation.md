# Installation

FASTSim is available as a package for use in [Python 3.10 through 3.15](https://www.python.org).

For the quickest start, install from [PyPI](https://pypi.org/project/fastsim/). Install from source if you are developing FASTSim or need unreleased changes.

## Install from PyPI (recommended)

In an active Python environment, run:

```bash
pip install fastsim
```

This will install the latest release of FASTSim as a Python package.

See [](#next-steps) for further resources.

## Install from Source (advanced)

First, clone the repository:

```bash
git clone https://github.com/NatLabRockies/fastsim.git
cd fastsim
```

Then, install Rust and build FASTSim. FASTSim's backend is written entirely in Rust, so it is required to build from source.

- **Option A: Pixi (recommended)**

   [Pixi](https://pixi.prefix.dev/latest/) manages the Rust toolchain and Python dependencies for you. After [installing pixi](https://pixi.prefix.dev/latest/installation/), compile FASTSim with a release profile build and an editable Python package:

   ```bash
   pixi install
   ```

   Run `pixi install -e dev` instead to use the developer environment, which compiles faster using the debug profile, at the cost of runtime performance.

- **Option B: Manual**

   Install Python using your environment manager of choice, then install the [Rust toolchain](https://www.rust-lang.org/tools/install), e.g. using rustup:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

   Then, compile FASTSim with maturin via pip.

   ```bash
   pip install -e .
   ```

   Run `pip install -e . --group dev` instead to install with developer dependencies.

For more detail on developer builds or alternate pixi environments, see [](developers/environment-setup.md).

(next-steps)=
## Next Steps

See the following resources:

:::{card} [](getting-started.ipynb)
:link: getting-started.ipynb

A high-level overview showing how to load pre-defined vehicles, run simulations, and inspect results.

:::

:::::{card}

[](user-guide/user-guide.md):

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