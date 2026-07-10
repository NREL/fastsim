# Installation

FASTSim supports Python 3.10 through 3.12. Install it from PyPI for the
quickest start, or build it from source for the latest development version.

## From PyPI

In an active Python environment, install the latest release with pip:

```
pip install fastsim
```

## From Source

Building from source gives you the latest development version. FASTSim's core
is written in Rust, so you will need the
[Rust toolchain](https://www.rust-lang.org/tools/install) installed first. The
build tool (maturin) is installed automatically during the build, so the Rust
toolchain is the only prerequisite you need to set up yourself.

1. Clone the repository and enter it:

   ```
   git clone https://github.com/NatLabRockies/fastsim.git
   cd fastsim
   ```

2. Install the package from the repository root:

   ```
   pip install .
   ```

   - Add the editable `-e` flag to detect source changes each time FASTSim is imported.
   - Add the `--group dev` flag to install optional developer dependencies.

Building from source is not necessary for most use cases. For more detail on
the developer build, see [Compilation from Source](developers/compilation-from-source.md).

## Next Steps

With FASTSim installed, head to
[Getting Started](../demo_notebooks/getting_started/demo_getting_started.ipynb)
to load a vehicle, run a simulation, and inspect the results.
