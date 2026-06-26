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

For an editable install that also includes the development dependencies, run
`pip install -e ".[dev]"` from the repository root instead. Source changes are
then picked up the next time FASTSim is imported.

Building from source is not necessary for most use cases. For more detail on
the developer build, see [Compilation from Source](compilation-from-source.md).

## Next Steps

With FASTSim installed, head to
[Getting Started](../demo_notebooks/getting_started/demo_getting_started.ipynb)
to load a vehicle, run a simulation, and inspect the results.
