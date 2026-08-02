# Compiling from Source

First, set up your environment (see [Environment Setup](environment-setup.md)\). The `pixi install`/`pip install` steps will compile FASTSim from source.

To recompile with release profile optimizations (longer compilation, but faster runtime):
```bash
maturin develop
```

Pixi users can run `pixi run py-build-release`.

To recompile with the debug profile, run:

```bash
maturin develop --profile dev
```

Pixi users can run `pixi run py-build`.
