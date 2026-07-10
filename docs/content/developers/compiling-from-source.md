# Compiling from Source

First, set up your environment (see [Environment Setup](environment-setup.md)).

To compile FASTSim from source and install into your environment, run:
```bash
maturin develop
```

Or, to compile with optimizations (longer compilation, but faster at runtime):
```bash
maturin develop --release
```
