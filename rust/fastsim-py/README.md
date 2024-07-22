Update this for fastsim

# How to update .pyi file
1. Install mypy: `pip install mypy`
2. Get the html documentation: 
```bash
    cd FASTSim/fastsim-core
    cargo doc --open
```
3. Go to the python folder in FASTSim: 
```bash
    cd FASTSim/fastsim-py/python/
```
4. Generate a new pyi file:
```bash
    stubgen fastsim_py/
```
5. You should see a new file `out/fastsim_py.pyi` and within it there will be stubs for all the classes and functions in the `fastsim_py` module

6. Go to the html documentation that was opened and search the classes, update the `out/fastsim_py.pyi` accordingly.
