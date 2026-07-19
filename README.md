<img src="docs/assets/fastsim-icon.svg" height="200">

[![homepage](https://img.shields.io/badge/homepage-fastsim-blue)](https://www.nlr.gov/transportation/fastsim.html)
[![tests](https://github.com/NatLabRockies/fastsim/actions/workflows/tests.yaml/badge.svg)](https://github.com/NatLabRockies/fastsim/actions/workflows/tests.yaml)
[![release](https://github.com/NatLabRockies/fastsim/actions/workflows/release.yaml/badge.svg)](https://github.com/NatLabRockies/fastsim/actions/workflows/release.yaml?event=release)
[![python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/fastsim/)
[![documentation](https://img.shields.io/badge/documentation-book-blue.svg)](https://natlabrockies.github.io/fastsim/)
[![github](https://img.shields.io/badge/github-fastsim-blue.svg)](https://github.com/NatLabRockies/fastsim)

The **Future Automotive Systems Technology Simulator** (FASTSim) provides a simple
way to compare powertrains and estimate the impact of technology improvements
on on-road vehicle efficiency and performance. FASTSim has been used to model
light-duty passenger cars, two-wheelers, and medium- and heavy-duty vocational
vehicles.

FASTSim is designed for rapid analysis, making it practical to run many
simulations and compare technology scenarios across vehicle classes and use
cases. FASTSim outputs also feed other NLR tools, including:
- [RouteE](https://www.nlr.gov/transportation/route-energy-prediction-model):
  an energy prediction tool and energy-aware routing engine
- [T3CO](https://www.nlr.gov/transportation/t3co):
  a medium- and heavy-duty vehicle total cost of ownership assessment tool
- [ADOPT](https://www.nlr.gov/transportation/adopt):
  a technology-driven consumer choice and vehicle market adoption model

FASTSim models conventional vehicles, hybrids, plug-in hybrids,
and battery electric vehicles over a set of standard regulatory drive cycles.
You can also define custom vehicles and custom drive cycles to simulate
any on-road vehicle over realistic scenarios.

For detailed documentation on FASTSim, including usage, background, and modeling philosophy, see https://natlabrockies.github.io/fastsim.

For a collection of prebuilt vehicle models, see the [FASTSim Vehicle Database](https://github.com/NatLabRockies/fastsim-vehicles).

# Installation

FASTSim is available as a package for use in
[Python 3.10 through 3.15](https://www.python.org).

In an active Python environment, run:

```bash
pip install fastsim
```

This installs the latest FASTSim release as a Python package.

For more detailed instructions, including compilation of FASTSim from source, see the
[Installation](https://natlabrockies.github.io/fastsim/installation)
page of
[FASTSim's documentation](https://natlabrockies.github.io/fastsim).

## Usage

For instruction on using FASTSim to simulate on-road vehicle performance, see the
[User Guide](https://natlabrockies.github.io/vehicles)
section of
[FASTSim's documentation](https://natlabrockies.github.io/fastsim).
