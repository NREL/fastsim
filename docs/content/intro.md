---
site:
  hide_title_block: true
---

<div style="height: 2rem;"></div>

:::{image} ../assets/edit-fastsim-icon.png
:alt: FASTSim logo
:width: 300px
:align: center
:::

# FASTSim Documentation

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

For information on FASTSim's history and modeling philosophy, see the
[Background](background/what-is-fastsim.md) section.

For a collection of prebuilt vehicle models, see the [FASTSim Vehicle Database](https://github.com/NatLabRockies/fastsim-vehicles).

## Get Started

- [](installation.md): install FASTSim and dependencies
- [](getting-started.ipynb): run your first simulation
- [](user-guide/user-guide.md):
    - [Modeling Vehicles](user-guide/vehicle-models/vehicle.md): define and configure vehicle models
    - [Drive Cycles](user-guide/drive-cycles/drive-cycle.ipynb): work with built-in and custom cycles
    - [Running Simulations](user-guide/running-simulations/simdrive.ipynb): execute simulations and inspect results

## Learn More

- GitHub Repository: https://github.com/NatLabRockies/fastsim
- FASTSim Vehicle Database: https://github.com/NatLabRockies/fastsim-vehicles
- Documentation: https://natlabrockies.github.io/fastsim
- Release Notes: https://github.com/NatLabRockies/fastsim/releases
- NLR FASTSim Homepage: https://www.nlr.gov/transportation/fastsim

## Contact
- Open a GitHub issue at https://github.com/NatLabRockies/fastsim/issues.
- Email [fastsim@nlr.gov](mailto:fastsim@nlr.gov) to reach the development team directly.