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

The Future Automotive Systems Technology Simulator (**FASTSim**) is an open-source vehicle powertrain simulation tool for evaluating the efficiency and performance of on-road vehicles.
Developed by the National Laboratory of the Rockies (NLR), it provides fast, credible estimates of powertrain performance and energy consumption.
FASTSim models powertrain technologies from conventional internal-combustion vehicles to hybrids, plug-in hybrids, and battery electric vehicles, and has been used to simulate light-duty passenger vehicles, two-wheelers, and a wide variety of medium- and heavy-duty vehicles.

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

For a searchable collection of prebuilt vehicle models, see the [FASTSim Vehicle Database](https://natlabrockies.github.io/fastsim-vehicles/).

## Getting Started

- [](installation.md): install FASTSim and dependencies
- [](getting-started.ipynb): run your first simulation
- [](user-guide/user-guide.md):
    - [Modeling Vehicles](user-guide/vehicle-models/vehicle.md): define and configure vehicle models
    - [Drive Cycles](user-guide/drive-cycles/drive-cycle.ipynb): work with built-in and custom cycles
    - [Running Simulations](user-guide/running-simulations/simdrive.ipynb): execute simulations and inspect results

## Learn More

- GitHub Repository: https://github.com/NatLabRockies/fastsim
- FASTSim Vehicle Database: https://natlabrockies.github.io/fastsim-vehicles/
- Documentation: https://natlabrockies.github.io/fastsim
- Release Notes: https://github.com/NatLabRockies/fastsim/releases
- NLR FASTSim Homepage: https://www.nlr.gov/transportation/fastsim

## Contact
- Open a GitHub issue at https://github.com/NatLabRockies/fastsim/issues.
- Email [fastsim@nlr.gov](mailto:fastsim@nlr.gov) to reach the FASTSim team directly.
