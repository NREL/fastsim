# What is FASTSim?

FASTSim (Future Automotive Systems Technology Simulator) is a high-level
vehicle simulation tool developed at the [National Laboratory of the Rockies
(NLR)](https://www.nlr.gov) to support transportation analysis. The tool
answers common technology-assessment questions quickly, with fewer required
inputs than high-fidelity physics models.

## History

:::{note}
This section needs more detailed input
:::

<!-- FASTSim started as an NLR vehicle powertrain analysis model focused on speed,
accessibility, and broad scenario coverage. The 2015 SAE paper documents the
core approach: use high-level component representations and drive-cycle
simulation to compare powertrains quickly.

NLR's 2021 validation report then formalized FASTSim's modeling philosophy as
an accuracy-complexity tradeoff, with FASTSim occupying a practical "sweet
spot" for large batch analysis and real-world cycle studies. -->

Today, the actively maintained [GitHub page](https://github.com/NatLabRockies/fastsim) houses the latest version of the open-source FASTSim Python package, with core logic written in Rust for maximum simulation speed, modeling flexibility, and runtime stability guarantees.
Releases are made to the [PyPI](https://pypi.org/project/fastsim) (Python Package Index) for easy distribution and installation via `pip`. The [NLR.gov FASTSim homepage](https://www.nlr.gov/transportation/fastsim) distributes the legacy Excel and Python-only versions and describes FASTSim's extensive publication history.

## Modeling Philosophy

FASTSim sits between very simple screening tools and highly detailed component/control models,
with flexibility to increase model detail when required.
The core philosophy is to balance:

- Accuracy
- Input burden and calibration effort
- Runtime speed
- Transparency and reproducibility

FASTSim occupies a “sweet spot” along the continuum of modeling tools based on each tool’s
trade-off between accuracy and complexity, where “complexity” includes the required number of
input parameters, availability of required input data, time required to obtain the inputs and perform
calibration, software requirements, and computational overhead to run (Figure ES-1). FASTSim
is designed to balance predictive accuracy with model complexity across a wide range of analytical
tasks. Across its range of capabilities, FASTSim is particularly well suited for quickly and
conveniently conducting large numbers of simulations over representative real-world driving
distributions and/or myriad vehicle design variations. In such analyses, the uncertainties and
efficiency impacts from the broad spectrum of operating conditions or design variants far exceed
small uncertainties resulting from modeling simplifications within FASTSim.

For more information on FASTSim's modeling philosophy and validation against real-world data,
refer to the [2021 FASTSim Validation Report](https://docs.nlr.gov/docs/fy22osti/81097.pdf).

The validation report compares FASTSim outputs with laboratory and on-road data across
all modeled powertrains. It shows that:

- Basic FASTSim models perform well for many high-level efficiency and energy studies.
- Customized FASTSim models can closely match second-by-second and trip-level data in
  targeted calibration efforts.

## Conclusion

Because FASTSim is open source, computationally lightweight, and built for
large batches of simulations, it is well suited for analyses that need to be
shared, replicated, and debated across many stakeholders.

Across NLR's suite of analysis software, FASTSim also serves as a core simulation layer that can
feed downstream tools for routing, cost of ownership, market adoption analysis, and more.
