# What is FASTSim?

The Future Automotive Systems Technology Simulator (**FASTSim**) is an open-source vehicle powertrain simulation tool for evaluating the efficiency and performance of on-road vehicles.
Developed by the National Laboratory of the Rockies (NLR), it provides fast, credible estimates of powertrain performance and energy consumption.
FASTSim models powertrain technologies from conventional internal-combustion vehicles to hybrids, plug-in hybrids, battery electric vehicles, and fuel-cell electric vehicles, and has been used to simulate light-duty passenger vehicles, two-wheelers, and a wide variety of medium- and heavy-duty vehicles.

FASTSim is used to answer common technology-assessment questions quickly, with fewer required inputs than high-fidelity physics models. It occupies a practical middle ground between basic efficiency assumptions and detailed component/control models. This page provides relevant context through the history and philosophy of the modeling approach and level of fidelity to real-world performance it aims to achieve.

## History

FASTSim was developed in the late 2000s and early 2010s to be an accessible, efficient, accurate, and robust tool for comparing vehicle powertrains. While more detailed models are able to answer specific questions, they require detailed data and expert knowledge to run and generate accurate results. For many types of analysis, however, the details can be simplified and still provide accurate results. FASTSim took this approach, modeling the vehicle components at as high a level as possible while still being accurate, which makes finding inputs, running the model, and interpreting results easier, faster, and less error prone [[Brooker et al., 2015](https://docs.nlr.gov/docs/fy15osti/63623.pdf)].

While many applications do not require it, FASTSim's open source approach allows for customization to capture temperature-dependent characteristics, component speed-related variations, and other detailed aspects.

Today, the actively maintained [GitHub page](https://github.com/NatLabRockies/fastsim) houses the latest version of the open-source FASTSim Python package, with core logic written in Rust for high simulation speed, modeling flexibility, and runtime stability guarantees.
Releases are made to the [PyPI](https://pypi.org/project/fastsim) (Python Package Index) repository for easy distribution and installation via `pip`. The [NLR.gov FASTSim homepage](https://www.nlr.gov/transportation/fastsim) distributes the legacy Excel and Python-only versions and describes FASTSim's extensive publication history.

## Modeling Philosophy

FASTSim sits between very simple screening tools and highly detailed component/control models, with flexibility to increase model detail when required. The core philosophy is to balance:

- Accuracy
- Input burden and calibration effort
- Runtime speed
- Transparency and reproducibility

FASTSim occupies a “sweet spot” along the continuum of modeling tools based on each tool’s trade-off between accuracy and complexity (Figure ES-1). Here, “complexity” includes:

- Required number of input parameters
- Availability of required input data
- Time needed to obtain inputs and perform calibration
- Software requirements and computational overhead

FASTSim is designed to balance predictive accuracy with model complexity across a wide range of analytical tasks. It is particularly well suited for quickly and conveniently conducting large numbers of simulations over representative real-world driving distributions and/or myriad vehicle design variations. In these analyses, uncertainties and efficiency impacts from operating conditions or design variants often exceed the small uncertainties introduced by FASTSim's modeling simplifications.

For more information on FASTSim's modeling philosophy and validation against real-world data, refer to the [2021 FASTSim Validation Report](https://docs.nlr.gov/docs/fy22osti/81097.pdf). The validation report compares FASTSim outputs with laboratory and on-road data across all modeled powertrain types. It shows that:

- Basic FASTSim models perform well for many high-level efficiency and energy applications.
- Calibrated FASTSim models can closely match second-by-second and trip-level data when sufficient calibration data is available.

## Conclusion

Because FASTSim is open source, computationally lightweight, and built for
large batches of simulations, it is well suited for analyses that need to be
shared, replicated, and debated across many stakeholders.

Across NLR's suite of analysis software, FASTSim also serves as a core simulation layer that can feed downstream tools for routing, cost of ownership, market adoption analysis, and more. For NLR tools related to FASTSim, see [Related Tools](related-tools.md).
