# Related Tools

## [RouteE](https://www.nlr.gov/transportation/route-energy-prediction-model)

NLR's **Route Energy Prediction** (RouteE) modeling tools enable vehicle energy estimation and energy-aware route planning.

The open-source RouteE suite consists of two key tools—**RouteE Powertrain** and **RouteE Compass**—along with the simplified RouteE API. 

- ### [RouteE Powertrain](https://github.com/NatLabRockies/routee-powertrain)

    Built as a modular Python package, RouteE Powertrain predicts the energy consumption of a given vehicle over a proposed route. It accounts for various driving conditions such as anticipated traffic congestion, traffic speed, road type, number of lanes, road grade, and turns. It enables users to obtain energy estimates for the full range of vehicle sizes—from light-duty vehicles to heavy-duty trucks and transit buses—for trips or routes where detailed drive cycle data may be unavailable. 

- ### [RouteE Compass](https://github.com/NatLabRockies/routee-compass)

    RouteE Compass is an energy-aware routing tool that incorporates energy consumption predictions from RouteE Powertrain into network routing algorithms. It can accommodate any vehicle size or powertrain technology. RouteE Compass is built as a high-performance Rust application with easy-to-use Python bindings. 

## [T3CO](https://www.nlr.gov/transportation/t3co)

NLR's **Transportation Technology Total Cost of Ownership** (T3CO) tool enables levelized assessments of the full life cycle costs of advanced technology commercial vehicles.

Medium- and heavy-duty commercial vehicles operate in diverse vocations with varied duty cycles, performance, and technical and economic requirements. T3CO accounts for these operational variations as well as advanced technology considerations.

The T3CO methodology, vetted by industry and stakeholders, employs best practices developed across U.S. Department of Energy total cost of ownership studies while addressing key gaps in standard methodologies. 

## [ADOPT](https://www.nlr.gov/transportation/adopt)

NLR's **Automotive Deployment Options Projection Tool** (ADOPT) is a vehicle consumer choice and stock model for light- and heavy-duty vehicles.

ADOPT estimates the impact of vehicle technology improvements on U.S. vehicle sales and energy use. It provides consumer choice estimates based on questions such as:

- How much impact do lower battery prices have on electric vehicle sales?
- How quickly would the market adopt a co-optimized engine/fuel combination that achieves a 10% efficiency improvement?
- How do fuel prices impact the electric vehicle/plug-in hybrid electric vehicle sales mix?
- How does vehicle lightweighting impact powertrain sales?
