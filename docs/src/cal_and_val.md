# Calibration and Validation of Vehicle Models
FASTSim powertrain models can have varying levels of calibration and resolution based on available calibration and validation data. In the simplest (US) cases, the only available validation data for a powertrain model is the EPA "window sticker" energy consumption rates. However, there are also situations in which detailed dynamometer or on-road data is available for a particular vehicle, enabling much more detailed model calibration. This documentation is meant to summarize these various calibration levels and the tools available to help with more detailed calibration.

## Calibration/Validation Levels

| Level | Calibration | Validation | 
| --- | --- | --- | 
| 0 | Vehicle is parameterized without any fitting to performance data. This is called __parameterization__, not calibration.  | Could be none or could be validated against aggregate energy consumption data like EPA window sticker values. | 
| 1 | Vehicle parameters are adjusted so that model results reasonably match test data for aggregate, cycle-level data (e.g. fuel usage, net SOC change). | Model results reasonably match at least some aggregate, cycle-level test data not used in any calibration process. |
| 2 | Vehicle parameters are adjusted so that model results reasonably match test data for time-resolved test data (e.g. instantaneous fuel usage, instantaneous cumulative fuel usage, instantaneous SOC). | Model results reasonably match at least some time-resolved test data not used in any calibration process. |
| 3 | Some amount of component-level thermal modeling is included and vehicle parameters are adjusted so that model results reasonably match test data for time-resolved test data (e.g. instantaneous fuel usage, instantaneous cumulative fuel usage, instantaneous SOC). | Model results reasonably match time-resolved test data not used in any calibration process that covers various temperatures and/vehcile transient thermal states. |

Examples of calibration levels 0, 2, and 3 from the [FASTSim Validation Report](https://www.nrel.gov/docs/fy22osti/81097.pdf):

![image](https://github.com/NREL/fastsim/assets/4818940/1b7dae5d-b328-406e-9e2c-07abadff7a3a)

![image](https://github.com/NREL/fastsim/assets/4818940/530f6a15-8400-4618-a97a-da67609f6ecd)

![image](https://github.com/NREL/fastsim/assets/4818940/8483661f-dee4-4d59-9d69-e6d54dae0100)

## Calibration Level 0 (Parameterization) Guidelines
As noted in the table above, parameterization of a new FASTSim powertrain model is performed when little or no ground truth performance data is available for a specific vehicle. One example of this is if EPA window-sticker fuel economy data is the only available performance data. In this situation, it is recommended to parameterize a FASTSim powertrain model using the most reliable vehicle parameters from available information (e.g., specification websites). This helps to avoid overfitting and relies on the robustness of the FASTSim approach to capture the most important powertrain dynamics and to simulate energy consumption.

- Create a new vehicle file, either from a template or an existing vehicle model (ideally of the same powertrain type)
- Enter vehicle parameters from various specification sources (__note__: it is recommended to document the source of specifications that are used to determine each parameter)
  - `veh_pt_type` and `fc_eff_type` are important high level descriptors that define the powertrain technology
    - `veh_pt_type`: Vehicle powertrain type
      - Parameter values:
        - `Conv`: conventional (ICE, gasoline or diesel) vehicle
        - `HEV`: hybrid electric vehicle
        - `PHEV`: plug-in hybrid electric vehicle
        - `BEV`: battery electric vehicle
    - `fc_eff_type`: Fuel converter efficiency type
      - This parameter is used to retrieve the default `fc_eff_map` for a particular engine type if a custom map is not provided
      - Unnecessary and not used for vehicles without a fuel converter (e.g. BEVs)
      - Parameter values:
        - `SI`: spark ignition
        - `Atkinson`: Atkinson cycle (typical for hybrids)
        - `Diesel`: diesel (compression ignition)
        - `H2FC`: hydrogen fuel cell (use with `veh_pt_type` set to `HEV`)
        - `HD_Diesel`: heavy-duty diesel
  - `veh_override_kg` is the simplest way to specify total vehicle mass
    - If not provided, the various component mass parameters will be used to calculate total vehicle mass
    - If `veh_override_kg` is provided, component mass parameters are unnecessary and not used
  - `drag_coef` and `wheel_rr_coef` can be calculated from dynamometer road load equation coefficients (ABCs) for vehicles tested by the US EPA using `fastsim.auxiliaries.abc_to_drag_coeffs`. Test data, including the road load coefficients from coast-down testing, for cars tested by the US EPA is available [here](https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy).
    - `drag_coef` is sometimes provided on specification websites and reasonable values informed by engineering judgement for `wheel_rr_coef` can be used , but when possible the ABCs and `fastsim.auxiliaries.abc_to_drag_coeffs` method should be used instead
  - `wheel_radius_m` is often not explicitly available for a vehicle, but a tire code can be supplied to `fastsim.utils.calculate_tire_radius` to calculate a radius
  - Note: For hybrids, 'total system power' is often provided (e.g., combined ICE and electric motor powers). This should not be used for either `fc_max_kw` or `mc_max_kw`, peak engine-only power should be used for `fc_max_kw` and peak electric motor-only power for `mc_max_kw`.

## Calibration Level 2 Guidelines
- Copy
  [calibration_demo.py](https://github.com/NREL/fastsim/blob/fastsim-2/python/fastsim/demos/calibration_demo.py)
  to your project directory and modify as needed.
- By default, this script selects the model that minimizes the euclidean error across
  all objectives, which may not be the way that you want to select your final design.
  By looking at the plots that get generated in `save_path`, you can use both the time
  series and parallel coordinates plots to down select an appropriate design.
- Because PyMOO is a multi-objective optimizer that finds a multi-dimensional Pareto
  surface, it will not necessarily return a single _best_ result -- rather, it will
  produce a pareto-optimal set of results, and you must down select.  Often, the design
  with minimal euclidean error will be the best design, but it's good to pick a handful
  of designs from the pareto set and check how they behave in the time-resolved plots
  that can be optionally generated by the optimization script.
- Run `python calibration_demo.py --help` to see details about how to run calibration
  and validation. Greater population size typically results in faster convergence at the
  expense of increased run time for each generation.  There's no benefit in having a
  number of processes larger than the population size.  `xtol` and `ftol` (see CLI help)
  can be used to adjust when the minimization is considered converged.  If the
  optimization is terminating when `n_max_gen` is hit, then that means it has not
  converged, and you may want to increase `n_max_gen`.
- Usually, start out with an existing vehicle model that is reasonably close to the
  new vehicle, and make sure to provide as many explicit parameters as possible.  In
  some cases, a reasonable engineering judgment is appropriate.
- Resample data to 1 Hz.  This is a good idea because higher frequency data will cause
  fastsim to run more slowly.  This can be done with `fastsim.resample.resample`.  Be
  sure to specify `rate_vars` (e.g. fuel power flow rate [W]), which will be time
  averaged over the previous time step in the new frequency.
- Identify test data signals and corresponding fastsim signals that need to match.
  These pairs of signals will be used to construct minimization objectives.  See
  where `obj_names` is defined in `calibration_demo.py` for an example.
- See where `cycs[key]` gets assigned to see an example of constructing a Cycle from a dataframe.  
- Partition out calibration/validation data by specifying a tuple of  regex patterns
  that correspond to cycle names.  See where `cal_cyc_patterns` is defined for an
  example. Typically, it's good to reserve about 25-33% of your data for validation.  
- To set parameters and corresponding ranges that the optimizer is allowed to use in
  getting the model to match test data, see where `params_and_bounds` is defined
  below.  

