# Calibration and Validation of Vehicle Models

# General Guidelines
- Copy
  [calibration_demo.py](https://github.com/NREL/fastsim/blob/fastsim-2/python/fastsim/demos/calibration_demo.py)
  to your project directory and modify as needed.
- By default, this script selects the model that minimizes the euclidean error across
  all objectives, which may not be the way that you want to select your final design.
  By looking at the plots that get generated in `save_path`, you can use both the time
  series and parallel coordinates plots to down select an appropriate design.
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
  example.
- To set parameters and corresponding ranges that the optimizer is allowed to use in
  getting the model to match test data, see where `params_and_bounds` is defined
  below.  

