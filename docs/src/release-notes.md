## Release Notes

2.1.2 -- SerdeAPI revamp with many new functions, various new vehicles, calibration demo, better error propagation, demo testing
2.1.1 -- license changed to Apache 2.0, default cycle grade and road type to zero if not provided, defaults to regenerative braking parameters, optional documentation fields now generated in Rust
2.1.0 -- release and installation improvements, RustVehicle init cleanup, calibration improvements
2.0.11 - 2.0.22 -- PyPI fixes.  Also, Rust version is now >100x faster than Python version.
2.0.10 -- logging fixes, proc macro reorganization, some CAVs performance fixes  
2.0.9 -- support for mac ARM/RISC architecture  
2.0.8 -- performance improvements  
2.0.6 -- `dist_v2_m` fixes and preliminary CAV functionality  
2.0.5 -- added `to_rust` method for cycle  
2.0.4 -- exposed `veh.set_veh_mass`  
2.0.3 -- exposed `veh.__post_init__`  
2.0.2 -- provisioned for non-default vehdb path  
2.0.1 -- bug fix  
2.0.0 -- All second-by-second calculations are now implemented in both rust and python.  Rust provides a ~30x speedup  
1.3.1 -- `fastsim.simdrive.copy_sim_drive` function can deepcopy jit to non-jit (and back) for pickling  
1.2.6 -- time dilation bug fix for zero speed  
1.2.4 -- bug fix changing `==` to `=`  
1.2.3 -- `veh_file` can be passed as standalone argument.  `fcEffType` can be anything if `fcEffMap` is provided, but typing is otherwise enforced.  
1.2.2 -- added checks for some conflicting vehicle parameters.  Vehicle parameters `fcEffType` and `vehPtType` must now be str type.  
1.2.1 -- improved time dilation and added test for it  
1.1.7 -- get_numba_veh() and get_numba_cyc() can now be called from already jitted objects  
1.1.6 -- another bug fix for numba compatibility with corresponding unit test  
1.1.5 -- bug fix for numba compatibility of fcPeakEffOverride and mcPeakEffOverride  
1.1.4 -- nan bug fix for fcPeakEffOverride and mcPeakEffOverride  
1.1.3 -- provisioned for optional load time motor and engine peak overrides  
1.1.2 -- made vehicle loading _more_ more robust  
1.1.1 -- made vehicle loading more robust  
1.1.0 -- separated jitclasses into own module, made vehicle engine and motor efficiency setting more robust  
1.0.4 -- bug fix with custom engine curve  
1.0.3 -- bug fixes, faster testing  
1.0.2 -- forced type np.float64 on vehicle mass attributes  
1.0.1 -- Added `vehYear` attribute to vehicle and other minor changes.
1.0.0 -- Implemented unittest package.  Fixed energy audit calculations to be based on achieved speed.  Updated this file.  Improved documentation.  Vehicle can be instantiated as dict.
0.1.5 -- Updated to be compatible with ADOPT
0.1.4 -- Bug fix: `mcEffMap` is now robust to having zero as first element
0.1.3 -- Bug fix: `fastsim.vehicle.Vehicle` method `set_init_calcs` no longer overrides `fcEffMap`.
0.1.2 -- Fixes os-dependency of xlwings by not running stuff that needs xlwings.  Improvements in functional test.  Refinment utomated typying of jitclass objects.
0.1.1 -- Now includes label fuel economy and/or battery kW-hr/mi values that match excel and test for benchmarking against Excel values and CPU time.

