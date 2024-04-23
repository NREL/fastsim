# Table of Contents

* [fastsim](#fastsim)
  * [package\_root](#fastsim.package_root)
* [fastsim.cycle](#fastsim.cycle)
  * [CycleCache](#fastsim.cycle.CycleCache)
    * [interp\_grade](#fastsim.cycle.CycleCache.interp_grade)
    * [interp\_elevation](#fastsim.cycle.CycleCache.interp_elevation)
  * [Cycle](#fastsim.cycle.Cycle)
    * [from\_file](#fastsim.cycle.Cycle.from_file)
    * [from\_dict](#fastsim.cycle.Cycle.from_dict)
    * [get\_numba\_cyc](#fastsim.cycle.Cycle.get_numba_cyc)
    * [build\_cache](#fastsim.cycle.Cycle.build_cache)
    * [dt\_s\_at\_i](#fastsim.cycle.Cycle.dt_s_at_i)
    * [delta\_elev\_m](#fastsim.cycle.Cycle.delta_elev_m)
    * [\_\_len\_\_](#fastsim.cycle.Cycle.__len__)
    * [to\_dict](#fastsim.cycle.Cycle.to_dict)
    * [reset\_orphaned](#fastsim.cycle.Cycle.reset_orphaned)
    * [copy](#fastsim.cycle.Cycle.copy)
    * [average\_grade\_over\_range](#fastsim.cycle.Cycle.average_grade_over_range)
    * [calc\_distance\_to\_next\_stop\_from](#fastsim.cycle.Cycle.calc_distance_to_next_stop_from)
    * [modify\_by\_const\_jerk\_trajectory](#fastsim.cycle.Cycle.modify_by_const_jerk_trajectory)
    * [modify\_with\_braking\_trajectory](#fastsim.cycle.Cycle.modify_with_braking_trajectory)
  * [LegacyCycle](#fastsim.cycle.LegacyCycle)
    * [\_\_init\_\_](#fastsim.cycle.LegacyCycle.__init__)
  * [cyc\_equal](#fastsim.cycle.cyc_equal)
  * [to\_microtrips](#fastsim.cycle.to_microtrips)
  * [make\_cycle](#fastsim.cycle.make_cycle)
  * [equals](#fastsim.cycle.equals)
  * [concat](#fastsim.cycle.concat)
  * [resample](#fastsim.cycle.resample)
  * [clip\_by\_times](#fastsim.cycle.clip_by_times)
  * [accelerations](#fastsim.cycle.accelerations)
  * [peak\_acceleration](#fastsim.cycle.peak_acceleration)
  * [peak\_deceleration](#fastsim.cycle.peak_deceleration)
  * [calc\_constant\_jerk\_trajectory](#fastsim.cycle.calc_constant_jerk_trajectory)
  * [accel\_for\_constant\_jerk](#fastsim.cycle.accel_for_constant_jerk)
  * [speed\_for\_constant\_jerk](#fastsim.cycle.speed_for_constant_jerk)
  * [dist\_for\_constant\_jerk](#fastsim.cycle.dist_for_constant_jerk)
  * [detect\_passing](#fastsim.cycle.detect_passing)
  * [average\_step\_speeds](#fastsim.cycle.average_step_speeds)
  * [average\_step\_speed\_at](#fastsim.cycle.average_step_speed_at)
  * [trapz\_step\_distances](#fastsim.cycle.trapz_step_distances)
  * [trapz\_step\_start\_distance](#fastsim.cycle.trapz_step_start_distance)
  * [trapz\_distance\_for\_step](#fastsim.cycle.trapz_distance_for_step)
  * [trapz\_distance\_over\_range](#fastsim.cycle.trapz_distance_over_range)
  * [extend\_cycle](#fastsim.cycle.extend_cycle)
  * [create\_dist\_and\_target\_speeds\_by\_microtrip](#fastsim.cycle.create_dist_and_target_speeds_by_microtrip)
  * [copy\_cycle](#fastsim.cycle.copy_cycle)
* [fastsim.vehicle\_base](#fastsim.vehicle_base)
* [fastsim.resample](#fastsim.resample)
  * [resample](#fastsim.resample.resample)
* [fastsim.auxiliaries](#fastsim.auxiliaries)
  * [R\_air](#fastsim.auxiliaries.R_air)
  * [abc\_to\_drag\_coeffs](#fastsim.auxiliaries.abc_to_drag_coeffs)
  * [drag\_coeffs\_to\_abc](#fastsim.auxiliaries.drag_coeffs_to_abc)
* [fastsim.simdrivelabel](#fastsim.simdrivelabel)
  * [get\_label\_fe](#fastsim.simdrivelabel.get_label_fe)
* [fastsim.simdrive](#fastsim.simdrive)
  * [SimDriveParams](#fastsim.simdrive.SimDriveParams)
    * [from\_dict](#fastsim.simdrive.SimDriveParams.from_dict)
    * [\_\_init\_\_](#fastsim.simdrive.SimDriveParams.__init__)
    * [to\_rust](#fastsim.simdrive.SimDriveParams.to_rust)
    * [reset\_orphaned](#fastsim.simdrive.SimDriveParams.reset_orphaned)
  * [copy\_sim\_params](#fastsim.simdrive.copy_sim_params)
  * [sim\_params\_equal](#fastsim.simdrive.sim_params_equal)
  * [SimDrive](#fastsim.simdrive.SimDrive)
    * [\_\_init\_\_](#fastsim.simdrive.SimDrive.__init__)
    * [gap\_to\_lead\_vehicle\_m](#fastsim.simdrive.SimDrive.gap_to_lead_vehicle_m)
    * [sim\_drive](#fastsim.simdrive.SimDrive.sim_drive)
    * [init\_for\_step](#fastsim.simdrive.SimDrive.init_for_step)
    * [sim\_drive\_walk](#fastsim.simdrive.SimDrive.sim_drive_walk)
    * [activate\_eco\_cruise](#fastsim.simdrive.SimDrive.activate_eco_cruise)
    * [sim\_drive\_step](#fastsim.simdrive.SimDrive.sim_drive_step)
    * [solve\_step](#fastsim.simdrive.SimDrive.solve_step)
    * [set\_misc\_calcs](#fastsim.simdrive.SimDrive.set_misc_calcs)
    * [set\_comp\_lims](#fastsim.simdrive.SimDrive.set_comp_lims)
    * [set\_power\_calcs](#fastsim.simdrive.SimDrive.set_power_calcs)
    * [set\_ach\_speed](#fastsim.simdrive.SimDrive.set_ach_speed)
    * [set\_hybrid\_cont\_calcs](#fastsim.simdrive.SimDrive.set_hybrid_cont_calcs)
    * [set\_fc\_forced\_state](#fastsim.simdrive.SimDrive.set_fc_forced_state)
    * [set\_hybrid\_cont\_decisions](#fastsim.simdrive.SimDrive.set_hybrid_cont_decisions)
    * [set\_fc\_power](#fastsim.simdrive.SimDrive.set_fc_power)
    * [set\_post\_scalars](#fastsim.simdrive.SimDrive.set_post_scalars)
    * [to\_rust](#fastsim.simdrive.SimDrive.to_rust)
  * [copy\_sim\_drive](#fastsim.simdrive.copy_sim_drive)
  * [sim\_drive\_equal](#fastsim.simdrive.sim_drive_equal)
  * [run\_simdrive\_for\_accel\_test](#fastsim.simdrive.run_simdrive_for_accel_test)
  * [SimDrivePost](#fastsim.simdrive.SimDrivePost)
    * [\_\_init\_\_](#fastsim.simdrive.SimDrivePost.__init__)
    * [get\_diagnostics](#fastsim.simdrive.SimDrivePost.get_diagnostics)
    * [set\_battery\_wear](#fastsim.simdrive.SimDrivePost.set_battery_wear)
  * [SimDriveJit](#fastsim.simdrive.SimDriveJit)
  * [estimate\_soc\_corrected\_fuel\_kJ](#fastsim.simdrive.estimate_soc_corrected_fuel_kJ)
* [fastsim.tests.test\_utils](#fastsim.tests.test_utils)
* [fastsim.tests.test\_simdrive](#fastsim.tests.test_simdrive)
  * [TestSimDriveClassic](#fastsim.tests.test_simdrive.TestSimDriveClassic)
    * [test\_sim\_drive\_step](#fastsim.tests.test_simdrive.TestSimDriveClassic.test_sim_drive_step)
    * [test\_sim\_drive\_walk](#fastsim.tests.test_simdrive.TestSimDriveClassic.test_sim_drive_walk)
* [fastsim.tests.test\_logging](#fastsim.tests.test_logging)
* [fastsim.tests.test\_following](#fastsim.tests.test_following)
  * [TestFollowing](#fastsim.tests.test_following.TestFollowing)
    * [test\_that\_we\_have\_a\_gap\_between\_us\_and\_the\_lead\_vehicle](#fastsim.tests.test_following.TestFollowing.test_that_we_have_a_gap_between_us_and_the_lead_vehicle)
    * [test\_that\_the\_gap\_changes\_over\_the\_cycle](#fastsim.tests.test_following.TestFollowing.test_that_the_gap_changes_over_the_cycle)
    * [test\_that\_following\_works\_over\_parameter\_sweep](#fastsim.tests.test_following.TestFollowing.test_that_following_works_over_parameter_sweep)
    * [test\_that\_we\_can\_use\_the\_idm](#fastsim.tests.test_following.TestFollowing.test_that_we_can_use_the_idm)
    * [test\_sweeping\_idm\_parameters](#fastsim.tests.test_following.TestFollowing.test_sweeping_idm_parameters)
    * [test\_distance\_based\_grade\_on\_following](#fastsim.tests.test_following.TestFollowing.test_distance_based_grade_on_following)
* [fastsim.tests.test\_vehicle](#fastsim.tests.test_vehicle)
  * [TestVehicle](#fastsim.tests.test_vehicle.TestVehicle)
    * [test\_equal](#fastsim.tests.test_vehicle.TestVehicle.test_equal)
    * [test\_properties](#fastsim.tests.test_vehicle.TestVehicle.test_properties)
    * [test\_fc\_efficiency\_override](#fastsim.tests.test_vehicle.TestVehicle.test_fc_efficiency_override)
    * [test\_set\_derived\_init](#fastsim.tests.test_vehicle.TestVehicle.test_set_derived_init)
* [fastsim.tests.test\_copy](#fastsim.tests.test_copy)
  * [TestCopy](#fastsim.tests.test_copy.TestCopy)
    * [test\_copy\_cycle](#fastsim.tests.test_copy.TestCopy.test_copy_cycle)
    * [test\_copy\_physical\_properties](#fastsim.tests.test_copy.TestCopy.test_copy_physical_properties)
    * [test\_copy\_vehicle](#fastsim.tests.test_copy.TestCopy.test_copy_vehicle)
    * [test\_copy\_sim\_params](#fastsim.tests.test_copy.TestCopy.test_copy_sim_params)
    * [test\_copy\_sim\_drive](#fastsim.tests.test_copy.TestCopy.test_copy_sim_drive)
* [fastsim.tests](#fastsim.tests)
  * [run\_functional\_tests](#fastsim.tests.run_functional_tests)
* [fastsim.tests.test\_auxiliaries](#fastsim.tests.test_auxiliaries)
* [fastsim.tests.test\_simdrivelabel](#fastsim.tests.test_simdrivelabel)
* [fastsim.tests.test\_simdrive\_sweep](#fastsim.tests.test_simdrive_sweep)
  * [main](#fastsim.tests.test_simdrive_sweep.main)
  * [TestSimDriveSweep](#fastsim.tests.test_simdrive_sweep.TestSimDriveSweep)
    * [test\_sweep](#fastsim.tests.test_simdrive_sweep.TestSimDriveSweep.test_sweep)
* [fastsim.tests.test\_cycle](#fastsim.tests.test_cycle)
  * [calc\_distance\_traveled\_m](#fastsim.tests.test_cycle.calc_distance_traveled_m)
  * [dicts\_are\_equal](#fastsim.tests.test_cycle.dicts_are_equal)
  * [TestCycle](#fastsim.tests.test_cycle.TestCycle)
    * [test\_monotonicity](#fastsim.tests.test_cycle.TestCycle.test_monotonicity)
    * [test\_load\_dict](#fastsim.tests.test_cycle.TestCycle.test_load_dict)
    * [test\_that\_udds\_has\_18\_microtrips](#fastsim.tests.test_cycle.TestCycle.test_that_udds_has_18_microtrips)
    * [test\_roundtrip\_of\_microtrip\_and\_concat](#fastsim.tests.test_cycle.TestCycle.test_roundtrip_of_microtrip_and_concat)
    * [test\_roundtrip\_of\_microtrip\_and\_concat\_using\_keep\_name\_arg](#fastsim.tests.test_cycle.TestCycle.test_roundtrip_of_microtrip_and_concat_using_keep_name_arg)
    * [test\_set\_from\_dict\_for\_a\_microtrip](#fastsim.tests.test_cycle.TestCycle.test_set_from_dict_for_a_microtrip)
    * [test\_duration\_of\_concatenated\_cycles\_is\_the\_sum\_of\_the\_components](#fastsim.tests.test_cycle.TestCycle.test_duration_of_concatenated_cycles_is_the_sum_of_the_components)
    * [test\_cycle\_equality](#fastsim.tests.test_cycle.TestCycle.test_cycle_equality)
    * [test\_that\_cycle\_resampling\_works\_as\_expected](#fastsim.tests.test_cycle.TestCycle.test_that_cycle_resampling_works_as_expected)
    * [test\_resampling\_and\_concatenating\_cycles](#fastsim.tests.test_cycle.TestCycle.test_resampling_and_concatenating_cycles)
    * [test\_resampling\_with\_hold\_keys](#fastsim.tests.test_cycle.TestCycle.test_resampling_with_hold_keys)
    * [test\_that\_resampling\_preserves\_total\_distance\_traveled\_using\_rate\_keys](#fastsim.tests.test_cycle.TestCycle.test_that_resampling_preserves_total_distance_traveled_using_rate_keys)
    * [test\_clip\_by\_times](#fastsim.tests.test_cycle.TestCycle.test_clip_by_times)
    * [test\_get\_accelerations](#fastsim.tests.test_cycle.TestCycle.test_get_accelerations)
    * [test\_that\_copy\_creates\_idential\_structures](#fastsim.tests.test_cycle.TestCycle.test_that_copy_creates_idential_structures)
    * [test\_make\_cycle](#fastsim.tests.test_cycle.TestCycle.test_make_cycle)
    * [test\_key\_conversion](#fastsim.tests.test_cycle.TestCycle.test_key_conversion)
    * [test\_get\_grade\_by\_distance](#fastsim.tests.test_cycle.TestCycle.test_get_grade_by_distance)
    * [test\_dt\_s\_vs\_dt\_s\_at\_i](#fastsim.tests.test_cycle.TestCycle.test_dt_s_vs_dt_s_at_i)
    * [test\_trapz\_step\_start\_distance](#fastsim.tests.test_cycle.TestCycle.test_trapz_step_start_distance)
    * [test\_that\_cycle\_cache\_interp\_grade\_substitutes\_for\_average\_grade\_over\_range](#fastsim.tests.test_cycle.TestCycle.test_that_cycle_cache_interp_grade_substitutes_for_average_grade_over_range)
    * [test\_that\_trapz\_step\_start\_distance\_equals\_cache\_trapz\_distances](#fastsim.tests.test_cycle.TestCycle.test_that_trapz_step_start_distance_equals_cache_trapz_distances)
    * [test\_average\_grade\_over\_range\_with\_and\_without\_cache](#fastsim.tests.test_cycle.TestCycle.test_average_grade_over_range_with_and_without_cache)
* [fastsim.tests.test\_cav\_sweep](#fastsim.tests.test_cav_sweep)
* [fastsim.tests.test\_soc\_correction](#fastsim.tests.test_soc_correction)
  * [TestSocCorrection](#fastsim.tests.test_soc_correction.TestSocCorrection)
    * [test\_that\_soc\_correction\_method\_works](#fastsim.tests.test_soc_correction.TestSocCorrection.test_that_soc_correction_method_works)
* [fastsim.tests.test\_coasting](#fastsim.tests.test_coasting)
  * [make\_coasting\_plot](#fastsim.tests.test_coasting.make_coasting_plot)
  * [make\_dvdd\_plot](#fastsim.tests.test_coasting.make_dvdd_plot)
  * [TestCoasting](#fastsim.tests.test_coasting.TestCoasting)
    * [test\_cycle\_reported\_distance\_traveled\_m](#fastsim.tests.test_coasting.TestCoasting.test_cycle_reported_distance_traveled_m)
    * [test\_cycle\_modifications\_with\_constant\_jerk](#fastsim.tests.test_coasting.TestCoasting.test_cycle_modifications_with_constant_jerk)
    * [test\_that\_cycle\_modifications\_work\_as\_expected](#fastsim.tests.test_coasting.TestCoasting.test_that_cycle_modifications_work_as_expected)
    * [test\_that\_we\_can\_coast](#fastsim.tests.test_coasting.TestCoasting.test_that_we_can_coast)
    * [test\_eco\_approach\_modeling](#fastsim.tests.test_coasting.TestCoasting.test_eco_approach_modeling)
    * [test\_consistency\_of\_constant\_jerk\_trajectory](#fastsim.tests.test_coasting.TestCoasting.test_consistency_of_constant_jerk_trajectory)
    * [test\_that\_final\_speed\_of\_cycle\_modification\_matches\_trajectory\_calcs](#fastsim.tests.test_coasting.TestCoasting.test_that_final_speed_of_cycle_modification_matches_trajectory_calcs)
    * [test\_that\_cycle\_distance\_reported\_is\_correct](#fastsim.tests.test_coasting.TestCoasting.test_that_cycle_distance_reported_is_correct)
    * [test\_brake\_trajectory](#fastsim.tests.test_coasting.TestCoasting.test_brake_trajectory)
    * [test\_logic\_to\_enter\_eco\_approach\_automatically](#fastsim.tests.test_coasting.TestCoasting.test_logic_to_enter_eco_approach_automatically)
    * [test\_that\_coasting\_works\_going\_uphill](#fastsim.tests.test_coasting.TestCoasting.test_that_coasting_works_going_uphill)
    * [test\_that\_coasting\_logic\_works\_going\_uphill](#fastsim.tests.test_coasting.TestCoasting.test_that_coasting_logic_works_going_uphill)
    * [test\_that\_coasting\_logic\_works\_going\_downhill](#fastsim.tests.test_coasting.TestCoasting.test_that_coasting_logic_works_going_downhill)
    * [test\_that\_coasting\_works\_with\_multiple\_stops\_and\_grades](#fastsim.tests.test_coasting.TestCoasting.test_that_coasting_works_with_multiple_stops_and_grades)
* [fastsim.tests.test\_rust](#fastsim.tests.test_rust)
  * [TestRust](#fastsim.tests.test_rust.TestRust)
    * [test\_discrepancies](#fastsim.tests.test_rust.TestRust.test_discrepancies)
    * [test\_vehicle\_for\_discrepancies](#fastsim.tests.test_rust.TestRust.test_vehicle_for_discrepancies)
    * [test\_fueling\_prediction\_for\_multiple\_vehicle](#fastsim.tests.test_rust.TestRust.test_fueling_prediction_for_multiple_vehicle)
* [fastsim.tests.test\_vs\_excel](#fastsim.tests.test_vs_excel)
  * [run](#fastsim.tests.test_vs_excel.run)
  * [run\_excel](#fastsim.tests.test_vs_excel.run_excel)
  * [compare](#fastsim.tests.test_vs_excel.compare)
  * [main](#fastsim.tests.test_vs_excel.main)
  * [TestExcel](#fastsim.tests.test_vs_excel.TestExcel)
    * [test\_vs\_excel](#fastsim.tests.test_vs_excel.TestExcel.test_vs_excel)
* [fastsim.tests.test\_eco\_cruise](#fastsim.tests.test_eco_cruise)
* [fastsim.vehicle](#fastsim.vehicle)
  * [clean\_data](#fastsim.vehicle.clean_data)
  * [Vehicle](#fastsim.vehicle.Vehicle)
    * [from\_vehdb](#fastsim.vehicle.Vehicle.from_vehdb)
    * [from\_file](#fastsim.vehicle.Vehicle.from_file)
    * [from\_df](#fastsim.vehicle.Vehicle.from_df)
    * [from\_dict](#fastsim.vehicle.Vehicle.from_dict)
    * [\_\_post\_init\_\_](#fastsim.vehicle.Vehicle.__post_init__)
    * [set\_derived](#fastsim.vehicle.Vehicle.set_derived)
    * [set\_veh\_mass](#fastsim.vehicle.Vehicle.set_veh_mass)
    * [veh\_type\_selection](#fastsim.vehicle.Vehicle.veh_type_selection)
    * [get\_mcPeakEff](#fastsim.vehicle.Vehicle.get_mcPeakEff)
    * [set\_mcPeakEff](#fastsim.vehicle.Vehicle.set_mcPeakEff)
    * [get\_fcPeakEff](#fastsim.vehicle.Vehicle.get_fcPeakEff)
    * [set\_fcPeakEff](#fastsim.vehicle.Vehicle.set_fcPeakEff)
    * [get\_numba\_veh](#fastsim.vehicle.Vehicle.get_numba_veh)
    * [to\_rust](#fastsim.vehicle.Vehicle.to_rust)
    * [reset\_orphaned](#fastsim.vehicle.Vehicle.reset_orphaned)
  * [LegacyVehicle](#fastsim.vehicle.LegacyVehicle)
    * [\_\_init\_\_](#fastsim.vehicle.LegacyVehicle.__init__)
  * [to\_native\_type](#fastsim.vehicle.to_native_type)
  * [copy\_vehicle](#fastsim.vehicle.copy_vehicle)
  * [veh\_equal](#fastsim.vehicle.veh_equal)
* [fastsim.utils](#fastsim.utils)
* [fastsim.utils.vehicle\_import\_preproc](#fastsim.utils.vehicle_import_preproc)
  * [process\_csv](#fastsim.utils.vehicle_import_preproc.process_csv)
  * [write\_csvs\_for\_each\_year](#fastsim.utils.vehicle_import_preproc.write_csvs_for_each_year)
  * [sort\_fueleconomygov\_data\_by\_year](#fastsim.utils.vehicle_import_preproc.sort_fueleconomygov_data_by_year)
  * [xlsx\_to\_csv](#fastsim.utils.vehicle_import_preproc.xlsx_to_csv)
  * [process\_epa\_test\_data](#fastsim.utils.vehicle_import_preproc.process_epa_test_data)
  * [create\_zip\_archives\_by\_year](#fastsim.utils.vehicle_import_preproc.create_zip_archives_by_year)
* [fastsim.utils.utilities](#fastsim.utils.utilities)
  * [R\_air](#fastsim.utils.utilities.R_air)
  * [get\_rho\_air](#fastsim.utils.utilities.get_rho_air)
  * [l\_\_100km\_to\_mpg](#fastsim.utils.utilities.l__100km_to_mpg)
  * [mpg\_to\_l\_\_100km](#fastsim.utils.utilities.mpg_to_l__100km)
  * [rollav](#fastsim.utils.utilities.rollav)
  * [camel\_to\_snake](#fastsim.utils.utilities.camel_to_snake)
  * [set\_log\_level](#fastsim.utils.utilities.set_log_level)
  * [disable\_logging](#fastsim.utils.utilities.disable_logging)
  * [enable\_logging](#fastsim.utils.utilities.enable_logging)
  * [suppress\_logging](#fastsim.utils.utilities.suppress_logging)
  * [get\_containers\_with\_path](#fastsim.utils.utilities.get_containers_with_path)
  * [get\_attr\_with\_path](#fastsim.utils.utilities.get_attr_with_path)
  * [set\_attr\_with\_path](#fastsim.utils.utilities.set_attr_with_path)
  * [set\_attrs\_with\_path](#fastsim.utils.utilities.set_attrs_with_path)
  * [calculate\_tire\_radius](#fastsim.utils.utilities.calculate_tire_radius)
  * [show\_plots](#fastsim.utils.utilities.show_plots)
  * [copy\_demo\_files](#fastsim.utils.utilities.copy_demo_files)
* [fastsim.inspect\_utils](#fastsim.inspect_utils)
  * [isprop](#fastsim.inspect_utils.isprop)
  * [isfunc](#fastsim.inspect_utils.isfunc)
  * [get\_attrs](#fastsim.inspect_utils.get_attrs)
* [fastsim.parameters](#fastsim.parameters)
  * [PhysicalProperties](#fastsim.parameters.PhysicalProperties)
  * [copy\_physical\_properties](#fastsim.parameters.copy_physical_properties)
  * [physical\_properties\_equal](#fastsim.parameters.physical_properties_equal)
  * [fc\_perc\_out\_array](#fastsim.parameters.fc_perc_out_array)
  * [chg\_eff](#fastsim.parameters.chg_eff)
* [fastsim.demos.vehicle\_import\_demo](#fastsim.demos.vehicle_import_demo)
  * [other\_inputs](#fastsim.demos.vehicle_import_demo.other_inputs)
* [fastsim.demos.fusion\_thermal\_cal\_post](#fastsim.demos.fusion_thermal_cal_post)
  * [save\_path](#fastsim.demos.fusion_thermal_cal_post.save_path)
* [fastsim.demos.cav\_sweep](#fastsim.demos.cav_sweep)
  * [ABSOLUTE\_EXTENDED\_TIME\_S](#fastsim.demos.cav_sweep.ABSOLUTE_EXTENDED_TIME_S)
  * [make\_debug\_plot](#fastsim.demos.cav_sweep.make_debug_plot)
  * [load\_cycle](#fastsim.demos.cav_sweep.load_cycle)
  * [main](#fastsim.demos.cav_sweep.main)
* [fastsim.demos.time\_dilation\_demo](#fastsim.demos.time_dilation_demo)
* [fastsim.demos.accel\_demo](#fastsim.demos.accel_demo)
  * [create\_accel\_cyc](#fastsim.demos.accel_demo.create_accel_cyc)
  * [main](#fastsim.demos.accel_demo.main)
* [fastsim.demos.cav\_demo](#fastsim.demos.cav_demo)
* [fastsim.demos](#fastsim.demos)
* [fastsim.demos.demo\_eu\_vehicle\_wltp](#fastsim.demos.demo_eu_vehicle_wltp)
* [fastsim.demos.fusion\_thermal\_demo](#fastsim.demos.fusion_thermal_demo)
* [fastsim.demos.timing\_demo](#fastsim.demos.timing_demo)
* [fastsim.demos.mp\_parallel\_demo](#fastsim.demos.mp_parallel_demo)
* [fastsim.demos.2017\_Ford\_F150\_thermal\_val](#fastsim.demos.2017_Ford_F150_thermal_val)
  * [lhv\_fuel\_btu\_per\_lbm](#fastsim.demos.2017_Ford_F150_thermal_val.lhv_fuel_btu_per_lbm)
* [fastsim.demos.test\_demos](#fastsim.demos.test_demos)
* [fastsim.demos.fusion\_thermal\_cal](#fastsim.demos.fusion_thermal_cal)
  * [lhv\_fuel\_btu\_per\_lbm](#fastsim.demos.fusion_thermal_cal.lhv_fuel_btu_per_lbm)
* [fastsim.demos.stop\_start\_demo](#fastsim.demos.stop_start_demo)
* [fastsim.demos.demo\_abc\_drag\_coef\_conv](#fastsim.demos.demo_abc_drag_coef_conv)
* [fastsim.demos.demo](#fastsim.demos.demo)
  * [v2](#fastsim.demos.demo.v2)
  * [data\_path](#fastsim.demos.demo.data_path)
  * [veh](#fastsim.demos.demo.veh)
  * [veh](#fastsim.demos.demo.veh)
  * [veh](#fastsim.demos.demo.veh)
* [fastsim.demos.wltc\_calibration](#fastsim.demos.wltc_calibration)
  * [WILLANS\_FACTOR](#fastsim.demos.wltc_calibration.WILLANS_FACTOR)
  * [E10\_HEAT\_VALUE](#fastsim.demos.wltc_calibration.E10_HEAT_VALUE)
* [fastsim.calibration](#fastsim.calibration)
  * [get\_error\_val](#fastsim.calibration.get_error_val)
  * [ModelObjectives](#fastsim.calibration.ModelObjectives)
    * [get\_errors](#fastsim.calibration.ModelObjectives.get_errors)
    * [update\_params](#fastsim.calibration.ModelObjectives.update_params)
  * [get\_parser](#fastsim.calibration.get_parser)

<a id="fastsim"></a>

# fastsim

Package containing modules for running FASTSim.
For example usage, see

<a id="fastsim.package_root"></a>

#### package\_root

```python
def package_root() -> Path
```

Returns the package root directory.

<a id="fastsim.cycle"></a>

# fastsim.cycle

Module containing classes and methods for cycle data.

<a id="fastsim.cycle.CycleCache"></a>

## CycleCache Objects

```python
class CycleCache()
```

<a id="fastsim.cycle.CycleCache.interp_grade"></a>

#### interp\_grade

```python
def interp_grade(dist: float)
```

Interpolate the single-point grade at the given distance.
Assumes that the grade at i applies from sample point (i-1, i]

<a id="fastsim.cycle.CycleCache.interp_elevation"></a>

#### interp\_elevation

```python
def interp_elevation(dist: float)
```

Interpolate the elevation at the given distance

<a id="fastsim.cycle.Cycle"></a>

## Cycle Objects

```python
@dataclass
class Cycle(object)
```

Object for containing time, speed, road grade, and road charging vectors
for drive cycle.  Instantiate with the `from_file` or `from_dict` method.

<a id="fastsim.cycle.Cycle.from_file"></a>

#### from\_file

```python
@classmethod
def from_file(cls, filename: str) -> Self
```

Load cycle from filename (str).
Can be absolute or relative path.  If relative, looks in working dir first
and then in `fastsim/resources/cycles`.  

File must contain columns for:
-- `cycSecs` or `time_s`
-- `cycMps` or `mps`
-- `cycGrade` or `grade` (optional)
-- `cycRoadType` or `road_type` (optional)

<a id="fastsim.cycle.Cycle.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, cyc_dict: dict) -> Self
```

Load cycle from dict, which must contain keys for:
-- `cycSecs` or `time_s`
-- `cycMps` or `mps`
-- `cycGrade` or `grade` (optional)
-- `cycRoadType` or `road_type` (optional)

<a id="fastsim.cycle.Cycle.get_numba_cyc"></a>

#### get\_numba\_cyc

```python
def get_numba_cyc()
```

Deprecated.

<a id="fastsim.cycle.Cycle.build_cache"></a>

#### build\_cache

```python
def build_cache() -> CycleCache
```

Calculates a dataclass containing expensive-to-calculate items. The data created
can persist between calls and optionally be passed into methods that can use
it which will result in a performance enhancement.
RETURN: CycleCache

<a id="fastsim.cycle.Cycle.dt_s_at_i"></a>

#### dt\_s\_at\_i

```python
def dt_s_at_i(i: int) -> float
```

Calculate the time-step duration for time-step `i`.
Returns: the time-step duration in seconds

<a id="fastsim.cycle.Cycle.delta_elev_m"></a>

#### delta\_elev\_m

```python
@property
def delta_elev_m() -> np.ndarray
```

Cumulative elevation change w.r.t. to initial

<a id="fastsim.cycle.Cycle.__len__"></a>

#### \_\_len\_\_

```python
def __len__() -> int
```

return cycle length

<a id="fastsim.cycle.Cycle.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict[str, np.ndarray]
```

Returns cycle as dict rather than class instance.

<a id="fastsim.cycle.Cycle.reset_orphaned"></a>

#### reset\_orphaned

```python
def reset_orphaned()
```

Dummy method for flexibility between Rust/Python version interfaces

<a id="fastsim.cycle.Cycle.copy"></a>

#### copy

```python
def copy() -> Self
```

Return a copy of this Cycle instance.

<a id="fastsim.cycle.Cycle.average_grade_over_range"></a>

#### average\_grade\_over\_range

```python
def average_grade_over_range(distance_start_m,
                             delta_distance_m,
                             cache: Optional[CycleCache] = None)
```

Returns the average grade over the given range of distances
- distance_start_m: non-negative-number, the distance at start of evaluation area (m)
- delta_distance_m: non-negative-number, the distance traveled from distance_start_m (m)
RETURN: number, the average grade (rise over run) over the given distance range
Note: grade is assumed to be constant from just after the previous sample point
until the current sample point. That is, grade[i] applies over the range of
distances, d, from (d[i - 1], d[i]]

<a id="fastsim.cycle.Cycle.calc_distance_to_next_stop_from"></a>

#### calc\_distance\_to\_next\_stop\_from

```python
def calc_distance_to_next_stop_from(distance_m: float,
                                    cache: Optional[CycleCache] = None
                                    ) -> float
```

Calculate the distance to next stop from `distance_m`
- distance_m: non-negative-number, the current distance from start (m)
RETURN: returns the distance to the next stop from distance_m
NOTE: distance may be negative if we're beyond the last stop

<a id="fastsim.cycle.Cycle.modify_by_const_jerk_trajectory"></a>

#### modify\_by\_const\_jerk\_trajectory

```python
def modify_by_const_jerk_trajectory(idx, n, jerk_m__s3, accel0_m__s2)
```

Modifies the cycle using the given constant-jerk trajectory parameters
- idx: non-negative integer, the point in the cycle to initiate
modification (note: THIS point is modified since trajectory should be calculated from idx-1)
- jerk_m__s3: number, the "Jerk" associated with the trajectory (m/s3)
- accel0_m__s2: number, the initial acceleration (m/s2)
NOTE:
- modifies cyc in place to hit any critical rendezvous_points by a trajectory adjustment
- CAUTION: NOT ROBUST AGAINST VARIABLE DURATION TIME-STEPS
RETURN: Number, final modified speed (m/s)

<a id="fastsim.cycle.Cycle.modify_with_braking_trajectory"></a>

#### modify\_with\_braking\_trajectory

```python
def modify_with_braking_trajectory(brake_accel_m__s2: float,
                                   idx: int,
                                   dts_m: Optional[float] = None) -> tuple
```

Add a braking trajectory that would cover the same distance as the given constant brake deceleration
- brake_accel_m__s2: negative number, the braking acceleration (m/s2)
- idx: non-negative integer, the index where to initiate the stop trajectory, start of the step (i in FASTSim)
- dts_m: None | float: if given, this is the desired distance-to-stop in meters. If not given, it is
    calculated based on braking deceleration.
RETURN: (non-negative-number, positive-integer)
- the final speed of the modified trajectory (m/s) 
- the number of time-steps required to complete the braking maneuver
NOTE:
- modifies the cycle in place for the braking trajectory

<a id="fastsim.cycle.LegacyCycle"></a>

## LegacyCycle Objects

```python
class LegacyCycle(object)
```

Implementation of Cycle with legacy keys.

<a id="fastsim.cycle.LegacyCycle.__init__"></a>

#### \_\_init\_\_

```python
def __init__(cycle: Cycle)
```

Given cycle, returns legacy cycle.

<a id="fastsim.cycle.cyc_equal"></a>

#### cyc\_equal

```python
def cyc_equal(a: Cycle, b: Cycle) -> bool
```

Return True if a and b are equal

<a id="fastsim.cycle.to_microtrips"></a>

#### to\_microtrips

```python
def to_microtrips(cycle, stop_speed_m__s=1e-6, keep_name=False)
```

Split a cycle into an array of microtrips with one microtrip being a start
to subsequent stop plus any idle (stopped time).

**Arguments**:

  ----------
- `cycle` - drive cycle converted to dictionary by cycle.to_dict()
- `stop_speed_m__s` - speed at which vehicle is considered stopped for trip
  separation
- `keep_name` - (optional) bool, if True and cycle contains "name", adds
  that name to all microtrips

<a id="fastsim.cycle.make_cycle"></a>

#### make\_cycle

```python
def make_cycle(ts, vs, gs=None, rs=None) -> dict
```

(Array Num) (Array Num) (Array Num)? -> Dict
Create a cycle from times, speeds, and grades. If grades is not
specified, it is set to zero.

**Arguments**:

  ----------
- `ts` - array of times [s]
- `vs` - array of vehicle speeds [mps]
- `gs` - array of grades
- `rs` - array of road types (charging or not)

<a id="fastsim.cycle.equals"></a>

#### equals

```python
def equals(c1, c2) -> bool
```

Dict Dict -> Bool
Returns true if the two cycles are equal, false otherwise

**Arguments**:

  ----------
- `c1` - cycle as dictionary from to_dict()
- `c2` - cycle as dictionary from to_dict()

<a id="fastsim.cycle.concat"></a>

#### concat

```python
def concat(cycles, name=None)
```

Concatenates cycles together one after another into a single dictionary
(Array Dict) String -> Dict

**Arguments**:

  ----------
- `cycles` - (Array Dict)
- `name` - (optional) string or None, if a string, adds the "name" key to the output

<a id="fastsim.cycle.resample"></a>

#### resample

```python
def resample(cycle: Dict[str, Any],
             new_dt: Optional[float] = None,
             start_time: Optional[float] = None,
             end_time: Optional[float] = None,
             hold_keys: Optional[Set[str]] = None,
             hold_keys_next: Optional[Set[str]] = None,
             rate_keys: Optional[Set[str]] = None)
```

Cycle new_dt=?Real start_time=?Real end_time=?Real -> Cycle
Resample a cycle with a new delta time from start time to end time.

- cycle: Dict with keys
'time_s': numpy.array Real giving the elapsed time
- new_dt: Real, optional
the new delta time of the sampling. Defaults to the
difference between the first two times of the cycle passed in
- start_time: Real, optional
the start time of the sample. Defaults to 0.0 seconds
- end_time: Real, optional
the end time of the cycle. Defaults to the last time of the passed in
cycle.
- hold_keys: None or (Set String), if specified, yields values that
should be interpolated step-wise, holding their value until
an explicit change (i.e., NOT interpolated)
- hold_keys_next: None or (Set String), similar to hold_keys but yields
values that should be interpolated step-wise, taking the
NEXT value as the value (vs hold_keys which uses the previous)
- rate_keys: None or (Set String), if specified, yields values that maintain
the interpolated value of the given rate. So, for example,
if a speed, will set the speed such that the distance traveled
is consistent. Note: using rate keys for mps may result in
non-zero starting and ending speeds
Resamples all non-time metrics by the new sample time.

<a id="fastsim.cycle.clip_by_times"></a>

#### clip\_by\_times

```python
def clip_by_times(cycle, t_end, t_start=0)
```

Cycle Number Number -> Cycle
INPUT:
- cycle: Dict, a legitimate driving cycle
- t_start: Number, time to start
- t_end: Number, time to end
RETURNS: Dict, the cycle with fields snipped
    to times >= t_start and <= t_end
Clip the cycle to the given times and return

<a id="fastsim.cycle.accelerations"></a>

#### accelerations

```python
def accelerations(cycle)
```

Cycle -> Real
Return the acceleration of the given cycle
INPUTS:
- cycle: Dict, a legitimate driving cycle
OUTPUTS: Real, the maximum acceleration

<a id="fastsim.cycle.peak_acceleration"></a>

#### peak\_acceleration

```python
def peak_acceleration(cycle)
```

Cycle -> Real
Return the maximum acceleration of the given cycle
INPUTS:
- cycle: Dict, a legitimate driving cycle
OUTPUTS: Real, the maximum acceleration

<a id="fastsim.cycle.peak_deceleration"></a>

#### peak\_deceleration

```python
def peak_deceleration(cycle)
```

Cycle -> Real
Return the minimum acceleration (maximum deceleration) of the given cycle
INPUTS:
- cycle: Dict, a legitimate driving cycle
OUTPUTS: Real, the maximum acceleration

<a id="fastsim.cycle.calc_constant_jerk_trajectory"></a>

#### calc\_constant\_jerk\_trajectory

```python
def calc_constant_jerk_trajectory(n: int, D0: float, v0: float, Dr: float,
                                  vr: float, dt: float) -> tuple
```

Num Num Num Num Num Int -> (Tuple 'jerk_m__s3': Num, 'accel_m__s2': Num)
INPUTS:
- n: Int, number of time-steps away from rendezvous
- D0: Num, distance of simulated vehicle (m/s)
- v0: Num, speed of simulated vehicle (m/s)
- Dr: Num, distance of rendezvous point (m)
- vr: Num, speed of rendezvous point (m/s)
- dt: Num, step duration (s)
RETURNS: (Tuple 'jerk_m__s3': Num, 'accel_m__s2': Num)
Returns the constant jerk and acceleration for initial time step.

<a id="fastsim.cycle.accel_for_constant_jerk"></a>

#### accel\_for\_constant\_jerk

```python
def accel_for_constant_jerk(n, a0, k, dt)
```

Calculate the acceleration n timesteps away
INPUTS:
- n: Int, number of times steps away to calculate
- a0: Num, initial acceleration (m/s2)
- k: Num, constant jerk (m/s3)
- dt: Num, time-step duration in seconds
NOTE:
- this is the constant acceleration over the time-step from sample n to sample n+1
RETURN: Num, the acceleration n timesteps away (m/s2)

<a id="fastsim.cycle.speed_for_constant_jerk"></a>

#### speed\_for\_constant\_jerk

```python
def speed_for_constant_jerk(n, v0, a0, k, dt)
```

Int Num Num Num Num -> Num
Calculate speed (m/s) n timesteps away
INPUTS:
- n: Int, numer of timesteps away to calculate
- v0: Num, initial speed (m/s)
- a0: Num, initial acceleration (m/s2)
- k: Num, constant jerk
- dt: Num, duration of a timestep (s)
NOTE:
- this is the speed at sample n
- if n == 0, speed is v0
- if n == 1, speed is v0 + a0*dt, etc.
RETURN: Num, the speed n timesteps away (m/s)

<a id="fastsim.cycle.dist_for_constant_jerk"></a>

#### dist\_for\_constant\_jerk

```python
def dist_for_constant_jerk(n, d0, v0, a0, k, dt)
```

Calculate distance (m) after n timesteps
INPUTS:
- n: Int, numer of timesteps away to calculate
- d0: Num, initial distance (m)
- v0: Num, initial speed (m/s)
- a0: Num, initial acceleration (m/s2)
- k: Num, constant jerk
- dt: Num, duration of a timestep (s)
NOTE:
- this is the distance traveled from start (i.e., n=0) measured at sample point n
RETURN: Num, the distance at n timesteps away (m)

<a id="fastsim.cycle.detect_passing"></a>

#### detect\_passing

```python
def detect_passing(cyc: Cycle,
                   cyc0: Cycle,
                   i: int,
                   dist_tol_m: float = 0.1) -> PassingInfo
```

Reports back information of the first point where cyc passes cyc0, starting at
step i until the next stop of cyc.
- cyc: fastsim.Cycle, the proposed cycle of the vehicle under simulation
- cyc0: fastsim.Cycle, the reference/lead vehicle/shadow cycle to compare with
- i: int, the time-step index to consider
- dist_tol_m: float, the distance tolerance away from lead vehicle to be seen as
    "deviated" from the reference/shadow trace (m)
RETURNS: PassingInfo

<a id="fastsim.cycle.average_step_speeds"></a>

#### average\_step\_speeds

```python
def average_step_speeds(cyc: Cycle) -> np.ndarray
```

Calculate the average speed per each step in m/s

<a id="fastsim.cycle.average_step_speed_at"></a>

#### average\_step\_speed\_at

```python
def average_step_speed_at(cyc: Cycle, i: int) -> float
```

Calculate the average step speed at step i in m/s
(i.e., from sample point i-1 to i)

<a id="fastsim.cycle.trapz_step_distances"></a>

#### trapz\_step\_distances

```python
def trapz_step_distances(cyc: Cycle) -> np.ndarray
```

Sum of the distance traveled over each step using
trapezoidal integration

<a id="fastsim.cycle.trapz_step_start_distance"></a>

#### trapz\_step\_start\_distance

```python
def trapz_step_start_distance(cyc: Cycle, i: int) -> float
```

The distance traveled from start at the beginning of step i
(i.e., distance traveled up to sample point i-1)
Distance is in meters.

<a id="fastsim.cycle.trapz_distance_for_step"></a>

#### trapz\_distance\_for\_step

```python
def trapz_distance_for_step(cyc: Cycle, i: int) -> float
```

The distance traveled during step i in meters
(i.e., from sample point i-1 to i)

<a id="fastsim.cycle.trapz_distance_over_range"></a>

#### trapz\_distance\_over\_range

```python
def trapz_distance_over_range(cyc: Cycle, i_start: int, i_end: int) -> float
```

Calculate the distance from step i_start to the start of step i_end
(i.e., distance from sample point i_start-1 to i_end-1)

<a id="fastsim.cycle.extend_cycle"></a>

#### extend\_cycle

```python
def extend_cycle(cyc: Cycle,
                 absolute_time_s: float = 0.0,
                 time_fraction: float = 0.0,
                 use_rust: bool = False) -> Cycle
```

- cyc: fastsim.cycle.Cycle
- absolute_time_s: float, the seconds to extend
- time_fraction: float, the fraction of the original cycle time to add on
- use_rust: bool, if True, return a RustCycle instance, else a normal Python Cycle
RETURNS: fastsim.cycle.Cycle (or fastsimrust.RustCycle), the new cycle with stopped time appended
NOTE: additional time is rounded to the nearest second

<a id="fastsim.cycle.create_dist_and_target_speeds_by_microtrip"></a>

#### create\_dist\_and\_target\_speeds\_by\_microtrip

```python
def create_dist_and_target_speeds_by_microtrip(
        cyc: Cycle,
        blend_factor: float = 0.0,
        min_target_speed_mps: float = 8.0) -> list
```

Create distance and target speeds by microtrip
This helper function splits a cycle up into microtrips and returns a list of 2-tuples of:
(distance from start in meters, target speed in meters/second)

- cyc: the cycle to operate on
- blend_factor: float, from 0 to 1
    if 0, use average speed of the microtrip
    if 1, use average speed while moving (i.e., no stopped time)
    else something in between
- min_target_speed_mps: float, the minimum target speed allowed (m/s)
RETURN: list of 2-tuple of (float, float) representing the distance of start of
    each microtrip and target speed for that microtrip
NOTE: target speed per microtrip is not allowed to be below min_target_speed_mps

<a id="fastsim.cycle.copy_cycle"></a>

#### copy\_cycle

```python
def copy_cycle(
    cyc: Cycle,
    return_type: str = None,
    deep: bool = True
) -> Dict[str, np.ndarray] | Cycle | LegacyCycle | RustCycle
```

Returns copy of Cycle.

**Arguments**:

- `cyc` - instantianed Cycle or CycleJit
  return_type:
- `default` - infer from type of cyc
- `'dict'` - dict
- `'python'` - Cycle
- `'legacy'` - LegacyCycle
- `'rust'` - RustCycle
- `deep` - if True, uses deepcopy on everything

<a id="fastsim.vehicle_base"></a>

# fastsim.vehicle\_base

Boiler plate stuff needed for vehicle.py

<a id="fastsim.resample"></a>

# fastsim.resample

<a id="fastsim.resample.resample"></a>

#### resample

```python
def resample(df: pd.DataFrame,
             dt_new: Optional[float] = 1.0,
             time_col: Optional[str] = "Time[s]",
             rate_vars: Optional[Tuple[str]] = [],
             hold_vars: Optional[Tuple[str]] = []) -> pd.DataFrame
```

Resamples dataframe `df`.

**Arguments**:

  - df: dataframe to resample
  - dt_new: new time step size, default 1.0 s
  - time_col: column for time in s
  - rate_vars: list of variables that represent rates that need to be time averaged
  - hold_vars: vars that need zero-order hold from previous nearest time step
  (e.g. quantized variables like current gear)

<a id="fastsim.auxiliaries"></a>

# fastsim.auxiliaries

Auxiliary functions that require fastsim and provide faster access FASTSim vehicle properties.

<a id="fastsim.auxiliaries.R_air"></a>

#### R\_air

J/(kg*K)

<a id="fastsim.auxiliaries.abc_to_drag_coeffs"></a>

#### abc\_to\_drag\_coeffs

```python
def abc_to_drag_coeffs(veh: Vehicle,
                       a_lbf: float,
                       b_lbf__mph: float,
                       c_lbf__mph2: float,
                       custom_rho: bool = False,
                       custom_rho_temp_degC: float = 20.,
                       custom_rho_elevation_m: float = 180.,
                       simdrive_optimize: bool = True,
                       show_plots: bool = False,
                       use_rust=True) -> Tuple[float, float]
```

For a given vehicle and target A, B, and C
coefficients; calculate and return drag and rolling resistance
coefficients.

**Arguments**:

  ----------
- `veh` - vehicle.Vehicle with all parameters correct except for drag and rolling resistance coefficients
  a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]
- `custom_rho` - if True, use `fastsim.utilities.get_rho_air()` to calculate the current ambient density
- `custom_rho_temp_degC` - ambient temperature [degree C] for `get_rho_air()`;
  will only be used when `custom_rho` is True
- `custom_rho_elevation_m` - location elevation [degree C] for `get_rho_air()`;
  will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
- `simdrive_optimize` - if True, use `SimDrive` to optimize the drag and rolling resistance;
  otherwise, directly use target A, B, C to calculate the results
- `show_plots` - if True, plots are shown
- `use_rust` - if True, use rust implementation of drag coefficient calculation.

<a id="fastsim.auxiliaries.drag_coeffs_to_abc"></a>

#### drag\_coeffs\_to\_abc

```python
def drag_coeffs_to_abc(veh,
                       custom_rho: bool = False,
                       custom_rho_temp_degC: float = 20.,
                       custom_rho_elevation_m: float = 180.,
                       fit_with_curve: bool = False,
                       show_plots: bool = False) -> Tuple[float, float, float]
```

For a given vehicle mass, frontal area, dragCoef, and wheelRrCoef,
calculate and return ABCs.

**Arguments**:

  ----------
- `veh` - vehicle.Vehicle with correct drag and rolling resistance
- `custom_rho` - if True, use `fastsim.utilities.get_rho_air()` to calculate the current ambient density
- `custom_rho_temp_degC` - ambient temperature [degree C] for `get_rho_air()`; will only be used when `custom_rho` is True
- `custom_rho_elevation_m` - location elevation [degree C] for `get_rho_air()`; will only be used when `custom_rho` is True; default value is elevation of Chicago, IL
- `fit_with_curve` - if True, use `scipy.curve_fit` to get A, B, Cs; otherwise, directly calculate A, B, Cs from given drag and rolling resistance
- `show_plots` - if True, plots are shown
  

**Returns**:

  a_lbf, b_lbf__mph, c_lbf__mph2: coastdown coefficients for road load [lbf] vs speed [mph]

<a id="fastsim.simdrivelabel"></a>

# fastsim.simdrivelabel

Module containing classes and methods for calculating label fuel economy.

<a id="fastsim.simdrivelabel.get_label_fe"></a>

#### get\_label\_fe

```python
def get_label_fe(veh: vehicle.Vehicle,
                 full_detail: bool = False,
                 verbose: bool = False,
                 chg_eff: float = None,
                 use_rust=False)
```

Generates label fuel economy (FE) values for a provided vehicle.

**Arguments**:

  ----------
  veh : vehicle.Vehicle()
  full_detail : boolean, default False
  If True, sim_drive objects for each cycle are also returned.
  verbose : boolean, default false
  If true, print out key results
  chg_eff : float between 0 and 1
  Override for chg_eff -- currently not functional
- `use_rust` - bool, if True, use rust version of classes, else Python
  
  Returns label fuel economy values as a dict and (optionally)
  simdrive.SimDriveClassic objects.

<a id="fastsim.simdrive"></a>

# fastsim.simdrive

Module containing classes and methods for simulating vehicle drive cycle.

<a id="fastsim.simdrive.SimDriveParams"></a>

## SimDriveParams Objects

```python
class SimDriveParams(object)
```

Class containing attributes used for configuring sim_drive.
Usually the defaults are ok, and there will be no need to use this.

See comments in code for descriptions of various parameters that
affect simulation behavior. If, for example, you want to suppress
warning messages, use the following pastable code EXAMPLE:

>>> import logging
>>> logging.getLogger("fastsim").setLevel(logging.DEBUG)

<a id="fastsim.simdrive.SimDriveParams.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, sdp_dict)
```

Create from a dictionary

<a id="fastsim.simdrive.SimDriveParams.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Default values that affect simulation behavior.
Can be modified after instantiation.

<a id="fastsim.simdrive.SimDriveParams.to_rust"></a>

#### to\_rust

```python
def to_rust()
```

Change to the Rust version

<a id="fastsim.simdrive.SimDriveParams.reset_orphaned"></a>

#### reset\_orphaned

```python
def reset_orphaned()
```

Dummy method for flexibility between Rust/Python version interfaces

<a id="fastsim.simdrive.copy_sim_params"></a>

#### copy\_sim\_params

```python
def copy_sim_params(sdp: SimDriveParams, return_type: str = None)
```

Returns copy of SimDriveParams.

**Arguments**:

- `sdp` - instantianed SimDriveParams or RustSimDriveParams
  return_type:
- `default` - infer from type of sdp
- `'dict'` - dict
- `'python'` - SimDriveParams
- `'rust'` - RustSimDriveParams
- `deep` - if True, uses deepcopy on everything

<a id="fastsim.simdrive.sim_params_equal"></a>

#### sim\_params\_equal

```python
def sim_params_equal(a: SimDriveParams, b: SimDriveParams) -> bool
```

Returns True if objects are structurally equal (i.e., equal by value), else false.

**Arguments**:

- `a` - instantiated SimDriveParams object
- `b` - instantiated SimDriveParams object

<a id="fastsim.simdrive.SimDrive"></a>

## SimDrive Objects

```python
class SimDrive(object)
```

Class containing methods for running FASTSim vehicle
fuel economy simulations. This class is not compiled and will
run slower for large batch runs.

**Arguments**:

  ----------
- `cyc` - cycle.Cycle instance
- `veh` - vehicle.Vehicle instance

<a id="fastsim.simdrive.SimDrive.__init__"></a>

#### \_\_init\_\_

```python
def __init__(cyc: cycle.Cycle, veh: vehicle.Vehicle)
```

Initalizes arrays, given vehicle.Vehicle() and cycle.Cycle() as arguments.
sim_params is needed only if non-default behavior is desired.

<a id="fastsim.simdrive.SimDrive.gap_to_lead_vehicle_m"></a>

#### gap\_to\_lead\_vehicle\_m

```python
@property
def gap_to_lead_vehicle_m()
```

Provides the gap-with lead vehicle from start to finish

<a id="fastsim.simdrive.SimDrive.sim_drive"></a>

#### sim\_drive

```python
def sim_drive(init_soc: Optional[float] = None,
              aux_in_kw_override: Optional[np.ndarray] = None)
```

Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.
Arguments
------------
init_soc: initial SOC for electrified vehicles.  
aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
    Default of None causes veh.aux_kw to be used.

<a id="fastsim.simdrive.SimDrive.init_for_step"></a>

#### init\_for\_step

```python
def init_for_step(init_soc: float,
                  aux_in_kw_override: Optional[np.ndarray] = None)
```

This is a specialty method which should be called prior to using
sim_drive_step in a loop.

Arguments
------------
init_soc: initial battery state-of-charge (SOC) for electrified vehicles
aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
        Default of None causes veh.aux_kw to be used.

<a id="fastsim.simdrive.SimDrive.sim_drive_walk"></a>

#### sim\_drive\_walk

```python
def sim_drive_walk(init_soc: float,
                   aux_in_kw_override: Optional[np.ndarray] = None)
```

Receives second-by-second cycle information, vehicle properties, 
and an initial state of charge and runs sim_drive_step to perform a 
backward facing powertrain simulation. Method 'sim_drive' runs this
iteratively to achieve correct SOC initial and final conditions, as 
needed.

Arguments
------------
init_soc: initial battery state-of-charge (SOC) for electrified vehicles
aux_in_kw: aux_in_kw override.  Array of same length as cyc.time_s.  
        Default of None causes veh.aux_kw to be used.

<a id="fastsim.simdrive.SimDrive.activate_eco_cruise"></a>

#### activate\_eco\_cruise

```python
def activate_eco_cruise(by_microtrip: bool = False,
                        extend_fraction: float = 0.1,
                        blend_factor: float = 0.0,
                        min_target_speed_m_per_s: float = 8.0)
```

Sets the intelligent driver model parameters for an eco-cruise driving trajectory.
This is a convenience method instead of setting the sim_params.idm* parameters yourself.

- by_microtrip: bool, if True, target speed is set by microtrip, else by cycle
- extend_fraction: float, the fraction of time to extend the cycle to allow for catch-up
    of the following vehicle
- blend_factor: float, a value between 0 and 1; only used of by_microtrip is True, blends
    between microtrip average speed and microtrip average speed when moving. Must be
    between 0 and 1 inclusive

<a id="fastsim.simdrive.SimDrive.sim_drive_step"></a>

#### sim\_drive\_step

```python
def sim_drive_step()
```

Step through 1 time step.
TODO: create self.set_speed_for_target_gap(self.i):
TODO: consider implementing for battery SOC dependence

<a id="fastsim.simdrive.SimDrive.solve_step"></a>

#### solve\_step

```python
def solve_step(i)
```

Perform all the calculations to solve 1 time step.

<a id="fastsim.simdrive.SimDrive.set_misc_calcs"></a>

#### set\_misc\_calcs

```python
def set_misc_calcs(i)
```

Sets misc. calculations at time step 'i'

**Arguments**:

  ----------
- `i` - index of time step

<a id="fastsim.simdrive.SimDrive.set_comp_lims"></a>

#### set\_comp\_lims

```python
def set_comp_lims(i)
```

Sets component limits for time step 'i'
Arguments
------------
i: index of time step
init_soc: initial SOC for electrified vehicles

<a id="fastsim.simdrive.SimDrive.set_power_calcs"></a>

#### set\_power\_calcs

```python
def set_power_calcs(i)
```

Calculate power requirements to meet cycle and determine if
cycle can be met.  
Arguments
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_ach_speed"></a>

#### set\_ach\_speed

```python
def set_ach_speed(i)
```

Calculate actual speed achieved if vehicle hardware cannot achieve trace speed.
Arguments
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_hybrid_cont_calcs"></a>

#### set\_hybrid\_cont\_calcs

```python
def set_hybrid_cont_calcs(i)
```

Hybrid control calculations.
Arguments
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_fc_forced_state"></a>

#### set\_fc\_forced\_state

```python
def set_fc_forced_state(i)
```

Calculate control variables related to engine on/off state
Arguments       
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_hybrid_cont_decisions"></a>

#### set\_hybrid\_cont\_decisions

```python
def set_hybrid_cont_decisions(i)
```

Hybrid control decisions.
Arguments
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_fc_power"></a>

#### set\_fc\_power

```python
def set_fc_power(i)
```

Sets fcKwOutAch and fcKwInAch.
Arguments
------------
i: index of time step

<a id="fastsim.simdrive.SimDrive.set_post_scalars"></a>

#### set\_post\_scalars

```python
def set_post_scalars()
```

Sets scalar variables that can be calculated after a cycle is run.
This includes mpgge, various energy metrics, and others

<a id="fastsim.simdrive.SimDrive.to_rust"></a>

#### to\_rust

```python
def to_rust()
```

Create a rust version of SimDrive

<a id="fastsim.simdrive.copy_sim_drive"></a>

#### copy\_sim\_drive

```python
def copy_sim_drive(sd: SimDrive,
                   return_type: str = None,
                   deep: bool = True) -> SimDrive
```

Returns copy of SimDriveClassic or SimDriveJit as SimDriveClassic.

**Arguments**:

  ----------
- `sd` - instantiated SimDriveClassic or SimDriveJit
- `return_type` - default, 'python', 'legacy', or 'rust'
  - default: infer from type of sd
  - 'python': Cycle
  - 'legacy': LegacyCycle
  - 'rust': RustCycle
- `deep` - if True, uses deepcopy on everything

<a id="fastsim.simdrive.sim_drive_equal"></a>

#### sim\_drive\_equal

```python
def sim_drive_equal(a: SimDrive, b: SimDrive) -> bool
```



<a id="fastsim.simdrive.run_simdrive_for_accel_test"></a>

#### run\_simdrive\_for\_accel\_test

```python
def run_simdrive_for_accel_test(sd: SimDrive)
```

Initialize and run sim_drive_walk as appropriate for vehicle attribute vehPtType.

<a id="fastsim.simdrive.SimDrivePost"></a>

## SimDrivePost Objects

```python
class SimDrivePost(object)
```

Class for post-processing of SimDrive instance.  Requires already-run
SimDrive instance.

<a id="fastsim.simdrive.SimDrivePost.__init__"></a>

#### \_\_init\_\_

```python
def __init__(sim_drive: SimDrive)
```

**Arguments**:

  ---------------
- `sim_drive` - solved sim_drive object

<a id="fastsim.simdrive.SimDrivePost.get_diagnostics"></a>

#### get\_diagnostics

```python
def get_diagnostics()
```

This method is to be run after runing sim_drive if diagnostic variables
are needed.  Diagnostic variables are returned in a dict.  Diagnostic variables include:
- final integrated value of all positive powers
- final integrated value of all negative powers
- total distance traveled
- miles per gallon gasoline equivalent (mpgge)

<a id="fastsim.simdrive.SimDrivePost.set_battery_wear"></a>

#### set\_battery\_wear

```python
def set_battery_wear()
```

Battery wear calcs

<a id="fastsim.simdrive.SimDriveJit"></a>

#### SimDriveJit

```python
def SimDriveJit(cyc_jit, veh_jit)
```

deprecated

<a id="fastsim.simdrive.estimate_soc_corrected_fuel_kJ"></a>

#### estimate\_soc\_corrected\_fuel\_kJ

```python
def estimate_soc_corrected_fuel_kJ(sd: SimDrive) -> float
```

- sd: SimDriveClassic, the simdrive instance after simulation
RETURN: number, the kJ of fuel corrected for SOC imbalance

<a id="fastsim.tests.test_utils"></a>

# fastsim.tests.test\_utils

<a id="fastsim.tests.test_simdrive"></a>

# fastsim.tests.test\_simdrive

Test suite for simdrive instantiation and usage.

<a id="fastsim.tests.test_simdrive.TestSimDriveClassic"></a>

## TestSimDriveClassic Objects

```python
class TestSimDriveClassic(unittest.TestCase)
```

Tests for fastsim.simdrive.SimDriveClassic methods

<a id="fastsim.tests.test_simdrive.TestSimDriveClassic.test_sim_drive_step"></a>

#### test\_sim\_drive\_step

```python
def test_sim_drive_step()
```

Verify that sim_drive_step produces an expected result.

<a id="fastsim.tests.test_simdrive.TestSimDriveClassic.test_sim_drive_walk"></a>

#### test\_sim\_drive\_walk

```python
def test_sim_drive_walk()
```

Verify that sim_drive_walk produces an expected result.

<a id="fastsim.tests.test_logging"></a>

# fastsim.tests.test\_logging

<a id="fastsim.tests.test_following"></a>

# fastsim.tests.test\_following

Tests that check the drive cycle modification functionality.

<a id="fastsim.tests.test_following.TestFollowing"></a>

## TestFollowing Objects

```python
class TestFollowing(unittest.TestCase)
```

<a id="fastsim.tests.test_following.TestFollowing.test_that_we_have_a_gap_between_us_and_the_lead_vehicle"></a>

#### test\_that\_we\_have\_a\_gap\_between\_us\_and\_the\_lead\_vehicle

```python
def test_that_we_have_a_gap_between_us_and_the_lead_vehicle()
```

A positive gap should exist between us and the lead vehicle

<a id="fastsim.tests.test_following.TestFollowing.test_that_the_gap_changes_over_the_cycle"></a>

#### test\_that\_the\_gap\_changes\_over\_the\_cycle

```python
def test_that_the_gap_changes_over_the_cycle()
```

Ensure that our gap calculation is doing something

<a id="fastsim.tests.test_following.TestFollowing.test_that_following_works_over_parameter_sweep"></a>

#### test\_that\_following\_works\_over\_parameter\_sweep

```python
def test_that_following_works_over_parameter_sweep()
```

We're going to sweep through all of the parameters and see how it goes

<a id="fastsim.tests.test_following.TestFollowing.test_that_we_can_use_the_idm"></a>

#### test\_that\_we\_can\_use\_the\_idm

```python
def test_that_we_can_use_the_idm()
```

Tests use of the IDM model for following

<a id="fastsim.tests.test_following.TestFollowing.test_sweeping_idm_parameters"></a>

#### test\_sweeping\_idm\_parameters

```python
def test_sweeping_idm_parameters()
```

Tests use of the IDM model for following

<a id="fastsim.tests.test_following.TestFollowing.test_distance_based_grade_on_following"></a>

#### test\_distance\_based\_grade\_on\_following

```python
def test_distance_based_grade_on_following()
```

Tests use of the IDM model for following

<a id="fastsim.tests.test_vehicle"></a>

# fastsim.tests.test\_vehicle

Test suite for cycle instantiation and manipulation.

<a id="fastsim.tests.test_vehicle.TestVehicle"></a>

## TestVehicle Objects

```python
class TestVehicle(unittest.TestCase)
```

<a id="fastsim.tests.test_vehicle.TestVehicle.test_equal"></a>

#### test\_equal

```python
def test_equal()
```

Verify that a copied Vehicle and original are equal.

<a id="fastsim.tests.test_vehicle.TestVehicle.test_properties"></a>

#### test\_properties

```python
def test_properties()
```

Verify that some of the property variables are working as expected.

<a id="fastsim.tests.test_vehicle.TestVehicle.test_fc_efficiency_override"></a>

#### test\_fc\_efficiency\_override

```python
def test_fc_efficiency_override()
```

Verify that we can scale FC

<a id="fastsim.tests.test_vehicle.TestVehicle.test_set_derived_init"></a>

#### test\_set\_derived\_init

```python
def test_set_derived_init()
```

Verify that we can set derived parameters or not on init.

<a id="fastsim.tests.test_copy"></a>

# fastsim.tests.test\_copy

Test various copy utilities

<a id="fastsim.tests.test_copy.TestCopy"></a>

## TestCopy Objects

```python
class TestCopy(unittest.TestCase)
```

<a id="fastsim.tests.test_copy.TestCopy.test_copy_cycle"></a>

#### test\_copy\_cycle

```python
def test_copy_cycle()
```

Test that cycle_copy works as expected

<a id="fastsim.tests.test_copy.TestCopy.test_copy_physical_properties"></a>

#### test\_copy\_physical\_properties

```python
def test_copy_physical_properties()
```

Test that copy_physical_properties works as expected

<a id="fastsim.tests.test_copy.TestCopy.test_copy_vehicle"></a>

#### test\_copy\_vehicle

```python
def test_copy_vehicle()
```

Test that vehicle_copy works as expected

<a id="fastsim.tests.test_copy.TestCopy.test_copy_sim_params"></a>

#### test\_copy\_sim\_params

```python
def test_copy_sim_params()
```

Test that copy_sim_params works as expected

<a id="fastsim.tests.test_copy.TestCopy.test_copy_sim_drive"></a>

#### test\_copy\_sim\_drive

```python
def test_copy_sim_drive()
```

Test that copy_sim_drive works as expected

<a id="fastsim.tests"></a>

# fastsim.tests

Package containing tests for FASTSim.

<a id="fastsim.tests.run_functional_tests"></a>

#### run\_functional\_tests

```python
def run_functional_tests()
```

Runs all functional tests.

<a id="fastsim.tests.test_auxiliaries"></a>

# fastsim.tests.test\_auxiliaries

<a id="fastsim.tests.test_simdrivelabel"></a>

# fastsim.tests.test\_simdrivelabel

<a id="fastsim.tests.test_simdrive_sweep"></a>

# fastsim.tests.test\_simdrive\_sweep

Test script that saves results from 26 vehicles currently in master branch of FASTSim as of 17 December 2019 for 3 standard cycles.
From command line, pass True (default if left blank) or False argument to use JIT compilation or not, respectively.

<a id="fastsim.tests.test_simdrive_sweep.main"></a>

#### main

```python
def main(err_tol=1e-4, verbose=True, use_rust=False)
```

Runs test test for 26 vehicles and 3 cycles.
Test compares cumulative positive and negative energy
values to a benchmark from earlier.

**Arguments**:

  ----------
  err_tol : error tolerance
  default of 1e-4 was selected to prevent minor errors from showing.
  As of 31 December 2020, a recent python update caused errors that
  are smaller than this and therefore ok to neglect.
- `verbose` - if True, prints progress
- `use_rust` - Boolean, if True, use Rust version of classes, else python version
  

**Returns**:

  --------
  df_err : pandas datafram, fractional errors
  df : pandas dataframe, new values
  df0 : pandas dataframe, original benchmark values
- `col_for_max_error` - string or None, the column name of the column having max absolute error
- `max_abs_err` - number or None, the maximum absolute error if it exists

<a id="fastsim.tests.test_simdrive_sweep.TestSimDriveSweep"></a>

## TestSimDriveSweep Objects

```python
class TestSimDriveSweep(unittest.TestCase)
```

<a id="fastsim.tests.test_simdrive_sweep.TestSimDriveSweep.test_sweep"></a>

#### test\_sweep

```python
def test_sweep()
```

Compares results against benchmark.

<a id="fastsim.tests.test_cycle"></a>

# fastsim.tests.test\_cycle

Test suite for cycle instantiation and manipulation.

<a id="fastsim.tests.test_cycle.calc_distance_traveled_m"></a>

#### calc\_distance\_traveled\_m

```python
def calc_distance_traveled_m(cyc, up_to=None)
```

Calculate the distance traveled in meters
- cyc: a cycle dictionary
- up_to: None or a positive number indicating a time in seconds. Will calculate the distance up-to that given time
RETURN: Number, the distance traveled in meters

<a id="fastsim.tests.test_cycle.dicts_are_equal"></a>

#### dicts\_are\_equal

```python
def dicts_are_equal(d1, d2, d1_name=None, d2_name=None)
```

Checks if dictionaries are equal
- d1: dict
- d2: dict
- d1_name: None or string, the name used for dict 1 in messaging
- d2_name: None or string, the name used for dict 1 in messaging
RETURN: (boolean, (Array string)),
Returns (True, []) if the dictionaries are equal; otherwise, returns
(False, [... list of issues here])

<a id="fastsim.tests.test_cycle.TestCycle"></a>

## TestCycle Objects

```python
class TestCycle(unittest.TestCase)
```

<a id="fastsim.tests.test_cycle.TestCycle.test_monotonicity"></a>

#### test\_monotonicity

```python
def test_monotonicity()
```

checks that time is monotonically increasing

<a id="fastsim.tests.test_cycle.TestCycle.test_load_dict"></a>

#### test\_load\_dict

```python
def test_load_dict()
```

checks that conversion from dict works

<a id="fastsim.tests.test_cycle.TestCycle.test_that_udds_has_18_microtrips"></a>

#### test\_that\_udds\_has\_18\_microtrips

```python
def test_that_udds_has_18_microtrips()
```

Check that the number of microtrips equals expected

<a id="fastsim.tests.test_cycle.TestCycle.test_roundtrip_of_microtrip_and_concat"></a>

#### test\_roundtrip\_of\_microtrip\_and\_concat

```python
def test_roundtrip_of_microtrip_and_concat()
```

A cycle split into microtrips and concatenated back together should equal the original

<a id="fastsim.tests.test_cycle.TestCycle.test_roundtrip_of_microtrip_and_concat_using_keep_name_arg"></a>

#### test\_roundtrip\_of\_microtrip\_and\_concat\_using\_keep\_name\_arg

```python
def test_roundtrip_of_microtrip_and_concat_using_keep_name_arg()
```

A cycle split into microtrips and concatenated back together should equal the original

<a id="fastsim.tests.test_cycle.TestCycle.test_set_from_dict_for_a_microtrip"></a>

#### test\_set\_from\_dict\_for\_a\_microtrip

```python
def test_set_from_dict_for_a_microtrip()
```

Test splitting into microtrips and setting is as expected

<a id="fastsim.tests.test_cycle.TestCycle.test_duration_of_concatenated_cycles_is_the_sum_of_the_components"></a>

#### test\_duration\_of\_concatenated\_cycles\_is\_the\_sum\_of\_the\_components

```python
def test_duration_of_concatenated_cycles_is_the_sum_of_the_components()
```

Test that two cycles concatenated have the same duration as the sum of the constituents

<a id="fastsim.tests.test_cycle.TestCycle.test_cycle_equality"></a>

#### test\_cycle\_equality

```python
def test_cycle_equality()
```

Test structural equality of driving cycles

<a id="fastsim.tests.test_cycle.TestCycle.test_that_cycle_resampling_works_as_expected"></a>

#### test\_that\_cycle\_resampling\_works\_as\_expected

```python
def test_that_cycle_resampling_works_as_expected()
```

Test resampling the values of a cycle

<a id="fastsim.tests.test_cycle.TestCycle.test_resampling_and_concatenating_cycles"></a>

#### test\_resampling\_and\_concatenating\_cycles

```python
def test_resampling_and_concatenating_cycles()
```

Test that concatenating cycles at different sampling rates works as expected

<a id="fastsim.tests.test_cycle.TestCycle.test_resampling_with_hold_keys"></a>

#### test\_resampling\_with\_hold\_keys

```python
def test_resampling_with_hold_keys()
```

Test that 'hold_keys' works with resampling

<a id="fastsim.tests.test_cycle.TestCycle.test_that_resampling_preserves_total_distance_traveled_using_rate_keys"></a>

#### test\_that\_resampling\_preserves\_total\_distance\_traveled\_using\_rate\_keys

```python
def test_that_resampling_preserves_total_distance_traveled_using_rate_keys()
```

Distance traveled before and after resampling should be the same when rate_keys are used

<a id="fastsim.tests.test_cycle.TestCycle.test_clip_by_times"></a>

#### test\_clip\_by\_times

```python
def test_clip_by_times()
```

Test that clipping by times works as expected

<a id="fastsim.tests.test_cycle.TestCycle.test_get_accelerations"></a>

#### test\_get\_accelerations

```python
def test_get_accelerations()
```

Test getting and processing accelerations

<a id="fastsim.tests.test_cycle.TestCycle.test_that_copy_creates_idential_structures"></a>

#### test\_that\_copy\_creates\_idential\_structures

```python
def test_that_copy_creates_idential_structures()
```

Checks that copy methods produce identical cycles

<a id="fastsim.tests.test_cycle.TestCycle.test_make_cycle"></a>

#### test\_make\_cycle

```python
def test_make_cycle()
```

Check that make_cycle works as expected

<a id="fastsim.tests.test_cycle.TestCycle.test_key_conversion"></a>

#### test\_key\_conversion

```python
def test_key_conversion()
```

check that legacy keys can still be generated

<a id="fastsim.tests.test_cycle.TestCycle.test_get_grade_by_distance"></a>

#### test\_get\_grade\_by\_distance

```python
def test_get_grade_by_distance()
```

check that we can lookup grade by distance

<a id="fastsim.tests.test_cycle.TestCycle.test_dt_s_vs_dt_s_at_i"></a>

#### test\_dt\_s\_vs\_dt\_s\_at\_i

```python
def test_dt_s_vs_dt_s_at_i()
```

Test that dt_s_at_i is a true replacement for dt_s[i]

<a id="fastsim.tests.test_cycle.TestCycle.test_trapz_step_start_distance"></a>

#### test\_trapz\_step\_start\_distance

```python
def test_trapz_step_start_distance()
```

Test the implementation of trapz_step_start_distance

<a id="fastsim.tests.test_cycle.TestCycle.test_that_cycle_cache_interp_grade_substitutes_for_average_grade_over_range"></a>

#### test\_that\_cycle\_cache\_interp\_grade\_substitutes\_for\_average\_grade\_over\_range

```python
def test_that_cycle_cache_interp_grade_substitutes_for_average_grade_over_range(
)
```

Ensure that CycleCache.interp_grade actually predicts the same values as
Cycle.average_grade_over_range(d, 0.0, cache=None|CycleCache) with and without
using CycleCache

<a id="fastsim.tests.test_cycle.TestCycle.test_that_trapz_step_start_distance_equals_cache_trapz_distances"></a>

#### test\_that\_trapz\_step\_start\_distance\_equals\_cache\_trapz\_distances

```python
def test_that_trapz_step_start_distance_equals_cache_trapz_distances()
```

Test that cycle.trapz_step_start_distance(self.cyc0, i) == self._cyc0_cache.trapz_distances_m[i-1]

<a id="fastsim.tests.test_cycle.TestCycle.test_average_grade_over_range_with_and_without_cache"></a>

#### test\_average\_grade\_over\_range\_with\_and\_without\_cache

```python
def test_average_grade_over_range_with_and_without_cache()
```

Ensure that CycleCache usage only speeds things up; doesn't change values...

<a id="fastsim.tests.test_cav_sweep"></a>

# fastsim.tests.test\_cav\_sweep

Test fastsim/demos/cav_sweep.py for regressions

<a id="fastsim.tests.test_soc_correction"></a>

# fastsim.tests.test\_soc\_correction

Tests an HEV correction methodology versus other techniques

<a id="fastsim.tests.test_soc_correction.TestSocCorrection"></a>

## TestSocCorrection Objects

```python
class TestSocCorrection(unittest.TestCase)
```

<a id="fastsim.tests.test_soc_correction.TestSocCorrection.test_that_soc_correction_method_works"></a>

#### test\_that\_soc\_correction\_method\_works

```python
def test_that_soc_correction_method_works()
```

Test using an SOC equivalency method versus other techniques

<a id="fastsim.tests.test_coasting"></a>

# fastsim.tests.test\_coasting

Tests that check the drive cycle modification functionality.

<a id="fastsim.tests.test_coasting.make_coasting_plot"></a>

#### make\_coasting\_plot

```python
def make_coasting_plot(
        cyc0: fastsim.cycle.Cycle,
        cyc: fastsim.cycle.Cycle,
        use_mph: bool = False,
        title: Optional[str] = None,
        save_file: Optional[str] = None,
        do_show: bool = False,
        verbose: bool = False,
        gap_offset_m: float = 0.0,
        coast_brake_start_speed_m_per_s: Optional[float] = None)
```

- cyc0: Cycle, the reference cycle (the "shadow trace" or "lead vehicle")
- cyc: Cycle, the actual cycle driven
- use_mph: Bool, if True, plot in miles per hour, else m/s
- title: None or string, if string, set the title
- save_file: (Or None string), if specified, save the file to disk
- do_show: Bool, whether to show the file or not
- verbose: Bool, if True, prints out
- gap_offset_m: number, an offset to apply to the gap metrics (m)
- coast_brake_start_speed_m_per_s: None | number, if supplied, plots the coast-start speed (m/s)
RETURN: None
- saves creates the given file and shows it

<a id="fastsim.tests.test_coasting.make_dvdd_plot"></a>

#### make\_dvdd\_plot

```python
def make_dvdd_plot(cyc: fastsim.cycle.Cycle,
                   coast_to_break_speed_m__s: Union[float, None] = None,
                   use_mph: bool = False,
                   save_file: Union[None, str] = None,
                   do_show: bool = False,
                   curve_fit: bool = True,
                   additional_xs: Union[None, List[float]] = None,
                   additional_ys: Union[None, List[float]] = None)
```

Create a change in speed (dv) by change in distance (dd) plot

<a id="fastsim.tests.test_coasting.TestCoasting"></a>

## TestCoasting Objects

```python
class TestCoasting(unittest.TestCase)
```

<a id="fastsim.tests.test_coasting.TestCoasting.test_cycle_reported_distance_traveled_m"></a>

#### test\_cycle\_reported\_distance\_traveled\_m

```python
def test_cycle_reported_distance_traveled_m()
```



<a id="fastsim.tests.test_coasting.TestCoasting.test_cycle_modifications_with_constant_jerk"></a>

#### test\_cycle\_modifications\_with\_constant\_jerk

```python
def test_cycle_modifications_with_constant_jerk()
```



<a id="fastsim.tests.test_coasting.TestCoasting.test_that_cycle_modifications_work_as_expected"></a>

#### test\_that\_cycle\_modifications\_work\_as\_expected

```python
def test_that_cycle_modifications_work_as_expected()
```



<a id="fastsim.tests.test_coasting.TestCoasting.test_that_we_can_coast"></a>

#### test\_that\_we\_can\_coast

```python
def test_that_we_can_coast()
```

Test the standard interface to Eco-Approach for 'free coasting'

<a id="fastsim.tests.test_coasting.TestCoasting.test_eco_approach_modeling"></a>

#### test\_eco\_approach\_modeling

```python
def test_eco_approach_modeling()
```

Test a simplified model of eco-approach

<a id="fastsim.tests.test_coasting.TestCoasting.test_consistency_of_constant_jerk_trajectory"></a>

#### test\_consistency\_of\_constant\_jerk\_trajectory

```python
def test_consistency_of_constant_jerk_trajectory()
```

Confirm that acceleration, speed, and distances are as expected for constant jerk trajectory

<a id="fastsim.tests.test_coasting.TestCoasting.test_that_final_speed_of_cycle_modification_matches_trajectory_calcs"></a>

#### test\_that\_final\_speed\_of\_cycle\_modification\_matches\_trajectory\_calcs

```python
def test_that_final_speed_of_cycle_modification_matches_trajectory_calcs()
```



<a id="fastsim.tests.test_coasting.TestCoasting.test_that_cycle_distance_reported_is_correct"></a>

#### test\_that\_cycle\_distance\_reported\_is\_correct

```python
def test_that_cycle_distance_reported_is_correct()
```

Test the reported distances via cycDistMeters

<a id="fastsim.tests.test_coasting.TestCoasting.test_brake_trajectory"></a>

#### test\_brake\_trajectory

```python
def test_brake_trajectory()
```



<a id="fastsim.tests.test_coasting.TestCoasting.test_logic_to_enter_eco_approach_automatically"></a>

#### test\_logic\_to\_enter\_eco\_approach\_automatically

```python
def test_logic_to_enter_eco_approach_automatically()
```

Test that we can auto-enter eco-approach

<a id="fastsim.tests.test_coasting.TestCoasting.test_that_coasting_works_going_uphill"></a>

#### test\_that\_coasting\_works\_going\_uphill

```python
def test_that_coasting_works_going_uphill()
```

Test coasting logic while hill climbing

<a id="fastsim.tests.test_coasting.TestCoasting.test_that_coasting_logic_works_going_uphill"></a>

#### test\_that\_coasting\_logic\_works\_going\_uphill

```python
def test_that_coasting_logic_works_going_uphill()
```

When going uphill, we want to ensure we can still hit our coasting target

<a id="fastsim.tests.test_coasting.TestCoasting.test_that_coasting_logic_works_going_downhill"></a>

#### test\_that\_coasting\_logic\_works\_going\_downhill

```python
def test_that_coasting_logic_works_going_downhill()
```

When going downhill, ensure we can still hit our coasting target

<a id="fastsim.tests.test_coasting.TestCoasting.test_that_coasting_works_with_multiple_stops_and_grades"></a>

#### test\_that\_coasting\_works\_with\_multiple\_stops\_and\_grades

```python
def test_that_coasting_works_with_multiple_stops_and_grades()
```

Ensure coasting hits distance target with multiple stops and both uphill/downhill

<a id="fastsim.tests.test_rust"></a>

# fastsim.tests.test\_rust

Tests using the Rust versions of SimDrive, Cycle, and Vehicle

<a id="fastsim.tests.test_rust.TestRust"></a>

## TestRust Objects

```python
class TestRust(unittest.TestCase)
```

<a id="fastsim.tests.test_rust.TestRust.test_discrepancies"></a>

#### test\_discrepancies

```python
def test_discrepancies(veh_type="ALL", use_dict=True, cyc_name="udds")
```

Function for testing for Rust/Python discrepancies, both in the vehicle database
CSV as well as the individual model files. Uses test_vehicle_for_discrepancies as backend.

**Arguments**:

- `veh_type` - type of vehicle to test for discrepancies
  can be "CONV", "HEV", "PHEV", "BEV", or "ALL"
- `use_dict` - if True, use small cyc_dict to speed up test
  if false, default to UDDS
- `cyc_name` - name of cycle from database to use if use_dict == False

<a id="fastsim.tests.test_rust.TestRust.test_vehicle_for_discrepancies"></a>

#### test\_vehicle\_for\_discrepancies

```python
def test_vehicle_for_discrepancies(vnum=1,
                                   veh_filename=None,
                                   cyc_dict=None,
                                   cyc_name="udds")
```

Test for finding discrepancies between Rust and Python for single vehicle.

**Arguments**:

- `vnum` - vehicle database number, optional, default option without any arguments
- `veh_filename` - vehicle filename from vehdb folder, optional
- `cyc_dict` - cycle dictionary for custom cycle, optional
- `cyc_name` - cycle name from cycle database, optional

<a id="fastsim.tests.test_rust.TestRust.test_fueling_prediction_for_multiple_vehicle"></a>

#### test\_fueling\_prediction\_for\_multiple\_vehicle

```python
def test_fueling_prediction_for_multiple_vehicle()
```

This test assures that Rust and Python agree on at least one 
example of all permutations of veh_pt_type and fc_eff_type.

<a id="fastsim.tests.test_vs_excel"></a>

# fastsim.tests.test\_vs\_excel

Module for comparing python results with Excel by running all the vehicles
in both Excel (uses archived results if Excel version not available) and 
Python FASTSim for both UDDS and HWFET cycles.

<a id="fastsim.tests.test_vs_excel.run"></a>

#### run

```python
def run(vehicles=np.arange(1, 27), verbose=True, use_rust=False)
```

Runs python fastsim through 26 vehicles and returns list of dictionaries
containing scenario descriptions.

**Arguments**:

  **********
  verbose : Boolean
  if True, print progress
- `use_rust` - Boolean, if True, use Rust versions of classes

<a id="fastsim.tests.test_vs_excel.run_excel"></a>

#### run\_excel

```python
def run_excel(
        vehicles=np.arange(1, 28), prev_res_path=PREV_RES_PATH,
        rerun_excel=False)
```

Runs excel fastsim through 26 vehicles and returns list of dictionaries
containing scenario descriptions.

**Arguments**:

  -----------
  prev_res_path : path (str) to prevous results in pickle (*.p) file
  rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open

<a id="fastsim.tests.test_vs_excel.compare"></a>

#### compare

```python
def compare(res_python, res_excel, err_tol=0.001, verbose=True)
```

Finds common vehicle names in both excel and python
(hypothetically all of them, but there may be discrepancies) and then compares
fuel economy results.
Arguments: results from run_python and run_excel
Returns dict of comparsion results.

**Arguments**:

  ----------
  res_python : output of run_python
  res_excel : output of run_excel
  err_tol : (float) error tolerance, default=1e-3
  verbose : Boolean
  if True, print progress

<a id="fastsim.tests.test_vs_excel.main"></a>

#### main

```python
def main(err_tol=0.001,
         prev_res_path=PREV_RES_PATH,
         rerun_excel=False,
         verbose=False)
```

Function for running both python and excel and then comparing

**Arguments**:

  **********
  err_tol : (float) error tolerance, default=1e-3
  prev_res_path : path (str) to prevous results in pickle (*.p) file
  rerun_excel : (Boolean) if True, re-runs Excel FASTSim, which must be open
  verbose : Boolean
  if True, print progress

<a id="fastsim.tests.test_vs_excel.TestExcel"></a>

## TestExcel Objects

```python
class TestExcel(unittest.TestCase)
```

<a id="fastsim.tests.test_vs_excel.TestExcel.test_vs_excel"></a>

#### test\_vs\_excel

```python
def test_vs_excel()
```

Compares results against archived Excel results.

<a id="fastsim.tests.test_eco_cruise"></a>

# fastsim.tests.test\_eco\_cruise

Test the eco-cruise feature in FASTSim

<a id="fastsim.vehicle"></a>

# fastsim.vehicle

Module containing classes and methods for for loading vehicle data.

<a id="fastsim.vehicle.clean_data"></a>

#### clean\_data

```python
def clean_data(raw_data)
```

Cleans up data formatting.
Argument:
------------
raw_data: cell of vehicle dataframe

Output:
clean_data: cleaned up data

<a id="fastsim.vehicle.Vehicle"></a>

## Vehicle Objects

```python
@dataclass
class Vehicle(object)
```

Class for loading and contaning vehicle attributes
See `from_vehdb`, `from_file`, and `from_dict` methods for usage instructions.

<a id="fastsim.vehicle.Vehicle.from_vehdb"></a>

#### from\_vehdb

```python
@classmethod
def from_vehdb(cls,
               vnum: int,
               veh_file: str = None,
               to_rust: bool = False) -> Self
```

Load vehicle `vnum` from default vehdb or `veh_file`.

**Arguments**:

- `vnum` - vehicle number
- `veh_file` - path to vehicle database file
- `to_rust` - if True, convert to rust-compatible vehicle

<a id="fastsim.vehicle.Vehicle.from_file"></a>

#### from\_file

```python
@classmethod
def from_file(cls,
              filename: str,
              vnum: int = None,
              to_rust: bool = False) -> Self
```

Loads vehicle from file `filename` (str).  Looks in working dir and then
fastsim/resources/vehdb, which also contains numerous examples of vehicle csv files.
`vnum` is needed for multi-vehicle files.

**Arguments**:

- `filename` - path to vehicle database file
- `vnum` - vehicle number
- `to_rust` - if True, convert to rust-compatible vehicle

<a id="fastsim.vehicle.Vehicle.from_df"></a>

#### from\_df

```python
@classmethod
def from_df(cls,
            vehdf: pd.DataFrame,
            vnum: int,
            veh_file: Path,
            to_rust: bool = False) -> Self
```

Given vehdf, generates dict to feed to `from_dict`.

**Arguments**:

- `vehdf` - pandas dataframe of vehicle attributes
- `vnum` - vehicle number
- `veh_file` - path to vehicle database file
- `to_rust` - if True, convert to rust-compatible vehicle

<a id="fastsim.vehicle.Vehicle.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, veh_dict: dict, to_rust: bool = False) -> Self
```

Load vehicle from dict with snake_case key names.

**Arguments**:

- `veh_dict` - dict of vehicle attributes
- `to_rust` - if True, convert to rust-compatible vehicle

<a id="fastsim.vehicle.Vehicle.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__(converted_to_rust: bool = False)
```

Sets derived parameters.

**Arguments**:

  ----------
- `fc_peak_eff_override` - float (0, 1) or -1, if provided and not -1, overrides engine peak efficiency
  with proportional scaling.  Default of -1 has no effect.
- `mc_peak_eff_override` - float (0, 1) or -1, if provided and not -1, overrides motor peak efficiency
  with proportional scaling.  Default of -1 has no effect.

<a id="fastsim.vehicle.Vehicle.set_derived"></a>

#### set\_derived

```python
def set_derived()
```

Sets derived parameters.

**Arguments**:

  ----------
- `fc_peak_eff_override` - float (0, 1) or -1, if provided and not -1, overrides engine peak efficiency
  with proportional scaling.  Default of -1 has no effect.
- `mc_peak_eff_override` - float (0, 1) or -1, if provided and not -1, overrides motor peak efficiency
  with proportional scaling.  Default of -1 has no effect.

<a id="fastsim.vehicle.Vehicle.set_veh_mass"></a>

#### set\_veh\_mass

```python
def set_veh_mass()
```

Calculate total vehicle mass.  Sum up component masses if
positive real number is not specified for self.veh_override_kg

<a id="fastsim.vehicle.Vehicle.veh_type_selection"></a>

#### veh\_type\_selection

```python
@property
def veh_type_selection() -> str
```

Copying veh_pt_type to additional key
to be consistent with Excel version but not used in Python version

<a id="fastsim.vehicle.Vehicle.get_mcPeakEff"></a>

#### get\_mcPeakEff

```python
def get_mcPeakEff() -> float
```

Return `np.max(self.mc_eff_array)`

<a id="fastsim.vehicle.Vehicle.set_mcPeakEff"></a>

#### set\_mcPeakEff

```python
def set_mcPeakEff(new_peak)
```

Set motor peak efficiency EVERYWHERE.

**Arguments**:

  ----------
- `new_peak` - float, new peak motor efficiency in decimal form

<a id="fastsim.vehicle.Vehicle.get_fcPeakEff"></a>

#### get\_fcPeakEff

```python
def get_fcPeakEff() -> float
```

Return `np.max(self.fc_eff_array)`

<a id="fastsim.vehicle.Vehicle.set_fcPeakEff"></a>

#### set\_fcPeakEff

```python
def set_fcPeakEff(new_peak)
```

Set fc peak efficiency EVERWHERE.

**Arguments**:

  ----------
- `new_peak` - float, new peak fc efficiency in decimal form

<a id="fastsim.vehicle.Vehicle.get_numba_veh"></a>

#### get\_numba\_veh

```python
def get_numba_veh()
```

Deprecated.

<a id="fastsim.vehicle.Vehicle.to_rust"></a>

#### to\_rust

```python
def to_rust() -> RustVehicle
```

Return a Rust version of the vehicle

<a id="fastsim.vehicle.Vehicle.reset_orphaned"></a>

#### reset\_orphaned

```python
def reset_orphaned()
```

Dummy method for flexibility between Rust/Python version interfaces

<a id="fastsim.vehicle.LegacyVehicle"></a>

## LegacyVehicle Objects

```python
class LegacyVehicle(object)
```

Implementation of Vehicle with legacy keys.

<a id="fastsim.vehicle.LegacyVehicle.__init__"></a>

#### \_\_init\_\_

```python
def __init__(vehicle: Vehicle)
```

Given cycle, returns legacy cycle.

<a id="fastsim.vehicle.to_native_type"></a>

#### to\_native\_type

```python
def to_native_type(value)
```

Attempts to map from numpy and other types to python native for better yaml (de-)serialization

<a id="fastsim.vehicle.copy_vehicle"></a>

#### copy\_vehicle

```python
def copy_vehicle(
    veh: Vehicle,
    return_type: str = None,
    deep: bool = True
) -> Dict[str, np.ndarray] | Vehicle | LegacyVehicle | RustVehicle
```

Returns copy of Vehicle.

**Arguments**:

- `veh` - instantiated Vehicle or RustVehicle
  return_type:
- `'dict'` - dict
- `'vehicle'` - Vehicle
- `'legacy'` - LegacyVehicle
- `'rust'` - RustVehicle

<a id="fastsim.vehicle.veh_equal"></a>

#### veh\_equal

```python
def veh_equal(veh1: Vehicle, veh2: Vehicle, full_out: bool = False) -> bool
```

Given veh1 and veh2, which can be Vehicle and/or RustVehicle
instances, return True if equal.

**Arguments**:

  ----------

<a id="fastsim.utils"></a>

# fastsim.utils

<a id="fastsim.utils.vehicle_import_preproc"></a>

# fastsim.utils.vehicle\_import\_preproc

Module for pre-processing data from fueleconomy.gov and EPA vehicle testing that is used for "vehicle import" functionality.
Vehicle import allows FASTSim to import vehicles by specifying make, model, and year. 
See fastsim.demos.vehicle_import_demo for usage.

In order to run this pre-processing script, the data from the sources below should be placed in the "input_dir" (see the run function).

fueleconomy.gov data:
https://www.fueleconomy.gov/feg/download.shtml
- vehicles.csv
- emissions.csv

EPA Test data:
https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
- the data for emissions by year; e.g., 20tstcar-2021-03-02.xlsx
- note: there are multiple formats in use

<a id="fastsim.utils.vehicle_import_preproc.process_csv"></a>

#### process\_csv

```python
def process_csv(path: Path, fn)
```



<a id="fastsim.utils.vehicle_import_preproc.write_csvs_for_each_year"></a>

#### write\_csvs\_for\_each\_year

```python
def write_csvs_for_each_year(output_data_dir, basename, rows_by_year, header)
```



<a id="fastsim.utils.vehicle_import_preproc.sort_fueleconomygov_data_by_year"></a>

#### sort\_fueleconomygov\_data\_by\_year

```python
def sort_fueleconomygov_data_by_year(input_data_dir: Path,
                                     output_data_dir: Path)
```

Opens up the vehicles.csv and emissions.csv and breaks them up to be by year and saves them again.

<a id="fastsim.utils.vehicle_import_preproc.xlsx_to_csv"></a>

#### xlsx\_to\_csv

```python
def xlsx_to_csv(xlsx_path, csv_path)
```



<a id="fastsim.utils.vehicle_import_preproc.process_epa_test_data"></a>

#### process\_epa\_test\_data

```python
def process_epa_test_data(input_dir, output_dir)
```



<a id="fastsim.utils.vehicle_import_preproc.create_zip_archives_by_year"></a>

#### create\_zip\_archives\_by\_year

```python
def create_zip_archives_by_year(files_dir, zip_dir)
```

Takes files in the files_dir that start with \d\d\d\d-*.csv
and adds them to a \d\d\d\d.zip in the zip_dir

<a id="fastsim.utils.utilities"></a>

# fastsim.utils.utilities

Various optional utilities that may support some applications of FASTSim.

<a id="fastsim.utils.utilities.R_air"></a>

#### R\_air

J/(kg*K)

<a id="fastsim.utils.utilities.get_rho_air"></a>

#### get\_rho\_air

```python
def get_rho_air(temperature_degC, elevation_m=180)
```

Returns air density [kg/m**3] for given elevation and temperature.
Source: https://www.grc.nasa.gov/WWW/K-12/rocket/atmosmet.html

**Arguments**:

  ----------
  temperature_degC : ambient temperature [°C]
  elevation_m : elevation above sea level [m].
  Default 180 m is for Chicago, IL

<a id="fastsim.utils.utilities.l__100km_to_mpg"></a>

#### l\_\_100km\_to\_mpg

```python
def l__100km_to_mpg(l__100km)
```

Given fuel economy in L/100km, returns mpg.

<a id="fastsim.utils.utilities.mpg_to_l__100km"></a>

#### mpg\_to\_l\_\_100km

```python
def mpg_to_l__100km(mpg)
```

Given fuel economy in mpg, returns L/100km.

<a id="fastsim.utils.utilities.rollav"></a>

#### rollav

```python
def rollav(x, y, width=10)
```

Returns x-weighted backward-looking rolling average of y.
Good for resampling data that needs to preserve cumulative information.

**Arguments**:

  ----------
  x : x data
  y : y data (`len(y) == len(x)` must be True)
- `width` - rolling average width

<a id="fastsim.utils.utilities.camel_to_snake"></a>

#### camel\_to\_snake

```python
def camel_to_snake(name)
```

Given camelCase, returns snake_case.

<a id="fastsim.utils.utilities.set_log_level"></a>

#### set\_log\_level

```python
def set_log_level(level: str | int) -> int
```

Sets logging level for both Python and Rust FASTSim.
The default logging level is WARNING (30).
https://docs.python.org/3/library/logging.html#logging-levels

Parameters
----------
level: `str` | `int`
    Logging level to set. `str` level name or `int` logging level

    =========== ================
    Level       Numeric value
    =========== ================
    CRITICAL    50
    ERROR       40
    WARNING     30
    INFO        20
    DEBUG       10
    NOTSET      0

Returns
-------
`int`
    Previous log level

<a id="fastsim.utils.utilities.disable_logging"></a>

#### disable\_logging

```python
def disable_logging() -> int
```

Disable FASTSim logs from being shown by setting log level
to CRITICAL+1 (51).

Returns
-------
`int`
    Previous log level

<a id="fastsim.utils.utilities.enable_logging"></a>

#### enable\_logging

```python
def enable_logging(level: Optional[int | str] = None)
```

Re-enable FASTSim logging, optionally to a specified log level,
otherwise to the default WARNING (30) level.

Parameters
----------
level: `str` | `int`, optional
    Logging level to set. `str` level name or `int` logging level.
    See `utils.set_log_level()` docstring for more details on logging levels.

<a id="fastsim.utils.utilities.suppress_logging"></a>

#### suppress\_logging

```python
@contextmanager
def suppress_logging()
```

Disable, then re-enable FASTSim logging using a context manager.
The log level is returned to its previous value.
Logging is re-enabled even if the nested code throws an error.

**Example**:

``` python
with fastsim.utils.suppress_logging():
    ...  # do stuff with logging suppressed
```

<a id="fastsim.utils.utilities.get_containers_with_path"></a>

#### get\_containers\_with\_path

```python
def get_containers_with_path(struct: Any, path: str | list) -> list
```

Get all attributes containers from nested struct using `path` to attribute.

Parameters
----------
struct: Any
    Outermost struct where first name in `path` is an attribute
path: str | list
    Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`

Returns
-------
List[Any]
    Ordered list of containers, from outermost to innermost

<a id="fastsim.utils.utilities.get_attr_with_path"></a>

#### get\_attr\_with\_path

```python
def get_attr_with_path(struct: Any, path: str | list) -> Any
```

Get attribute from nested struct using `path` to attribute.

Parameters
----------
struct: Any
    Outermost struct where first name in `path` is an attribute
path: str | list
    Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`

Returns
-------
Any
    Requested attribute

<a id="fastsim.utils.utilities.set_attr_with_path"></a>

#### set\_attr\_with\_path

```python
def set_attr_with_path(struct: Any, path: str | list, value: Any) -> Any
```

Set attribute on nested struct using `path` to attribute.

Parameters
----------
struct: Any
    Outermost struct where first name in `path` is an attribute
path: str | list
    Dot-separated path, e.g. `"sd.veh.drag_coef"` or `["sd", "veh", "drag_coef"]`
value: Any

Returns
-------
Any
    `struct` with nested value set

<a id="fastsim.utils.utilities.set_attrs_with_path"></a>

#### set\_attrs\_with\_path

```python
def set_attrs_with_path(struct: Any, paths_and_values: Dict[str, Any]) -> Any
```

Set multiple attributes on nested struct using `path`: `value` pairs.

Parameters
----------
struct: Any
    Outermost struct where first name in `path` is an attribute
paths_and_values: Dict[str | list, Any]
    Mapping of dot-separated path (e.g. `sd.veh.drag_coef` or `["sd", "veh", "drag_coef"]`)
    to values (e.g. `0.32`)

Returns
-------
Any
    `struct` with nested values set

<a id="fastsim.utils.utilities.calculate_tire_radius"></a>

#### calculate\_tire\_radius

```python
def calculate_tire_radius(tire_code: str, units: str = "m")
```

Calculate tire radius from ISO tire code, with variable units

Unit options: "m", "cm", "mm", "ft", "in". Default is "m".

**Examples**:

  >>> fastsim.utils.calculate_tire_radius("P205/60R16")
  0.3262
  >>> fastsim.utils.calculate_tire_radius("225/70Rx19.5G", units="in")
  15.950787401574804

<a id="fastsim.utils.utilities.show_plots"></a>

#### show\_plots

```python
def show_plots() -> bool
```

Returns true if plots should be displayed

<a id="fastsim.utils.utilities.copy_demo_files"></a>

#### copy\_demo\_files

```python
def copy_demo_files(path_for_copies: Path = Path("demos"))
```

Copies demo files from demos folder into specified local directory

__Arguments__

- __- `path_for_copies`__: path to copy files into (relative or absolute in)

__Warning__

Running this function will overwrite existing files with the same name in the specified directory, so 
make sure any files with changes you'd like to keep are renamed.

<a id="fastsim.inspect_utils"></a>

# fastsim.inspect\_utils

Utilities to assist with object introspection.

<a id="fastsim.inspect_utils.isprop"></a>

#### isprop

```python
def isprop(attr) -> bool
```

Checks if instance attribute is a property.

<a id="fastsim.inspect_utils.isfunc"></a>

#### isfunc

```python
def isfunc(attr) -> bool
```

Checks if instance attribute is method.

<a id="fastsim.inspect_utils.get_attrs"></a>

#### get\_attrs

```python
def get_attrs(instance)
```

Given an instantiated object, returns attributes that are not:
-- callable  
-- special (i.e. start with `__`)  
-- properties

<a id="fastsim.parameters"></a>

# fastsim.parameters

Global constants representing unit conversions that shourd never change, 
physical properties that should rarely change, and vehicle model parameters 
that can be modified by advanced users.

<a id="fastsim.parameters.PhysicalProperties"></a>

## PhysicalProperties Objects

```python
class PhysicalProperties(object)
```

Container class for physical constants that could change under certain special
circumstances (e.g. high altitude or extreme weather)

<a id="fastsim.parameters.copy_physical_properties"></a>

#### copy\_physical\_properties

```python
def copy_physical_properties(p: PhysicalProperties,
                             return_type: str = None,
                             deep: bool = True)
```

Returns copy of PhysicalProperties.

**Arguments**:

- `p` - instantianed PhysicalProperties or RustPhysicalProperties
  return_type:
- `default` - infer from type of p
- `'dict'` - dict
- `'python'` - PhysicalProperties
- `'legacy'` - LegacyPhysicalProperties -- NOT IMPLEMENTED YET; is it needed?
- `'rust'` - RustPhysicalProperties
- `deep` - if True, uses deepcopy on everything

<a id="fastsim.parameters.physical_properties_equal"></a>

#### physical\_properties\_equal

```python
def physical_properties_equal(a: PhysicalProperties,
                              b: PhysicalProperties) -> bool
```

Return True if the physical properties are equal by value

<a id="fastsim.parameters.fc_perc_out_array"></a>

#### fc\_perc\_out\_array

hardcoded ***

<a id="fastsim.parameters.chg_eff"></a>

#### chg\_eff

charger efficiency for PEVs, this should probably not be hard coded long term

<a id="fastsim.demos.vehicle_import_demo"></a>

# fastsim.demos.vehicle\_import\_demo

Vehicle Import Demonstration
This module demonstrates the vehicle import API

<a id="fastsim.demos.vehicle_import_demo.other_inputs"></a>

#### other\_inputs

None -> calculate from EPA data

<a id="fastsim.demos.fusion_thermal_cal_post"></a>

# fastsim.demos.fusion\_thermal\_cal\_post

<a id="fastsim.demos.fusion_thermal_cal_post.save_path"></a>

#### save\_path

seems to be best

<a id="fastsim.demos.cav_sweep"></a>

# fastsim.demos.cav\_sweep

<a id="fastsim.demos.cav_sweep.ABSOLUTE_EXTENDED_TIME_S"></a>

#### ABSOLUTE\_EXTENDED\_TIME\_S

180.0

<a id="fastsim.demos.cav_sweep.make_debug_plot"></a>

#### make\_debug\_plot

```python
def make_debug_plot(sd: fastsim.simdrive.SimDrive,
                    save_file: Optional[str] = None,
                    do_show: bool = False)
```



<a id="fastsim.demos.cav_sweep.load_cycle"></a>

#### load\_cycle

```python
def load_cycle(cyc_name: str, use_rust: bool = False) -> fastsim.cycle.Cycle
```

Load the given cycle and return

<a id="fastsim.demos.cav_sweep.main"></a>

#### main

```python
def main(cycle_name=None,
         powertrain=None,
         do_show=None,
         use_rust=False,
         verbose=True,
         save_dir=None,
         maneuver=None)
```



<a id="fastsim.demos.time_dilation_demo"></a>

# fastsim.demos.time\_dilation\_demo

<a id="fastsim.demos.accel_demo"></a>

# fastsim.demos.accel\_demo

<a id="fastsim.demos.accel_demo.create_accel_cyc"></a>

#### create\_accel\_cyc

```python
def create_accel_cyc(length_in_seconds=300, spd_mph=89.48, grade=0.0, hz=10)
```

Create a synthetic Drive Cycle for acceleration targeting.
Defaults to a 15 second acceleration cycle. Should be adjusted based on target acceleration time
and initial vehicle acceleration time, so that time isn't wasted on cycles that are needlessly long.

spd_mph @ 89.48 FASTSim XL version mph default speed for acceleration cycles
grade @ 0 and hz @ 10 also matches XL version settings

<a id="fastsim.demos.accel_demo.main"></a>

#### main

```python
def main()
```

**Arguments**:

  ----------

<a id="fastsim.demos.cav_demo"></a>

# fastsim.demos.cav\_demo

<a id="fastsim.demos"></a>

# fastsim.demos

<a id="fastsim.demos.demo_eu_vehicle_wltp"></a>

# fastsim.demos.demo\_eu\_vehicle\_wltp

<a id="fastsim.demos.fusion_thermal_demo"></a>

# fastsim.demos.fusion\_thermal\_demo

<a id="fastsim.demos.timing_demo"></a>

# fastsim.demos.timing\_demo

<a id="fastsim.demos.mp_parallel_demo"></a>

# fastsim.demos.mp\_parallel\_demo

Script for demonstrating parallelization of FASTSim.  
Optional positional arguments:
    - processes: int, number of processes

<a id="fastsim.demos.2017_Ford_F150_thermal_val"></a>

# fastsim.demos.2017\_Ford\_F150\_thermal\_val

<a id="fastsim.demos.2017_Ford_F150_thermal_val.lhv_fuel_btu_per_lbm"></a>

#### lhv\_fuel\_btu\_per\_lbm

from "2012FordFusionV6Overview V5.pdf"

<a id="fastsim.demos.test_demos"></a>

# fastsim.demos.test\_demos

<a id="fastsim.demos.fusion_thermal_cal"></a>

# fastsim.demos.fusion\_thermal\_cal

<a id="fastsim.demos.fusion_thermal_cal.lhv_fuel_btu_per_lbm"></a>

#### lhv\_fuel\_btu\_per\_lbm

from "2012FordFusionV6Overview V5.pdf"

<a id="fastsim.demos.stop_start_demo"></a>

# fastsim.demos.stop\_start\_demo

<a id="fastsim.demos.demo_abc_drag_coef_conv"></a>

# fastsim.demos.demo\_abc\_drag\_coef\_conv

<a id="fastsim.demos.demo"></a>

# fastsim.demos.demo

<a id="fastsim.demos.demo.v2"></a>

#### v2

should not have derived params

<a id="fastsim.demos.demo.data_path"></a>

#### data\_path

path to drive cycles

<a id="fastsim.demos.demo.veh"></a>

#### veh

load vehicle model

<a id="fastsim.demos.demo.veh"></a>

#### veh

load vehicle model

<a id="fastsim.demos.demo.veh"></a>

#### veh

load vehicle using name

<a id="fastsim.demos.wltc_calibration"></a>

# fastsim.demos.wltc\_calibration

<a id="fastsim.demos.wltc_calibration.WILLANS_FACTOR"></a>

#### WILLANS\_FACTOR

gCO2/MJ

<a id="fastsim.demos.wltc_calibration.E10_HEAT_VALUE"></a>

#### E10\_HEAT\_VALUE

kWh/L

<a id="fastsim.calibration"></a>

# fastsim.calibration

<a id="fastsim.calibration.get_error_val"></a>

#### get\_error\_val

```python
def get_error_val(model, test, time_steps)
```

Returns time-averaged error for model and test signal.

**Arguments**:

  ----------
- `model` - array of values for signal from model
- `test` - array of values for signal from test data
- `time_steps` - array (or scalar for constant) of values for model time steps [s]
- `test` - array of values for signal from test
  
  Output:
  -------
- `err` - integral of absolute value of difference between model and
  test per time

<a id="fastsim.calibration.ModelObjectives"></a>

## ModelObjectives Objects

```python
@dataclass
class ModelObjectives(object)
```

Class for calculating eco-driving objectives

<a id="fastsim.calibration.ModelObjectives.get_errors"></a>

#### get\_errors

```python
def get_errors(
    sim_drives: Dict[str, fsr.RustSimDrive | fsr.SimDriveHot],
    return_mods: bool = False,
    plot: bool = False,
    plot_save_dir: Optional[str] = None,
    plot_perc_err: bool = False,
    show: bool = False,
    fontsize: float = 12,
    plotly: bool = False
) -> Union[
        Dict[str, Dict[str, float]],
        # or if return_mods is True
        Dict[str, fsim.simdrive.SimDrive],
]
```

Calculate model errors w.r.t. test data for each element in dfs/models for each objective.

**Arguments**:

  ----------
  - sim_drives: dictionary with user-defined keys and SimDrive or SimDriveHot instances
  - return_mods: if true, also returns dict of solved models
  - plot: if true, plots objectives using matplotlib.pyplot
  - plot_save_dir: directory in which to save plots.  If `None` (default), plots are not saved.
  - plot_perc_err: whether to include % error axes in plots
  - show: whether to show matplotlib.pyplot plots
  - fontsize: plot font size
  - plotly: whether to generate plotly plots, which can be opened manually in a browser window

<a id="fastsim.calibration.ModelObjectives.update_params"></a>

#### update\_params

```python
def update_params(xs: List[Any])
```

Updates model parameters based on `x`, which must match length of self.params

<a id="fastsim.calibration.get_parser"></a>

#### get\_parser

```python
def get_parser() -> argparse.ArgumentParser
```

Generate parser for optimization hyper params and misc. other params

