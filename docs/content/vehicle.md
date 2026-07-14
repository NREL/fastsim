# Vehicles in FASTSim

A vehicle model is a structured representation of the physical parameters of a real-world vehicle. FASTSim’s modeling framework has been exercised over a wide variety of on-road vehicles, from passenger cars, to two-wheelers, to medium- and heavy-duty vocational vehicles.

FASTSim's vehicle model is a hierarchy of components:

<!-- open main dropdown -->
::::::::{dropdown} Vehicle
:open:

- Total vehicle mass* `mass_kilograms`
- Auxiliary power load `pwr_aux_base_watts`
  - constant value for whole drive cycle
  - e.g. 600 W

:::::::{dropdown} Powertrain type (`pt_type`)
:open:
  Powertrain-specific vehicle parameters.

  One of:
  - `ConventionalVehicle` / `Conv`
  - `HybridElectricVehicle` / `HEV`
  - `PlugInHybridElectricVehicle` / `PHEV`
  - `BatteryElectricVehicle` / `BEV`

  ::::::{dropdown} ConventionalVehicle

  - Alternator efficiency `alt_eff`
    - Applies to auxiliary power load
  - Powertrain control `pt_cntrl`
    - Used for enabling/configuring auto start-stop
  - Decel fuel cutoff (DFCO) control `dfco_cntrl`
  - Conventional powertrain mass* `mass_kilograms`

  :::{dropdown} Fuel storage (tank) `fs`
  - Maximum power output `pwr_out_max_watts`
  - Time to peak power (linear ramp for power limitation) `pwr_ramp_lag_seconds`
  - Energy capacity `energy_capacity_joules`
  - Fuel storage mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - Fuel storage specific energy `specific_energy_joules_per_kilogram`
  :::

  :::::{dropdown} Fuel converter (e.g. engine) `fc`

  - Maximum power output `pwr_out_max_watts`
  - Initial power output `pwr_out_max_init_watts`
  - Time to peak power (linear ramp for power limitation) `pwr_ramp_lag_seconds`
  - Efficiency `eff_interp_from_pwr_out`
  - Idle fuel power `pwr_idle_fuel_watts`
    - Fuel converter power will only drop below this value if it is off
  - FC mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - FC specific power `specific_pwr_watts_per_kilogram`
  - FC thermal model `thrml`

    ::::{dropdown} None

    Disable FC thermal model

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml: None
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'therml': 'None',
      ...
    }
    ```
    :::
    ::::

    ::::{dropdown} FuelConverterThermal

    Lumped thermal capacitance FC model

    - Fuel converter thermal capacitance `heat_capacitance_joules_per_kelvin`
    - Characteristic length `length_for_convection_meters`
    - Heat transfer coefficient to ambient when vehicle is stopped `htc_to_amb_stop_watts_per_square_meter_kelvin`
    - Heat transfer coefficient between adiabatic flame temperature and fuel converter temperature `conductance_from_comb_watts_per_kelvin`
    - Max fraction of combustion heat that goes to fuel converter thermal mass `max_frac_from_comb`
    - Temperature at which thermostat starts to open `tstat_te_sto_kelvin`
    - Temperature delta over which thermostat is partially open `tstat_te_delta_kelvin`
    - Thermostat model `tstat_interp`
    - Ratio of active heat rejection from radiator to passive heat rejection `radiator_effectiveness`
      - Always greater than 1
    - Impact of temperature on efficiency `fc_eff_model`

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml:
      FuelConverterThermal:
        heat_capacitance_joules_per_kelvin: 72132.17831078888
        length_for_convection_meters: 0.5427107485587468
        htc_to_amb_stop_watts_per_square_meter_kelvin: 60.7820169953251
        conductance_from_comb_watts_per_kelvin: 2178.965385428629
        max_frac_from_comb: 0.5
        tstat_te_sto_kelvin: 358.15
        tstat_te_delta_kelvin: 5.0
        tstat_interp:
          data:
            grid:
              - v: 1
                dim:
                  - 2
                data:
                  - 85.0
                  - 90.0
            values:
              v: 1
              dim:
                - 2
              data:
                - 0.0
                - 1.0
          strategy: Linear
          extrapolate: Clamp
        radiator_effectiveness: 168.06311964877182
        fc_eff_model:
          Exponential:
            offset: 320.1873803210462
            lag: 24.998774179607892
            minimum: 0.1927168135089094
        state:
          i: 0
          te_adiabatic_kelvin: 3153.8328742654194
          temperature_kelvin: 295.15
          tstat_open_frac: 0.0
          htc_to_amb_watts_per_square_meter_kelvin: 0.0
          pwr_thrml_to_amb_watts: 0.0
          energy_thrml_to_amb_joules: 0.0
          eff_coeff: 1.0
          pwr_thrml_fc_to_cab_watts: 0.0
          energy_thrml_fc_to_cab_joules: 0.0
          pwr_fuel_as_heat_watts: 0.0
          energy_fuel_as_heat_joules: 0.0
          pwr_thrml_to_tm_watts: 0.0
          energy_thrml_to_tm_joules: 0.0
        history:
          i: []
          te_adiabatic_kelvin: []
          temperature_kelvin: []
          tstat_open_frac: []
          htc_to_amb_watts_per_square_meter_kelvin: []
          pwr_thrml_to_amb_watts: []
          energy_thrml_to_amb_joules: []
          eff_coeff: []
          pwr_thrml_fc_to_cab_watts: []
          energy_thrml_fc_to_cab_joules: []
          pwr_fuel_as_heat_watts: []
          energy_fuel_as_heat_joules: []
          pwr_thrml_to_tm_watts: []
          energy_thrml_to_tm_joules: []
        save_interval: 1
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'thrml': {
        'FuelConverterThermal': {
          'heat_capacitance_joules_per_kelvin': 72132.17831078888,
          'length_for_convection_meters': 0.5427107485587468,
          'htc_to_amb_stop_watts_per_square_meter_kelvin': 60.7820169953251,
          'conductance_from_comb_watts_per_kelvin': 2178.965385428629,
          'max_frac_from_comb': 0.5,
          'tstat_te_sto_kelvin': 358.15,
          'tstat_te_delta_kelvin': 5.0,
          'tstat_interp': {
            'data': {
              'grid': [{
                'v': 1,
                'dim': [2],
                'data': [85.0, 90.0]
              }],
            'values': {
              'v': 1,
              'dim': [2],
              'data': [0.0, 1.0]
            }},
            'strategy': 'Linear',
            'extrapolate': 'Clamp'
          },
          'radiator_effectiveness': 168.06311964877182,
          'fc_eff_model': {
            'Exponential': {
              'offset': 320.1873803210462,
              'lag': 24.998774179607892,
              'minimum': 0.1927168135089094
            }
          },
          'state': {
            'i': 0,
            'te_adiabatic_kelvin': 3153.8328742654194,
            'temperature_kelvin': 295.15,
            'tstat_open_frac': 0.0,
            'htc_to_amb_watts_per_square_meter_kelvin': 0.0,
            'pwr_thrml_to_amb_watts': 0.0,
            'energy_thrml_to_amb_joules': 0.0,
            'eff_coeff': 1.0,
            'pwr_thrml_fc_to_cab_watts': 0.0,
            'energy_thrml_fc_to_cab_joules': 0.0,
            'pwr_fuel_as_heat_watts': 0.0,
            'energy_fuel_as_heat_joules': 0.0,
            'pwr_thrml_to_tm_watts': 0.0,
            'energy_thrml_to_tm_joules': 0.0
          },
          'history': {
            'i': [],
            'te_adiabatic_kelvin': [],
            'temperature_kelvin': [],
            'tstat_open_frac': [],
            'htc_to_amb_watts_per_square_meter_kelvin': [],
            'pwr_thrml_to_amb_watts': [],
            'energy_thrml_to_amb_joules': [],
            'eff_coeff': [],
            'pwr_thrml_fc_to_cab_watts': [],
            'energy_thrml_fc_to_cab_joules': [],
            'pwr_fuel_as_heat_watts': [],
            'energy_fuel_as_heat_joules': [],
            'pwr_thrml_to_tm_watts': [],
            'energy_thrml_to_tm_joules': []
          },
          'save_interval': 1
        }
      },
      ...
    }
    ```
    :::
    ::::
  :::::
  <!-- end of fc -->

  :::{dropdown} Transmission `transmission`
  - Efficiency `eff_interp`
  - Transmission mass* `mass`
  :::
  <!-- end of transmission -->

  ::::::
  <!-- end of ConventionalVehicle -->

  ::::::{dropdown} HybridElectricVehicle & PlugInHybridElectricVehicle

  - Powertrain control `pt_cntrl`
    - Used for enabling/configuring auto start-stop
  - Decel fuel cutoff (DFCO) control `dfco_cntrl`
  - Auxiliary power load control `aux_cntrl`
    - Configures whether auxiliary power should come from `res` or `fc` first
  - HEV/PHEV-specific simulation parameters `sim_params`
  - HEV/PHEV powertrain mass* `mass_kilograms`

  :::::{dropdown} Reversible Energy Storage (traction battery) `res`
  
  - Maximum power output `pwr_out_max_watts`
  - Energy capacity `energy_capacity_joules`
  - Maximum power output `pwr_out_max_watts`
  - Efficiency `eff_interp`
    - One of:
      - `Constant`(float)
      - `CRate`
      - `CRateSOCTemperature`
      - `CRateTemperature`
      - `CRateSOC`
  - Minimum state of charge `min_soc`
  - Maximum state of charge `max_soc`
  - RES mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - RES specific energy `specific_energy_joules_per_kilogram`
  - RES thermal model `thrml`

    ::::{dropdown} None

    Disable RES thermal model

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml: None
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'therml': 'None',
      ...
    }
    ```
    :::
    ::::

    ::::{dropdown} RESLumpedThermal

    Lumped thermal capacitance RES model

    - Thermal capacitance `heat_capacitance_joules_per_kelvin`
    - Heat transfer coefficient from RES to ambient `conductance_to_amb_watts_per_kelvin`
    - Heat transfer coefficient from RES to cabin `conductance_to_cab_watts_per_kelvin`

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml:
      RESLumpedThermal:
        heat_capacitance_joules_per_kelvin: 200000.0
        conductance_to_amb_watts_per_kelvin: 0.1
        conductance_to_cab_watts_per_kelvin: 0.5
        state:
          i: 0
          temperature_kelvin: 295.15
          temp_prev_kelvin: 295.15
          pwr_thrml_from_cabin_watts: 0.0
          energy_thrml_from_cabin_joules: 0.0
          pwr_thrml_from_amb_watts: 0.0
          energy_thrml_from_amb_joules: 0.0
          pwr_thrml_hvac_to_res_watts: 0.0
          energy_thrml_hvac_to_res_joules: 0.0
          pwr_thrml_loss_watts: 0.0
          energy_thrml_loss_joules: 0.0
        history:
          i: []
          temperature_kelvin: []
          temp_prev_kelvin: []
          pwr_thrml_from_cabin_watts: []
          energy_thrml_from_cabin_joules: []
          pwr_thrml_from_amb_watts: []
          energy_thrml_from_amb_joules: []
          pwr_thrml_hvac_to_res_watts: []
          energy_thrml_hvac_to_res_joules: []
          pwr_thrml_loss_watts: []
          energy_thrml_loss_joules: []
        save_interval: 1
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'thrml': {
        'RESLumpedThermal': {
          'heat_capacitance_joules_per_kelvin': 200000.0,
          'conductance_to_amb_watts_per_kelvin': 0.1,
          'conductance_to_cab_watts_per_kelvin': 0.5,
          'state': {
            'i': 0,
            'temperature_kelvin': 295.15,
            'temp_prev_kelvin': 295.15,
            'pwr_thrml_from_cabin_watts': 0.0,
            'energy_thrml_from_cabin_joules': 0.0,
            'pwr_thrml_from_amb_watts': 0.0,
            'energy_thrml_from_amb_joules': 0.0,
            'pwr_thrml_hvac_to_res_watts': 0.0,
            'energy_thrml_hvac_to_res_joules': 0.0,
            'pwr_thrml_loss_watts': 0.0,
            'energy_thrml_loss_joules': 0.0
          },
          'history': {
            'i': [],
            'temperature_kelvin': [],
            'temp_prev_kelvin': [],
            'pwr_thrml_from_cabin_watts': [],
            'energy_thrml_from_cabin_joules': [],
            'pwr_thrml_from_amb_watts': [],
            'energy_thrml_from_amb_joules': [],
            'pwr_thrml_hvac_to_res_watts': [],
            'energy_thrml_hvac_to_res_joules': [],
            'pwr_thrml_loss_watts': [],
            'energy_thrml_loss_joules': []
          },
          'save_interval': 1
        }
      },
      ...
    }
    ```
    :::
    ::::
  :::::
  <!-- end of RES -->

  :::{dropdown} Fuel storage (tank) `fs`
  - Maximum power output `pwr_out_max_watts`
  - Time to peak power (linear ramp for power limitation) `pwr_ramp_lag_seconds`
  - Energy capacity `energy_capacity_joules`
  - Fuel storage mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - Fuel storage specific energy `specific_energy_joules_per_kilogram`
  :::


  :::::{dropdown} Fuel converter (e.g. engine) `fc`

  - Maximum power output `pwr_out_max_watts`
  - Initial power output `pwr_out_max_init_watts`
  - Time to peak power (linear ramp for power limitation) `pwr_ramp_lag_seconds`
  - Efficiency `eff_interp_from_pwr_out`
  - Idle fuel power `pwr_idle_fuel_watts`
    - Fuel converter power will only drop below this value if it is off
  - FC mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - FC specific power `specific_pwr_watts_per_kilogram`
  - FC thermal model `thrml`

    ::::{dropdown} None

    Disable FC thermal model

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml: None
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'therml': 'None',
      ...
    }
    ```
    :::
    ::::

    ::::{dropdown} FuelConverterThermal

    Lumped thermal capacitance FC model

    - Fuel converter thermal capacitance `heat_capacitance_joules_per_kelvin`
    - Characteristic length `length_for_convection_meters`
    - Heat transfer coefficient to ambient when vehicle is stopped `htc_to_amb_stop_watts_per_square_meter_kelvin`
    - Heat transfer coefficient between adiabatic flame temperature and fuel converter temperature `conductance_from_comb_watts_per_kelvin`
    - Max fraction of combustion heat that goes to fuel converter thermal mass `max_frac_from_comb`
    - Temperature at which thermostat starts to open `tstat_te_sto_kelvin`
    - Temperature delta over which thermostat is partially open `tstat_te_delta_kelvin`
    - Thermostat model `tstat_interp`
    - Ratio of active heat rejection from radiator to passive heat rejection `radiator_effectiveness`
      - Always greater than 1
    - Impact of temperature on efficiency `fc_eff_model`

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml:
      FuelConverterThermal:
        heat_capacitance_joules_per_kelvin: 72132.17831078888
        length_for_convection_meters: 0.5427107485587468
        htc_to_amb_stop_watts_per_square_meter_kelvin: 60.7820169953251
        conductance_from_comb_watts_per_kelvin: 2178.965385428629
        max_frac_from_comb: 0.5
        tstat_te_sto_kelvin: 358.15
        tstat_te_delta_kelvin: 5.0
        tstat_interp:
          data:
            grid:
              - v: 1
                dim:
                  - 2
                data:
                  - 85.0
                  - 90.0
            values:
              v: 1
              dim:
                - 2
              data:
                - 0.0
                - 1.0
          strategy: Linear
          extrapolate: Clamp
        radiator_effectiveness: 168.06311964877182
        fc_eff_model:
          Exponential:
            offset: 320.1873803210462
            lag: 24.998774179607892
            minimum: 0.1927168135089094
        state:
          i: 0
          te_adiabatic_kelvin: 3153.8328742654194
          temperature_kelvin: 295.15
          tstat_open_frac: 0.0
          htc_to_amb_watts_per_square_meter_kelvin: 0.0
          pwr_thrml_to_amb_watts: 0.0
          energy_thrml_to_amb_joules: 0.0
          eff_coeff: 1.0
          pwr_thrml_fc_to_cab_watts: 0.0
          energy_thrml_fc_to_cab_joules: 0.0
          pwr_fuel_as_heat_watts: 0.0
          energy_fuel_as_heat_joules: 0.0
          pwr_thrml_to_tm_watts: 0.0
          energy_thrml_to_tm_joules: 0.0
        history:
          i: []
          te_adiabatic_kelvin: []
          temperature_kelvin: []
          tstat_open_frac: []
          htc_to_amb_watts_per_square_meter_kelvin: []
          pwr_thrml_to_amb_watts: []
          energy_thrml_to_amb_joules: []
          eff_coeff: []
          pwr_thrml_fc_to_cab_watts: []
          energy_thrml_fc_to_cab_joules: []
          pwr_fuel_as_heat_watts: []
          energy_fuel_as_heat_joules: []
          pwr_thrml_to_tm_watts: []
          energy_thrml_to_tm_joules: []
        save_interval: 1
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'thrml': {
        'FuelConverterThermal': {
          'heat_capacitance_joules_per_kelvin': 72132.17831078888,
          'length_for_convection_meters': 0.5427107485587468,
          'htc_to_amb_stop_watts_per_square_meter_kelvin': 60.7820169953251,
          'conductance_from_comb_watts_per_kelvin': 2178.965385428629,
          'max_frac_from_comb': 0.5,
          'tstat_te_sto_kelvin': 358.15,
          'tstat_te_delta_kelvin': 5.0,
          'tstat_interp': {
            'data': {
              'grid': [{
                'v': 1,
                'dim': [2],
                'data': [85.0, 90.0]
              }],
            'values': {
              'v': 1,
              'dim': [2],
              'data': [0.0, 1.0]
            }},
            'strategy': 'Linear',
            'extrapolate': 'Clamp'
          },
          'radiator_effectiveness': 168.06311964877182,
          'fc_eff_model': {
            'Exponential': {
              'offset': 320.1873803210462,
              'lag': 24.998774179607892,
              'minimum': 0.1927168135089094
            }
          },
          'state': {
            'i': 0,
            'te_adiabatic_kelvin': 3153.8328742654194,
            'temperature_kelvin': 295.15,
            'tstat_open_frac': 0.0,
            'htc_to_amb_watts_per_square_meter_kelvin': 0.0,
            'pwr_thrml_to_amb_watts': 0.0,
            'energy_thrml_to_amb_joules': 0.0,
            'eff_coeff': 1.0,
            'pwr_thrml_fc_to_cab_watts': 0.0,
            'energy_thrml_fc_to_cab_joules': 0.0,
            'pwr_fuel_as_heat_watts': 0.0,
            'energy_fuel_as_heat_joules': 0.0,
            'pwr_thrml_to_tm_watts': 0.0,
            'energy_thrml_to_tm_joules': 0.0
          },
          'history': {
            'i': [],
            'te_adiabatic_kelvin': [],
            'temperature_kelvin': [],
            'tstat_open_frac': [],
            'htc_to_amb_watts_per_square_meter_kelvin': [],
            'pwr_thrml_to_amb_watts': [],
            'energy_thrml_to_amb_joules': [],
            'eff_coeff': [],
            'pwr_thrml_fc_to_cab_watts': [],
            'energy_thrml_fc_to_cab_joules': [],
            'pwr_fuel_as_heat_watts': [],
            'energy_fuel_as_heat_joules': [],
            'pwr_thrml_to_tm_watts': [],
            'energy_thrml_to_tm_joules': []
          },
          'save_interval': 1
        }
      },
      ...
    }
    ```
    :::
    ::::
  :::::
  <!-- end of fc -->

  :::{dropdown} Electric Machine (motor) `em`
  Electric machine / motor model

  - `eff_interp_achieved`
  - `eff_interp_at_max_input`
  - Maximum output power `pwr_out_max_watts`
  - EM mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - EM specific power `specific_pwr_watts_per_kilogram`
  :::
  <!-- end of EM -->

  :::{dropdown} Transmission `transmission`
  - Efficiency `eff_interp`
  - Transmission mass* `mass`
  :::
  <!-- end of transmission -->

  ::::::
  <!-- end of HybridElectricVehicle -->
  

  ::::::{dropdown} BatteryElectricVehicle

  - BEV powertrain mass* `mass_kilograms`
  
  :::::{dropdown} Reversible Energy Storage (traction battery) `res`
  
  - Maximum power output `pwr_out_max_watts`
  - Energy capacity `energy_capacity_joules`
  - Maximum power output `pwr_out_max_watts`
  - Efficiency `eff_interp`
    - One of:
      - `Constant`(float)
      - `CRate`
      - `CRateSOCTemperature`
      - `CRateTemperature`
      - `CRateSOC`
  - Minimum state of charge `min_soc`
  - Maximum state of charge `max_soc`
  - RES mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - RES specific energy `specific_energy_joules_per_kilogram`
  - RES thermal model `thrml`

    ::::{dropdown} None

    Disable RES thermal model

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml: None
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'therml': 'None',
      ...
    }
    ```
    :::
    ::::

    ::::{dropdown} RESLumpedThermal

    Lumped thermal capacitance RES model

    - Thermal capacitance `heat_capacitance_joules_per_kelvin`
    - Heat transfer coefficient from RES to ambient `conductance_to_amb_watts_per_kelvin`
    - Heat transfer coefficient from RES to cabin `conductance_to_cab_watts_per_kelvin`

    Examples:

    :::{dropdown} YAML
    ```yaml
    thrml:
      RESLumpedThermal:
        heat_capacitance_joules_per_kelvin: 200000.0
        conductance_to_amb_watts_per_kelvin: 0.1
        conductance_to_cab_watts_per_kelvin: 0.5
        state:
          i: 0
          temperature_kelvin: 295.15
          temp_prev_kelvin: 295.15
          pwr_thrml_from_cabin_watts: 0.0
          energy_thrml_from_cabin_joules: 0.0
          pwr_thrml_from_amb_watts: 0.0
          energy_thrml_from_amb_joules: 0.0
          pwr_thrml_hvac_to_res_watts: 0.0
          energy_thrml_hvac_to_res_joules: 0.0
          pwr_thrml_loss_watts: 0.0
          energy_thrml_loss_joules: 0.0
        history:
          i: []
          temperature_kelvin: []
          temp_prev_kelvin: []
          pwr_thrml_from_cabin_watts: []
          energy_thrml_from_cabin_joules: []
          pwr_thrml_from_amb_watts: []
          energy_thrml_from_amb_joules: []
          pwr_thrml_hvac_to_res_watts: []
          energy_thrml_hvac_to_res_joules: []
          pwr_thrml_loss_watts: []
          energy_thrml_loss_joules: []
        save_interval: 1
    ```
    :::

    :::{dropdown} Python dictionary
    ```python
    {
      ...
      'thrml': {
        'RESLumpedThermal': {
          'heat_capacitance_joules_per_kelvin': 200000.0,
          'conductance_to_amb_watts_per_kelvin': 0.1,
          'conductance_to_cab_watts_per_kelvin': 0.5,
          'state': {
            'i': 0,
            'temperature_kelvin': 295.15,
            'temp_prev_kelvin': 295.15,
            'pwr_thrml_from_cabin_watts': 0.0,
            'energy_thrml_from_cabin_joules': 0.0,
            'pwr_thrml_from_amb_watts': 0.0,
            'energy_thrml_from_amb_joules': 0.0,
            'pwr_thrml_hvac_to_res_watts': 0.0,
            'energy_thrml_hvac_to_res_joules': 0.0,
            'pwr_thrml_loss_watts': 0.0,
            'energy_thrml_loss_joules': 0.0
          },
          'history': {
            'i': [],
            'temperature_kelvin': [],
            'temp_prev_kelvin': [],
            'pwr_thrml_from_cabin_watts': [],
            'energy_thrml_from_cabin_joules': [],
            'pwr_thrml_from_amb_watts': [],
            'energy_thrml_from_amb_joules': [],
            'pwr_thrml_hvac_to_res_watts': [],
            'energy_thrml_hvac_to_res_joules': [],
            'pwr_thrml_loss_watts': [],
            'energy_thrml_loss_joules': []
          },
          'save_interval': 1
        }
      },
      ...
    }
    ```
    :::
    ::::
  :::::
  <!-- end of RES -->

  :::{dropdown} Electric Machine (motor) `em`
  Electric machine / motor model

  - `eff_interp_achieved`
  - `eff_interp_at_max_input`
  - Maximum output power `pwr_out_max_watts`
  - EM mass* `mass_kilograms`
    - Optional component-level mass
    - Alternatively:
      - EM specific power `specific_pwr_watts_per_kilogram`
  :::
  <!-- end of EM -->

  :::{dropdown} Transmission `transmission`
  - Efficiency `eff_interp`
  - Transmission mass* `mass`
  :::
  <!-- end of transmission -->

  ::::::
  <!-- end of BatteryElectricVehicle -->
:::::::


:::{dropdown} Chassis (`chassis`)
:open:
- Aerodynamic drag coefficient `drag_coef`
- Vehicle frontal area `frontal_area_square_meters`
- Wheel rolling resistance coefficient `wheel_rr_coef`
- Number of wheels `num_wheels`
- Wheel radius `wheel_radius_meters`
- Vehicle drivetrain configuration `drive_type`
  - "FWD", "RWD", "AWD", or "FourWD"
  - Only affects traction limitation / tire slip calculation, so if not applicable just choose FWD
- Height of vehicle center of gravity `cg_height_meters`
  - Only affects traction limitation / tire slip calculation, safe to leave at ~0.6 m for most vehicles
- Wheel friction coefficient `wheel_fric_coef`
  - Only affects traction limitation / tire slip calculation, safe to leave at ~0.7 for most vehicles
- Vehicle wheelbase `wheel_base_meters`
- Alternatively set with a tire code `tire_code` e.g. "225/60R18"
- Component-level masses*:
  - Chassis mass `mass_kilograms`
  - Alternatively:
    - Glider mass `glider_mass_kilograms`
    - Cargo mass `cargo_mass_kilograms`
:::


:::::{dropdown} Cabin model (`cabin`)
:open:

Models cabin temperature, heat can flow between cabin and:
- surroundings/ambient conditions
- fuel converter (e.g. engine)
- reversible energy storage (traction battery)

::::{dropdown} None
Disable cabin thermal model

Examples:

:::{dropdown} YAML
```yaml
cabin: None
```
:::

:::{dropdown} Python dictionary
```python
{
  ...
  'cabin': 'None',
  ...
}
```
:::
::::

::::{dropdown} LumpedCabin
Lumped thermal capacitance cabin model
- `cab_shell_htc_to_amb_watts_per_square_meter_kelvin`
  - Inverse of cabin shell thermal resistance
- `cab_htc_to_amb_stop_watts_per_square_meter_kelvin`
  - Heat transfer coefficient from cabin outer surface to ambient when vehicle is stopped
- `heat_capacitance_joules_per_kelvin`
  - Cabin thermal capacitance
- `length_meters`
  - Cabin length (modeled as a flat plate)
- `width_meters`
  - Cabin width (modeled as a flat plate)

Examples:

:::{dropdown} YAML
  ```yaml
  cabin:
    LumpedCabin:
      cab_shell_htc_to_amb_watts_per_square_meter_kelvin: 10.111988385072632
      cab_htc_to_amb_stop_watts_per_square_meter_kelvin: 153.22457193685642
      heat_capacitance_joules_per_kelvin: 250000
      length_meters: 3.302 # 130 in. estimate
      width_meters: 2.02184 # vehicle width without mirrors
      state:
        i: 0
        temperature_kelvin: 295.15
        temp_prev_kelvin: 295.15
        pwr_thrml_from_hvac_watts: 0.0
        energy_thrml_from_hvac_joules: 0.0
        pwr_thrml_from_amb_watts: 0.0
        energy_thrml_from_amb_joules: 0.0
        pwr_thrml_to_res_watts: 0.0
        energy_thrml_to_res_joules: 0.0
        reynolds_for_plate: 0.0
      history:
        i: []
        temperature_kelvin: []
        temp_prev_kelvin: []
        pwr_thrml_from_hvac_watts: []
        energy_thrml_from_hvac_joules: []
        pwr_thrml_from_amb_watts: []
        energy_thrml_from_amb_joules: []
        pwr_thrml_to_res_watts: []
        energy_thrml_to_res_joules: []
        reynolds_for_plate: []
      save_interval: 1
  ```
  :::

  :::{dropdown} Python dictionary
  ```python
  {
    ...
    'cabin': {'LumpedCabin': {'cab_shell_htc_to_amb_watts_per_square_meter_kelvin': 10.111988385072632,
      'cab_htc_to_amb_stop_watts_per_square_meter_kelvin': 153.22457193685642,
      'heat_capacitance_joules_per_kelvin': 250000.0,
      'length_meters': 3.302,
      'width_meters': 2.02184,
      'state': {'i': 0,
        'temperature_kelvin': 295.15,
        'temp_prev_kelvin': 295.15,
        'pwr_thrml_from_hvac_watts': 0.0,
        'energy_thrml_from_hvac_joules': 0.0,
        'pwr_thrml_from_amb_watts': 0.0,
        'energy_thrml_from_amb_joules': 0.0,
        'pwr_thrml_to_res_watts': 0.0,
        'energy_thrml_to_res_joules': 0.0,
        'reynolds_for_plate': 0.0},
      'history': {'i': [],
        'temperature_kelvin': [],
        'temp_prev_kelvin': [],
        'pwr_thrml_from_hvac_watts': [],
        'energy_thrml_from_hvac_joules': [],
        'pwr_thrml_from_amb_watts': [],
        'energy_thrml_from_amb_joules': [],
        'pwr_thrml_to_res_watts': [],
        'energy_thrml_to_res_joules': [],
        'reynolds_for_plate': []},
      'save_interval': 1}},
    ...
  }
  ```
  :::

::::
:::::


:::::{dropdown} HVAC model (`hvac`)
:open:

Models HVAC behavior and related heat flow.

::::{dropdown} None

Disable HVAC thermal model

Examples:

:::{dropdown} YAML
```yaml
hvac: None
```
:::

:::{dropdown} Python dictionary
```python
{
  ...
  'hvac': 'None',
  ...
}
```
:::
::::

::::{dropdown} LumpedCabin
HVAC system for lumped thermal capacitance cabin model.

Examples
:::{dropdown} YAML
```yaml
hvac:
  LumpedCabin:
    te_set_kelvin: 295.15
    te_deadband_kelvin: 0.5
    p_watts_per_kelvin: 489.34499608760177
    i: 36.77964270414921
    pwr_i_max_watts: 10000.0
    d: 5.0
    pwr_thrml_max_watts: 15000.0
    frac_of_ideal_cop: 0.0778419513178728
    heat_source: FuelConverter
    pwr_aux_for_hvac_max_watts: 8000.0
    state:
      i: 0
      pwr_p_watts: 0.0
      energy_p_joules: 0.0
      pwr_i_watts: 0.0
      energy_i_joules: 0.0
      pwr_d_watts: 0.0
      energy_d_joules: 0.0
      cop: ~
      pwr_aux_for_hvac_watts: 0.0
      energy_aux_for_hvac_joules: 0.0
      pwr_thrml_hvac_to_cabin_watts: 0.0
      energy_thrml_hvac_to_cabin_joules: 0.0
      pwr_thrml_fc_to_cabin_watts: 0.0
      energy_thrml_fc_to_cabin_joules: 0.0
    history:
      i: []
      pwr_p_watts: []
      energy_p_joules: []
      pwr_i_watts: []
      energy_i_joules: []
      pwr_d_watts: []
      energy_d_joules: []
      cop: []
      pwr_aux_for_hvac_watts: []
      energy_aux_for_hvac_joules: []
      pwr_thrml_hvac_to_cabin_watts: []
      energy_thrml_hvac_to_cabin_joules: []
      pwr_thrml_fc_to_cabin_watts: []
      energy_thrml_fc_to_cabin_joules: []
    save_interval: 1
```
:::
:::{dropdown} Python dictionary
```python
{
  ...
  'hvac': {'LumpedCabin': {'te_set_kelvin': 295.15,
    'te_deadband_kelvin': 0.5,
    'p_watts_per_kelvin': 489.34499608760177,
    'i': 36.77964270414921,
    'pwr_i_max_watts': 10000.0,
    'd': 5.0,
    'pwr_thrml_max_watts': 15000.0,
    'frac_of_ideal_cop': 0.0778419513178728,
    'heat_source': 'FuelConverter',
    'pwr_aux_for_hvac_max_watts': 8000.0,
    'state': {'i': 0,
      'pwr_p_watts': 0.0,
      'energy_p_joules': 0.0,
      'pwr_i_watts': 0.0,
      'energy_i_joules': 0.0,
      'pwr_d_watts': 0.0,
      'energy_d_joules': 0.0,
      'cop': None,
      'pwr_aux_for_hvac_watts': 0.0,
      'energy_aux_for_hvac_joules': 0.0,
      'pwr_thrml_hvac_to_cabin_watts': 0.0,
      'energy_thrml_hvac_to_cabin_joules': 0.0,
      'pwr_thrml_fc_to_cabin_watts': 0.0,
      'energy_thrml_fc_to_cabin_joules': 0.0},
    'history': {'i': [],
      'pwr_p_watts': [],
      'energy_p_joules': [],
      'pwr_i_watts': [],
      'energy_i_joules': [],
      'pwr_d_watts': [],
      'energy_d_joules': [],
      'cop': [],
      'pwr_aux_for_hvac_watts': [],
      'energy_aux_for_hvac_joules': [],
      'pwr_thrml_hvac_to_cabin_watts': [],
      'energy_thrml_hvac_to_cabin_joules': [],
      'pwr_thrml_fc_to_cabin_watts': [],
      'energy_thrml_fc_to_cabin_joules': []},
    'save_interval': 1}},
  ...
}
```
:::

::::

::::{dropdown} LumpedCabinAndRES
HVAC system for lumped thermal capacitance cabin model and Reversible Energy Storage (traction battery) thermal model

Examples
:::{dropdown} YAML
```yaml
hvac:
  LumpedCabinAndRES:
    te_set_cab_kelvin: 295.15
    te_deadband_cab_kelvin: 0.5
    p_cabin_watts_per_kelvin: 881.6321317014631
    i_cabin: 34.0671708520381
    pwr_i_max_cabin_watts: 15000.0
    d_cabin: 10.0
    te_set_res_kelvin: 295.15
    te_deadband_res_kelvin: 15.0
    p_res_watts_per_kelvin: 7.776638335865715
    i_res: 1.014641941728727
    pwr_i_max_res_watts: 15000.0
    d_res: 10.0
    pwr_thrml_max_watts: 15000.0
    frac_of_ideal_cop: 0.1672948274768393
    cabin_heat_source: ResistanceHeater
    res_heat_source: ResistanceHeater
    res_cooling_source: HVAC
    pwr_aux_for_hvac_cab_max_watts: 5000.0
    pwr_aux_for_hvac_res_max_watts: 5000.0
    save_interval: 1
```
:::
:::{dropdown} Python dictionary
```python
{
  ...
  'hvac': {
    'LumpedCabinAndRES': {
      'te_set_cab_kelvin': 295.15,
      'te_deadband_cab_kelvin': 0.5,
      'p_cabin_watts_per_kelvin': 881.6321317014631,
      'i_cabin': 34.0671708520381,
      'pwr_i_max_cabin_watts': 15000.0,
      'd_cabin': 10.0,
      'te_set_res_kelvin': 295.15,
      'te_deadband_res_kelvin': 15.0,
      'p_res_watts_per_kelvin': 7.776638335865715,
      'i_res': 1.014641941728727,
      'pwr_i_max_res_watts': 15000.0,
      'd_res': 10.0,
      'pwr_thrml_max_watts': 15000.0,
      'frac_of_ideal_cop': 0.1672948274768393,
      'cabin_heat_source': 'ResistanceHeater',
      'res_heat_source': 'ResistanceHeater',
      'res_cooling_source': 'HVAC',
      'pwr_aux_for_hvac_cab_max_watts': 5000.0,
      'pwr_aux_for_hvac_res_max_watts': 5000.0,
      'state': {
        'i': 0,
        'pwr_p_cab_watts': 0.0,
        'energy_p_cab_joules': 0.0,
        'pwr_i_cab_watts': 0.0,
        'energy_i_cab_joules': 0.0,
        'pwr_d_cab_watts': 0.0,
        'energy_d_cab_joules': 0.0,
        'pwr_p_res_watts': 0.0,
        'energy_p_res_joules': 0.0,
        'pwr_i_res_watts': 0.0,
        'energy_i_res_joules': 0.0,
        'pwr_d_res_watts': 0.0,
        'energy_d_res_joules': 0.0,
        'cop': None,
        'te_ref_kelvin': None,
        'pwr_aux_for_cab_hvac_req_watts': 0.0,
        'pwr_thrml_to_cab_req_watts': 0.0,
        'pwr_aux_for_res_hvac_req_watts': 0.0,
        'pwr_thrml_to_res_req_watts': 0.0,
        'pwr_aux_for_cab_hvac_watts': 0.0,
        'energy_aux_for_cab_hvac_joules': 0.0,
        'pwr_aux_for_res_hvac_watts': 0.0,
        'energy_aux_for_res_hvac_joules': 0.0,
        'pwr_thrml_hvac_to_cabin_watts': 0.0,
        'energy_thrml_hvac_to_cabin_joules': 0.0,
        'pwr_thrml_fc_to_cabin_watts': 0.0,
        'energy_thrml_fc_to_cabin_joules': 0.0,
        'pwr_thrml_hvac_to_res_watts': 0.0,
        'energy_thrml_hvac_to_res_joules': 0.0,
        'te_ref_component': 'None',
        'cabin_mode': 'Inactive',
        'res_mode': 'Inactive'
      },
      'history': {
        'i': [],
        'pwr_p_cab_watts': [],
        'energy_p_cab_joules': [],
        'pwr_i_cab_watts': [],
        'energy_i_cab_joules': [],
        'pwr_d_cab_watts': [],
        'energy_d_cab_joules': [],
        'pwr_p_res_watts': [],
        'energy_p_res_joules': [],
        'pwr_i_res_watts': [],
        'energy_i_res_joules': [],
        'pwr_d_res_watts': [],
        'energy_d_res_joules': [],
        'cop': [],
        'te_ref_kelvin': [],
        'pwr_aux_for_cab_hvac_req_watts': [],
        'pwr_thrml_to_cab_req_watts': [],
        'pwr_aux_for_res_hvac_req_watts': [],
        'pwr_thrml_to_res_req_watts': [],
        'pwr_aux_for_cab_hvac_watts': [],
        'energy_aux_for_cab_hvac_joules': [],
        'pwr_aux_for_res_hvac_watts': [],
        'energy_aux_for_res_hvac_joules': [],
        'pwr_thrml_hvac_to_cabin_watts': [],
        'energy_thrml_hvac_to_cabin_joules': [],
        'pwr_thrml_fc_to_cabin_watts': [],
        'energy_thrml_fc_to_cabin_joules': [],
        'pwr_thrml_hvac_to_res_watts': [],
        'energy_thrml_hvac_to_res_joules': [],
        'te_ref_component': [],
        'cabin_mode': [],
        'res_mode': []
      },
      'save_interval': 1
    }
  },
  ...
}
```
:::
::::

<!-- end of main dropdown -->
::::::::


---

**Mass can be set for the entire vehicle at the outermost level of the hierarchy `mass_kilograms`, or set on each component and calculated automatically on initialization.*  
