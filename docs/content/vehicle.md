# Vehicles in FASTSim

A vehicle model is a structured representation of the physical parameters of a real-world vehicle. FASTSim’s modeling framework has been exercised over a wide variety of on-road vehicles, from passenger cars, to two-wheelers, to medium- and heavy-duty vocational vehicles.

FASTSim's vehicle model is a hierarchy of components:

<!-- open main dropdown -->
::::::{dropdown} Vehicle
:open:

- Total vehicle mass* `mass_kilograms`
- Auxiliary power load `pwr_aux_base_watts`
  - constant value for whole drive cycle
  - e.g. 600 W

::::{dropdown} Powertrain type (`pt_type`)
:open:
  TODO

  :::{dropdown} Conventional
  TODO
  :::

  :::{dropdown} Battery Electric
  TODO
  :::

  :::{dropdown} Hybrid Electric
  TODO
  :::

  :::{dropdown} Plug-in Hybrid Electric
  TODO
  :::
::::


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

:::{note}
Optional thermal component, select `"None"` to disable
:::


::::{dropdown} None
Disable cabin thermal model.

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

:::{note}
Optional thermal component, select `"None"` to disable
:::

Models HVAC behavior and related heat flow.

::::{dropdown} None

Disable HVAC thermal model.

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
Lumped thermal capacitance cabin model.

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
Lumped thermal capacitance cabin model, connected to Reversible Energy Storage (traction battery) thermal model

Examples
:::{dropdown} YAML
```yaml
TODO
```
:::
:::{dropdown} Python dictionary
```python
TODO
```
:::
::::

::::{dropdown} ReversibleEnergyStorageOnly
Reversible Energy Storage (traction battery) thermal management with no cabin thermal model

Examples
:::{dropdown} YAML
```yaml
TODO
```
:::
:::{dropdown} Python dictionary
```python
TODO
```
:::
::::

<!-- close main dropdown -->
::::::


---

**Mass can be set for the entire vehicle at the outermost level of the hierarchy `mass_kilograms`, or set on each component and calculated automatically on initialization.*  
