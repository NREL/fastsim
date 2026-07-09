# Vehicles in FASTSim

A vehicle model is a structured representation of the physical parameters of a real-world vehicle. FASTSim’s modeling framework has been exercised over a wide variety of on-road vehicles, from passenger cars, to two-wheelers, to medium- and heavy-duty vocational vehicles.

FASTSim's vehicle model is a hierarchy of components:

:::::{dropdown} Vehicle
:open:

- *Total vehicle mass `mass_kilograms`
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
- *Component-level masses:
  - Chassis mass `mass_kilograms`
  - Alternatively:
    - Glider mass `glider_mass_kilograms`
    - Cargo mass `cargo_mass_kilograms`
:::


::::{dropdown} **Cabin model (`cabin`)
:open:

:::{dropdown} None
No cabin thermal model
- Set with:
  - `~` in the vehicle YAML
  - `None` in the vehicle Python dictionary
:::

:::{dropdown} LumpedCabin
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
:::
::::


::::{dropdown} **HVAC model (`hvac`)
:open:

:::{dropdown} None
No HVAC thermal model
- Set with:
  - `~` in the vehicle YAML
  - `None` in the vehicle Python dictionary
:::

:::{dropdown} LumpedCabin
Lumped thermal capacitance cabin model
:::

:::{dropdown} LumpedCabinAndRES
Lumped thermal capacitance cabin model, connected to Reversible Energy Storage (traction battery) thermal model
:::

:::{dropdown} ReversibleEnergyStorageOnly
Reversible Energy Storage (traction battery) thermal management with no cabin thermal model
:::
::::


:::::

---

**Mass can be set for the entire vehicle at the outermost level of the hierarchy `mass_kilograms`, or set on each component and calculated automatically on initialization.*  
***Thermal modeling components, optional.*  
