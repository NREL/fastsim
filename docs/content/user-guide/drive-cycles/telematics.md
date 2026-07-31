# Using Telematics Data

FASTSim can simulate vehicle models using real-world telematics data. This is useful for validating vehicle models and estimating energy consumption over realistic driving scenarios.

To process telematics data into FASTSim drive cycles, start by mapping your signals to cycle inputs:

- Time:
  - Often the simplest data to read from telematics. Note that FASTSim is designed for 1 Hz data, so consider resampling with [`pandas.resample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) if necessary.
  - Input: `time_seconds`
- Speed:
  - Telematics data often has 'wheel-based' vehicle speed, as well as GNSS (GPS latitude/longitude) geospatial data. Speed can be derived from time series geospatial data using open-source geospatial Python libraries.
  - Make sure to convert speed to meters per second. Common conversion factors are:
    - 1 mph = 0.44704 m/s
    - 1 km/h = 0.277778 m/s
  - Input: `speed_meters_per_second`
- Grade:
  - Most telematics data does not include grade. To process grade signals from latitude/longitude, consider using the free and open-source tool [gradeit](https://github.com/NatLabRockies/gradeit) developed by NLR.
  - FASTSim accepts grade as a unitless ratio of rise/run, rather than a percentage. If you are working with percent grade data, simply divide by 100.
  - If you are working with road angle in degrees or radians, compute the tangent to get road grade ($\textrm{grade}=y/x=tan(\theta)$).
  - Input: `grade`
- Ambient Temperature:
  - When running thermal simulations, you can often feed in ambient temperature (or test cell temperature) directly from your data into FASTSim. Be sure to convert to Kelvin (add 273.15 to Celsius temperatures).
  - Input: `temp_amb_air_kelvin`
- Initial Elevation:
  - Distance and road grade are used to calculate the elevation over time within FASTSim, affecting air density in aerodynamic calculations. A default initial value of 121.92 m (400 ft) is used if nothing is provided.
  - Input: `init_elev_meters`

The biggest challenge with simulating over telematics data is often data quality. For each input, be sure to thoroughly check for missing and erroneous values before relying on simulation results. The choice of which column to use is yours; try to pick the most robust signal or apply pre-processing to get high quality input data.

Check that the data:
- is free of blanks/NaNs, skipped timestamps, etc.
- is strictly monotonically increasing in time
- contains only non-negative speed values
- is free of unrealistically large values or spikes in road grade and temperature
- is separated into individual cycles where necessary (split on long key-off stops, check/account for multiple days in the same log, etc.)
