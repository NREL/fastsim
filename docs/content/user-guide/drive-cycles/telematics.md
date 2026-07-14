# Using Telematics Data

Instead of using predefined regulatory drive cycles, we often wish to simulate vehicle models over real-world vehicle telematics data in order to validate vehicles models or exercise them in representative scenarios.

Firstly, you will need a mapping of your telematics signals to FASTSim's drive cycle inputs.

- Time:
  - Often the simplest data to read from telematics. However, consider using [`pandas.resample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html) to resample data to 1 Hz, as this is the data frequency that FASTSim was designed for.
  - Input: `time_seconds`
- Speed:
  - Telematics data often has 'wheel-based' vehicle speed, as well as GNSS (GPS latitude/longitude) geospatial data. Speed can be derived from time series geospatial data using open-source geospatial Python libraries.
  - Make sure to convert the speed into meters per second, below are conversion factors:
    - 1 mph = 0.44704 m/s
    - 1 km/h = 0.277778 m/s
  - Input: `speed_meters_per_second`
- Grade
  - Most telematics data does not include grade. To process grade signals from latitude/longitude, consider using the free and open-source tool [gradeit](https://github.com/NatLabRockies/gradeit) developed by NLR.
  -  FASTSim accepts grade as a unitless ratio of rise/run, rather than a percentage. If you are working with percent grade data, simply divide by 100. If you are working with degrees/radians, compute the tangent to get grade ($\textrm{grade}=y/x=tan(\theta)$).
  - Input: `grade`

The biggest challenge with simulating over telematics data is often data quality. For each input, be sure to thoroughly check for missing and erroneous values before relying on simulation results. The choice of which column to use is yours; try to pick the most robust signal or apply pre-processing to get high quality input data.
