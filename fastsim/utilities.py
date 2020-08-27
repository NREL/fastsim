"""Various optional utilities that may support some applications of FASTSim."""

R_air = 287  # J/(kg*K)

def get_rho_air(elevation_m, temperature_degC, full_output=False):
    """Returns air density [kg/m**3] for given elevation and temperature."""
    #     T = 15.04 - .00649 * h
    #     p = 101.29 * [(T + 273.1)/288.08]^5.256
    T_standard = 15.04 - 0.00649 * elevation_m  # nasa [degC]
    p = 101.29e3 * ((T_standard + 273.1) / 288.08) ** 5.256  # nasa [Pa]
    rho = p / (R_air * (temperature_degC + 273.15))  # [kg/m**3]

    if not(full_output):
        return rho
    else:
        return rho, p, T_standard
