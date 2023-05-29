import pandas as pd
from typing import Tuple, Optional
import numpy as np


def resample(
    df: pd.DataFrame,
    dt_new: Optional[float] = 1.0,
    time_col: Optional[str] = "Time[s]",
    rate_vars: Optional[Tuple[str]] = [],
    hold_vars: Optional[Tuple[str]] = [],

) -> pd.DataFrame:
    """
    Resamples dataframe `df`.
    Arguments:
    - df: dataframe to resample
    - dt_new: new time step size, default 1.0 s
    - time_col: column for time in s
    - rate_vars: list of variables that represent rates that need to be time averaged
    - hold_vars: vars that need zero-order hold from previous nearest time step 
        (e.g. quantized variables like current gear)
    """

    new_dict = dict()

    new_time = np.arange(
        0, np.floor(df[time_col].to_numpy()[-1] / dt_new) * dt_new + dt_new,
        dt_new
    )

    for col in df.columns:
        if col in rate_vars:
            # calculate average value over time step
            cumu_vals = (df[time_col].diff().fillna(0) * df[col]).cumsum()
            new_dict[col] = np.diff(
                np.interp(
                    x=new_time,
                    xp=df[time_col].to_numpy(),
                    fp=cumu_vals),
                prepend=0
            ) / dt_new

        elif col in hold_vars:
            assert col not in rate_vars
            pass  # this may need to be fleshed out

        else:
            # just interpolate -- i.e. state variables like temperatures
            new_dict[col] = np.interp(
                x=new_time,
                xp=df[time_col].to_numpy(),
                fp=df[col].to_numpy()
            )

    return pd.DataFrame(new_dict)
