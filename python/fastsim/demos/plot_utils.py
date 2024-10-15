from cycler import cycler
import numpy as np

figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
BASE_COLORS = [
    '#1f77b4', 
    '#ff7f0e', 
    '#2ca02c', 
    '#d62728', 
    '#9467bd', 
    '#8c564b', 
    '#e377c2',
    '#7f7f7f', 
    '#bcbd22', 
    '#17becf',
]
BASE_LINE_STYLES = ["--", "-.", ":",]

COLOR = "color"
LINESTYLE = "linestyle"
DEF_PAIR_ATTR = COLOR

def get_paired_cycler(pair_attr: str = DEF_PAIR_ATTR):
    """
    # Arguments:
    - `pair_attr`: whether the paired lines should match in `"color"` or `"linestyle"`        
    """
    assert (pair_attr == COLOR) or (pair_attr == LINESTYLE)
    # construct array of repeated 
    if pair_attr == LINESTYLE:
        series_list = BASE_COLORS
    else:
        series_list = BASE_LINE_STYLES
    series = [[c, c] for c in series_list]
    series = [x for sublist in series for x in sublist]

    if pair_attr == LINESTYLE:
        pairs = (BASE_LINE_STYLES[:2] * int(np.ceil(len(series) / 2)))[:len(series)]
    else:
        pairs = (BASE_COLORS[:2] * int(np.ceil(len(series) / 2)))[:len(series)]

    paired_cycler = (
        cycler(color=pairs if pair_attr == COLOR else series) +
        cycler(linestyle=pairs if pair_attr == LINESTYLE else series)
    )
    return paired_cycler

def get_uni_cycler(pair_attr: str = DEF_PAIR_ATTR):
    """
    # Arguments:
    - `pair_attr`: ensures consistent behavior with `get_paired_cycler`
    """
    assert (pair_attr == COLOR) or (pair_attr == LINESTYLE)
    if pair_attr == COLOR:
        colors = BASE_COLORS
        linestyles = ["--",] * len(colors)
    else:
        linestyles = BASE_LINE_STYLES
        colors = [BASE_COLORS[0],] * len(linestyles)
    uni_cycler = (
        cycler(color=colors) +
        cycler(linestyle=linestyles)
    )
    return uni_cycler


