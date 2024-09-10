from cycler import cycler
import numpy as np

figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
base_colors = [
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
base_line_styles = ["--", ":", "-.",]

COLOR = "color"
LINESTYLE = "linestyle"

def get_paired_cycler(pair_attr: str = LINESTYLE):
    assert (pair_attr == COLOR) or (pair_attr == LINESTYLE)
    # construct array of repeated 
    if pair_attr == LINESTYLE:
        series_list = base_colors
    else:
        series_list = base_line_styles
    series = [[c, c] for c in series_list]
    series = [x for sublist in series for x in sublist]

    if pair_attr == LINESTYLE:
        pairs = (base_line_styles[:2] * int(np.ceil(len(series) / 2)))[:len(series)]
    else:
        pairs = (base_colors[:2] * int(np.ceil(len(series) / 2)))[:len(series)]

    paired_cycler = (
        cycler(color=pairs if pair_attr == COLOR else series) +
        cycler(linestyle=pairs if pair_attr == LINESTYLE else series)
    )
    return paired_cycler

def get_uni_cycler():
    colors = base_colors
    baselinestyles = ["--",]
    linestyles = (baselinestyles * int(np.ceil(len(colors) / len(baselinestyles))))[:len(colors)]
    uni_cycler = (
        cycler(color=colors) +
        cycler(linestyle=linestyles)
    )
    return uni_cycler


