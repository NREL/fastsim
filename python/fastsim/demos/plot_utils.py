from cycler import cycler
import numpy as np

figsize_3_stacked = (10, 9)

# set up cycling of colors and linestyles
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf']
baselinestyles = ["--", "-.",]

def get_paired_cycler():
    colors = [[c, c] for c in base_colors]
    colors = [x for sublist in colors for x in sublist]
    linestyles = (baselinestyles * int(np.ceil(len(colors) / len(baselinestyles))))[:len(colors)]
    paired_cycler = (
        cycler(color=colors) +
        cycler(linestyle=linestyles)
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


