import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from configuration import Config, Debug
from collections import OrderedDict

path_csv_files = os.path.join(Config.path_zenodo, "sensitivity_analysis")

# Values of epsilon to plot
epsilon_to_plot = [0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1]
epsilon_to_plot_legend = ['ε = 0.1', 'ε = 0.3', 'ε = 0.4', 'ε = 0.45', 'ε = 0.5', 'ε = 0.55', 'ε = 0.6', 'ε = 0.8', 'Benchmark']

# Dictionary with models to plot
models_to_plot = {"(a) RLR": {'row_to_plot': 5}, "(b) RSIC": {'row_to_plot': 3}, "(c) RGMM": {'row_to_plot': 1}}

# Read the dates
dates = pd.read_csv(os.path.join(path_csv_files, "dates_OD_0919.csv"), names=['index', 'date']).drop(columns='index')
dates_array = dates.to_numpy().reshape(dates.shape[0])

# Change date format x-ticks
for idx_date, date_i in enumerate(dates_array):
    [day, month, year] = date_i.split("/")
    dates_array[idx_date] = f"{year}-{month}-{day}"

# Create figure (gridspec)
f, axarr = plt.subplots(3, 1, figsize=(15, 12))
plt.rcParams["figure.figsize"] = [2, 3.50]
plt.rcParams["figure.autolayout"] = True
ax = plt.GridSpec(3, 1)
ax.update(wspace=0.5, hspace=0.7)

# Define style
linestyles = ['-', '--', '-.', ':', '-', '-', '-', '--', '-.']
markerstyles = ['', '', '', '', 'o', 'v', 's', 'o', 'v']
colors = ['#43A047', '#00CC00', '#0033FF', '#F44336', '#FF00FF', '#9E9E9E', '#F1C40F', '#00FFFF']
colors = ['red', 'blue', '#00CC00', '#B05CF7', 'magenta', '#6CF75C', 'grey', 'cyan', 'orange']
marker_sizes = [4, 4, 4, 4, 4, 4, 4, 2, 2]

# Build dictionary with results and plot
epsilon = dict()
plot_count = 0

# Loop models
for idx_model, model in enumerate(models_to_plot):
    epsilon[model] = dict()
    axarr[idx_model] = plt.subplot(ax[idx_model])
    plot_count = 0

    # Loop csv files
    for file_name in os.scandir(path_csv_files):

        # If the file contains desired results, proceed to store and plot
        if file_name.name.startswith('eps'):
            data_epsilon = pd.read_csv(file_name.path)
            data_epsilon_crop = data_epsilon.iloc[[models_to_plot[model]['row_to_plot']]].to_numpy().reshape(44)
            epsilon[model][data_epsilon_crop[0]] = data_epsilon_crop[1:]
            epsilon_value = data_epsilon_crop[0]
            # If epsilon is 1 we actually want another row
            if epsilon_value == 1:
                data_epsilon_crop = data_epsilon.iloc[[models_to_plot[model]['row_to_plot']-1]].to_numpy().reshape(44)
                epsilon[model][data_epsilon_crop[0]] = data_epsilon_crop[1:]

    # Order dictionary
    epsilon[model] = OrderedDict(sorted(epsilon[model].items()))

    # Loop results
    for epsilon_i in epsilon_to_plot:
            axarr[idx_model].plot(dates_array, epsilon[model][epsilon_i], linewidth=1,
                                  linestyle=linestyles[plot_count], marker=markerstyles[plot_count],
                                  markersize=marker_sizes[plot_count], color=colors[plot_count])
            plot_count = plot_count + 1

    # Set plot grids
    axarr[idx_model].grid(alpha=0.2)

    # Rotate ticks (which include date labels) and center
    axarr[idx_model].set_xticklabels(axarr[idx_model].get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # Adjust ylim
    start, end = axarr[idx_model].get_ylim()
    start = np.round(start / 1e3) * 1e3
    end = np.round((end + 1200) / 1e3) * 1e3
    axarr[idx_model].yaxis.set_ticks(np.arange(start, end, 2e3))

    # Add x and y labels
    axarr[idx_model].set_xlabel(model, fontsize=14, fontname='Times New Roman')
    axarr[idx_model].set_ylabel('Water pixels', fontsize=12, fontname='Times New Roman')

    # Remove axis
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[idx_model].spines[axis].set_linewidth(0)
    if idx_model > 0:
        axarr[idx_model].tick_params(axis='x', colors='white')

# Legend
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=13)
leg = axarr[0].legend(epsilon_to_plot_legend, ncol=len(epsilon_to_plot), loc='upper center', bbox_to_anchor=(0.5, 1.3),
                      prop=font)
leg.get_frame().set_alpha(0.4)

# Save figure as pdf
if Debug.save_figures:
    plt.savefig(os.path.join(Config.path_figures, 'sensitivity_analysis.pdf'), format="pdf", bbox_inches="tight")