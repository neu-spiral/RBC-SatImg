import os
import glob
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from configuration import Config, Debug
from collections import OrderedDict

path_files = os.path.join(Config.path_zenodo, "evaluation_results", "sensitivity_analysis")
results = dict()
models = ["GMM", "LR", "SIC", "RGMM", "RLR", "RSIC"]
num_images_eval = 42

# Read the dates
#path_csv_files = os.path.join(path_files, "sensitivity_analysis")
dates = pd.read_csv(os.path.join(path_files, "dates_OD_0919.csv"), names=['index', 'date']).drop(columns='index')
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

for folder_name in os.scandir(path_files):
    # If the file contains desired plot_results, proceed to store and plot
    if folder_name.name.startswith('sensitivity_analysis'):
        print(folder_name.name)
        epsilon_value = float(folder_name.name.strip('sensitivity_analysis_epsilon_')[0:4])
        results[epsilon_value] = dict()
        for model_i in models:
            results[epsilon_value][model_i] = np.empty(shape=[1,num_images_eval])[0]
        for image_idx in range(0,num_images_eval):
            file_name  = os.path.join(folder_name, f"sensitivity_analysis_epsilon_{epsilon_value}_image_index_{image_idx}.pkl")
            print(file_name)
            print(image_idx)
            [date, output] = pickle.load(open(file_name, 'rb'))
            for model_i in models:
                results[epsilon_value][model_i][image_idx] = output[model_i]


# Order dictionary
results_sorted = OrderedDict(sorted(results.items()))

models_to_plot = ["RGMM", "RLR", "RSIC"]
epsilon_to_plot = epsilon_to_plot = [0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1]
epsilon_to_plot_legend = ['ε = 0.1', 'ε = 0.3', 'ε = 0.4', 'ε = 0.45', 'ε = 0.5', 'ε = 0.55', 'ε = 0.6', 'ε = 0.8', 'Benchmark']
epsilon_benchmark = 0.5

for idx_model, model in enumerate(models_to_plot):
    plot_count = 0
    axarr[idx_model] = plt.subplot(ax[idx_model])
    # Loop plot_results
    for epsilon_i in epsilon_to_plot:
        if epsilon_i == 1:
            axarr[idx_model].plot(dates_array[1:], results_sorted[epsilon_benchmark][model[1:]], linewidth=1,
                                  linestyle=linestyles[plot_count], marker=markerstyles[plot_count],
                                  markersize=marker_sizes[plot_count], color=colors[plot_count])
        else:
            axarr[idx_model].plot(dates_array[1:], results_sorted[epsilon_i][model], linewidth=1,
                                  linestyle=linestyles[plot_count], marker=markerstyles[plot_count],
                                  markersize=marker_sizes[plot_count], color=colors[plot_count])
        plot_count = plot_count + 1

    # Set plot grids
    axarr[idx_model].grid(alpha=0.2)

    # Rotate ticks (which include date labels) and center
    axarr[idx_model].set_xticklabels(axarr[idx_model].get_xticklabels(), rotation=45, ha="right", fontsize=13, family='Times New Roman')
    labels = axarr[idx_model].get_yticklabels()

    # Adjust ylim
    start, end = axarr[idx_model].get_ylim()
    start = np.round(start / 1e3) * 1e3
    end = np.round((end + 1200) / 1e3) * 1e3
    axarr[idx_model].yaxis.set_ticks(np.arange(start, end, 2e3))

    # Add x and y labels
    axarr[idx_model].set_title(model, fontsize=21, fontname='Times New Roman')
    axarr[idx_model].set_ylabel('Water pixels', fontsize=20, fontname='Times New Roman')
    axarr[idx_model].yaxis.set_label_coords(-0.07, 0.45)

    # Remove axis
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[idx_model].spines[axis].set_linewidth(0)
    if idx_model < 2:
        axarr[idx_model].tick_params(axis='x', colors='white')

axarr[0].set_yticklabels([2000, 4000, 6000, 8000, 10000], fontsize=17,
                                     family='Times New Roman')
axarr[1].set_yticklabels([3000, 5000, 7000, 9000], fontsize=17,
                                     family='Times New Roman')
axarr[2].set_yticklabels([1000, 3000, 5000, 7000, 9000], fontsize=17,
                                     family='Times New Roman')

# Legend
font = font_manager.FontProperties(family='Times New Roman',
                               style='normal', size=16)
leg = axarr[0].legend(epsilon_to_plot_legend, ncol=len(epsilon_to_plot), loc='upper center', bbox_to_anchor=(0.5,2),
                  prop=font)
leg.get_frame().set_alpha(0.4)
#plt.subplots_adjust(top=0.437)


# Save figure as pdf
if Debug.save_figures:
    plt.savefig(os.path.join(Config.path_figures, 'sensitivity_analysis.pdf'), format="pdf", bbox_inches="tight")