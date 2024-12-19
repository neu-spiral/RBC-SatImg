import os
import glob
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from configuration import Config, Debug
from collections import OrderedDict

# --------------------------------------------------------------------
# TO BE CHANGED BY USER
# --------------------------------------------------------------------
# TODO: Change accordingly:
path_files = os.path.join(Config.path_evaluation_results, "sensitivity_analysis", "oroville_dam")
path_save_figure = os.path.join(Config.path_evaluation_results, "sensitivity_analysis", "figures")
legend = True
save_figure = True

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
# Read the dates
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

results = dict()
models = ["GMM", "LR", "SIC", "RGMM", "RLR", "RSIC"]
num_images_eval = 42

dates = pd.read_csv(os.path.join(path_files, "dates_OD_0919.csv"), names=['date'])
dates_array = dates.to_numpy().reshape(dates.shape[0])

# Change date format x-ticks
for idx_date, date_i in enumerate(dates_array):
    [month, day, year] = date_i.split("/")
    dates_array[idx_date] = f"{year}-{month}-{day}"

# Create figure (gridspec)
f1, ax1 = plt.subplots(1,1,figsize=(15, 3.5))
f2, ax2 = plt.subplots(1,1,figsize=(15, 3.5))
f3, ax3 = plt.subplots(1,1,figsize=(15, 3.55))
list_axis = [ax1, ax2, ax3]

# Define style
linestyles = ['-', '-', '-', '-', '-', '--', '--', '--', '--']
markerstyles = ['.', 'x', 'd', '<', 's', '.', 'x', 'd', 'o']
colors = ['#43A047', '#00CC00', '#0033FF', '#F44336', '#FF00FF', '#9E9E9E', '#F1C40F', '#00FFFF']
colors = ['blue', 'orange', '#589b8c', 'magenta', '#6CF75C', 'red', 'cyan', '#b342ff', 'grey']
marker_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 1]


# --------------------------------------------------------------------
# LOOP OVER RESULTS
# --------------------------------------------------------------------

for folder_name in os.scandir(path_files):
    # If the file contains desired plot_figures, proceed to store and plot
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


# --------------------------------------------------------------------
# PLOT RESULTS
# --------------------------------------------------------------------

# Order dictionary
results_sorted = OrderedDict(sorted(results.items()))
models_to_plot = ["RGMM", "RLR", "RSIC"]
epsilon_to_plot = [0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1]
epsilon_to_plot_legend  = []
for eps_i in epsilon_to_plot:
    epsilon_to_plot_legend.append(r"$\epsilon$ = "+f"{eps_i}")
epsilon_to_plot_legend[-1] = 'Instantaneous Classifier'
epsilon_benchmark = 0.5

for idx_model, model in enumerate(models_to_plot):
    plot_count = 0
    #axarr[idx_model] = plt.subplot(ax[idx_model])
    # Loop plot_figures
    for epsilon_i in epsilon_to_plot:
        if epsilon_i == 1:
            list_axis[idx_model].plot(dates_array[1:], results_sorted[epsilon_benchmark][model[1:]], linewidth=0.7,
                                  linestyle=linestyles[plot_count], marker=markerstyles[plot_count],
                                  markersize=marker_sizes[plot_count], color=colors[plot_count])
        else:
            list_axis[idx_model].plot(dates_array[1:], results_sorted[epsilon_i][model], linewidth=0.7,
                                  linestyle=linestyles[plot_count], marker=markerstyles[plot_count],
                                  markersize=marker_sizes[plot_count], color=colors[plot_count])
        plot_count = plot_count + 1

    # Set plot grids
    list_axis[idx_model].grid(alpha=0.2)

    # Rotate ticks (which include date labels) and center
    list_axis[idx_model].set_xticklabels(list_axis[idx_model].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor",fontsize=14, family='Times New Roman')
    labels = list_axis[idx_model].get_yticklabels()

    # Adjust ylim
    start, end = list_axis[idx_model].get_ylim()
    start = np.round(start / 1e3) * 1e3
    end = np.round((end + 1200) / 1e3) * 1e3
    list_axis[idx_model].yaxis.set_ticks(np.arange(start, end, 2e3))

    # Add x and y labels
    # list_axis[idx_model].set_ylabel('Water pixels', fontsize=18)
    f1.text(0.065, 0.5, 'Water pixels', va='center', rotation='vertical', fontsize=18)
    f2.text(0.065, 0.5, 'Water pixels', va='center', rotation='vertical', fontsize=18)
    f3.text(0.065, 0.5, 'Water pixels', va='center', rotation='vertical', fontsize=18)
    list_axis[idx_model].yaxis.set_label_coords(-0.07, 0.45)

    # Remove axis
    for axis in ['top', 'bottom', 'left', 'right']:
        list_axis[idx_model].spines[axis].set_linewidth(0)
    if idx_model !=1:
        list_axis[idx_model].tick_params(axis='x', colors='white')

list_axis[0].set_yticklabels([2000, 4000, 6000, 8000, 10000], fontsize=16)
list_axis[1].set_yticklabels([3000, 5000, 7000, 9000], fontsize=16)
list_axis[2].set_yticklabels([1000, 3000, 5000, 7000, 9000], fontsize=16)

# --------------------------------------------------------------------
# LEGEND
# --------------------------------------------------------------------
if legend:
    leg = list_axis[2].legend(epsilon_to_plot_legend, ncol=5, loc='upper center', bbox_to_anchor=(0.5,1.4), fontsize=16)
    leg.get_frame().set_alpha(0.4)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1)
        legobj.set_markersize(5)

# --------------------------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------------------------
if save_figure:
    f1.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_GMM.pdf"), format="pdf",
               bbox_inches="tight")
    f2.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_RLR.pdf"), format="pdf", bbox_inches="tight")
    f3.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_RSIC.pdf"), format="pdf", bbox_inches="tight")

    #svg format
    f1.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_GMM.svg"), format="svg",
               bbox_inches="tight")
    f2.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_RLR.svg"), format="svg", bbox_inches="tight")
    f3.savefig(os.path.join(path_save_figure, "sensitivity_analysis_oroville_dam_RSIC.svg"), format="svg", bbox_inches="tight")

