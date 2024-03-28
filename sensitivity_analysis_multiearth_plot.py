import random
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from configuration import Config, Debug
from image_reader import ReadSentinel2
from training import training_main
from evaluation import evaluation_main
from datetime import datetime
import matplotlib.font_manager as font_manager

# --------------------------------------------------------------------
# TO BE CHANGED BY USER
# --------------------------------------------------------------------
# TODO: Change accordingly:
# path_save_figure = os.path.join(Config.path_evaluation_results, "sensitivity_analysis", "multiearth", "sensitivity_analysis_multiearth.pdf")
Config.path_evaluation_results = os.path.join(Config.path_evaluation_results, "sensitivity_analysis", f'{Config.scenario}', 'results')
# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
epsilon_vector = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]


# RSIC_results_date2 = []
# RGMM_results_date2 = []
# RLR_results_date2 = []

RSIC_results_avg = []
RGMM_results_avg = []
RLR_results_avg = []
RDWM_results_avg = []
RWN_results_avg = []

offset_model = 5
for eps_i in epsilon_vector:

    results_i = pickle.load(open(os.path.join(Config.path_evaluation_results, f"eps_{eps_i}"), 'rb'))
    results_acc = results_i["balanced_accuracy"]
    #
    # RSIC_results_date2.append(results_i[1]['RSIC'])
    # RGMM_results_date2.append(results_i[1]['RGMM'])
    # RLR_results_date2.append(results_i[1]['RLR'])

    RSIC_results_avg.append(np.mean(results_acc[:,offset_model]))
    RGMM_results_avg.append(np.mean(results_acc[:,offset_model+1]))
    RLR_results_avg.append(np.mean(results_acc[:,offset_model+2]))
    RDWM_results_avg.append(np.mean(results_acc[:, offset_model + 3]))
    RWN_results_avg.append(np.mean(results_acc[:, offset_model + 4]))

    # for label in results_i.keys():
    #     RSIC_results_avg_aux.append(results_i[label]['RSIC'])
    #     RGMM_results_avg_aux.append(results_i[label]['RGMM'])
    #     RLR_results_avg_aux.append(results_i[label]['RLR'])
    # RSIC_results_avg.append(np.mean(RSIC_results_avg_aux))
    # RGMM_results_avg.append(np.mean(RGMM_results_avg_aux))
    # RLR_results_avg.append(np.mean(RLR_results_avg_aux))

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT
# ------------------------------------------------------------------------

# Create figure
f1, axarr = plt.subplots(1,2, figsize=(18, 4.5),gridspec_kw={'width_ratios': [4, 1]})

last_value_index = len(epsilon_vector)-2
# First plot (whole epsilon range)
axarr[0].plot(epsilon_vector[0:last_value_index], RSIC_results_avg[0:last_value_index], marker='.',markersize=6,  linewidth = 0.8, color='orange', zorder=10)
axarr[0].plot(epsilon_vector[0:last_value_index], RGMM_results_avg[0:last_value_index], marker='x', markersize=5,  linewidth = 0.8, color='blue')
axarr[0].plot(epsilon_vector[0:last_value_index], RLR_results_avg[0:last_value_index], marker='d',markersize=4,  linewidth = 0.8, color= '#00CC00')
axarr[0].plot(epsilon_vector[0:last_value_index], RDWM_results_avg[0:last_value_index], marker='>', markersize=5,  linewidth = 0.8, color='magenta')
axarr[0].plot(epsilon_vector[0:last_value_index], RWN_results_avg[0:last_value_index], marker='<',markersize=4,  linewidth = 0.8, color= 'red')
axarr[0].grid(alpha=0.2)

# Plot small range
limit = 20
# RSIC
axarr[1].plot(epsilon_vector[0:limit], RSIC_results_avg[0:limit], marker='.' ,markersize=6, linewidth = 0.8, color='orange', zorder=10)
axarr[1].scatter(epsilon_vector[np.argmax(RSIC_results_avg)], RSIC_results_avg[np.argmax(RSIC_results_avg)], edgecolors='black', facecolors='none', s=90, zorder=10)
# RGMM
axarr[1].plot(epsilon_vector[0:limit], RGMM_results_avg[0:limit], marker='x',markersize=5,  linewidth = 0.8, color='blue')
axarr[1].scatter(epsilon_vector[np.argmax(RGMM_results_avg)], RGMM_results_avg[np.argmax(RGMM_results_avg)], edgecolors='black', facecolors='none', s=90, zorder=10)
# RLR
axarr[1].plot(epsilon_vector[0:limit], RLR_results_avg[0:limit], marker='d',markersize=4,  linewidth = 0.8, color= '#00CC00')
axarr[1].scatter(epsilon_vector[np.argmax(RLR_results_avg)], RLR_results_avg[np.argmax(RLR_results_avg)], edgecolors='black', facecolors='none', s=90, zorder=10)
# RDWM
axarr[1].plot(epsilon_vector[0:limit], RDWM_results_avg[0:limit], marker='>',markersize=4,  linewidth = 0.8, color= 'magenta')
axarr[1].scatter(epsilon_vector[np.argmax(RDWM_results_avg)], RDWM_results_avg[np.argmax(RDWM_results_avg)], edgecolors='black', facecolors='none', s=90, zorder=10)
# RWN
axarr[1].plot(epsilon_vector[0:limit], RWN_results_avg[0:limit], marker='>',markersize=4,  linewidth = 0.8, color= 'red')
axarr[1].scatter(epsilon_vector[np.argmax(RWN_results_avg)], RWN_results_avg[np.argmax(RWN_results_avg)], edgecolors='black', facecolors='none', s=90, zorder=10)
# grid
axarr[1].grid(alpha=0.2)

# Labels and ticks
axarr[0].set_ylabel('Classification accuracy (%)', fontsize=20, fontname='Times New Roman')
axarr[0].set_xlabel('$\epsilon$', fontsize=15.5, fontname='Times New Roman')
axarr[1].set_xlabel('$\epsilon$', fontsize=15.5, fontname='Times New Roman')

for j in [0,1]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[j].spines[axis].set_linewidth(0)
        for tick in axarr[j].get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(15.5)
        for tick in axarr[j].get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(15.5)

# # Colors graphs
# axarr[0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
# axarr[0].axvspan(0.5, 0.8, color='red', alpha=0.08)
# axarr[0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
# axarr[1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
# axarr[0].text(0.52, 88, 'Performance below \nbenchmark', fontsize=20, fontname='Times New Roman', color='r', weight='normal')
#
# # Position labels
# axarr[0].yaxis.set_label_coords(-0.05,0.5)
# plt.tight_layout()
#
# plt.subplots_adjust(wspace=0.001)


# Colors graphs
axarr[0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
#axarr2[0].axvspan(0.5, 0.8, color='red', alpha=0.08)
axarr[0].axvspan(0, 0.1, color='#d9dada', alpha=0.35)
axarr[1].axvspan(0, epsilon_vector[limit-1], color='#d9dada', alpha=0.35)

# Position labels
axarr[0].yaxis.set_label_coords(-0.05,0.5)
plt.subplots_adjust(wspace=0.025)
plt.tight_layout(w_pad=-0.2)

# Legend
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=18)
legend_strings=['RSIC', "RGMM", 'RLR']
leg = axarr[0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.62,1.2),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)

# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------
path_save_figure = os.path.join("/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/classification/multiearth_3/sensitivity_analysis","results_multiearth.pdf")
f1.savefig(path_save_figure, bbox_inches='tight', format='pdf', dpi=1000)
