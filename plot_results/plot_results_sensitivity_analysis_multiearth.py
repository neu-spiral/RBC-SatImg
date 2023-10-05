import os
import glob
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from configuration import Config, Debug
from collections import OrderedDict

"""
# ------------------------------------------------------------------------
# ----------------------- SUBPLOT 2020-06-10
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.csv')
df = pd.read_csv(path_results, decimal=',')

epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

# Create figure
f, axarr = plt.subplots(2,1, figsize=(10, 6))

# First plot (whole epsilon range)
axarr[0].plot(epsilon_vector, RSIC_results, marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,0].plot(epsilon_vector, 81.5*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0].plot(epsilon_vector, RGMM_results, marker='x', linewidth = 0.8, color='blue')
#axarr[0,0].plot(epsilon_vector, 80.9*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0].plot(epsilon_vector, RLR_results, marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,0].plot(epsilon_vector, 82.01*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0].grid(alpha=0.2)

axarr[0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')

# Plot small range
limit = 12
axarr[1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
axarr[1].set_yticks([85, 90, 95])
axarr[0,2].set_yticks([75, 80, 85])


# Plot small range (second)
limit_1 = 14
limit_2 = 19
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RSIC_results[limit_1:limit_2], marker='.', linewidth = 0.8, color='orange', zorder=10)
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 81.5*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RGMM_results[limit_1:limit_2], marker='x', linewidth = 0.8, color='blue')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 80.9*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RLR_results[limit_1:limit_2], marker='*', linewidth = 0.8, color= '#00CC00')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 82.01*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,2].grid(alpha=0.2)


#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,2].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[0,1].set_xticks([0.0, 0.1, 0.2, 0.3])


#axarr[0,1].xaxis.set_ticks([])
#axarr[0,1].set_xlabel('', fontsize=11, fontname='Times New Roman')
#axarr[0,0].xaxis.set_ticks([])
#axarr[0,0].set_xlabel('', fontsize=11, fontname='Times New Roman')

axarr[0,1].set_xlabel('', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('', fontsize=11, fontname='Times New Roman')
axarr[0,2].set_xlabel('', fontsize=11, fontname='Times New Roman')

# Remove axis
for i in [0,1]:
    for j in [0,1,2]:
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[i,j].spines[axis].set_linewidth(0)


i=0
for j in [0,1,2]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[i,j].spines[axis].set_linewidth(0)
        axarr[i,j].tick_params(axis='x', colors='white')

axarr[0,0].axvline(x=0.5, color='black', linewidth='0.3', alpha=0.3, linestyle='--')
axarr[1,0].axvline(x=0.5, color='black', linewidth='0.3', alpha=0.3, linestyle='--')

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT AVERAGE
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df = pd.read_csv(path_results, decimal=',')


epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

axarr[1,0].plot(epsilon_vector, RSIC_results, marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1,0].plot(epsilon_vector, 88.928*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,0].plot(epsilon_vector, RGMM_results, marker='x', linewidth = 0.8, color='blue')
#axarr[1,0].plot(epsilon_vector, 89.038*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,0].plot(epsilon_vector, RLR_results, marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[1,0].plot(epsilon_vector, 88.71*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,0].grid(alpha=0.2)

axarr[1,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[1].set_ylim([70,95])

# Plot small range
limit = 12
axarr[1,1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')




# Plot small range (second)
limit_1 = 14
limit_2 = 19
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RSIC_results[limit_1:limit_2], marker='.', linewidth = 0.8, color='orange', zorder=10)
axarr[1,2].plot(epsilon_vector[limit_1:limit_2],88.928*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RGMM_results[limit_1:limit_2], marker='x', linewidth = 0.8, color='blue')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], 89.038*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RLR_results[limit_1:limit_2], marker='*', linewidth = 0.8, color= '#00CC00')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], 88.71*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,2].grid(alpha=0.2)


#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,2].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[0,1].set_xticks([0.0, 0.1, 0.2, 0.3])

font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

legend_strings=['RSIC', 'SIC', 'RGMM', 'GMM', 'RLR', 'LR']
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)

for i in [0,1]:
    for j in [0, 1,2]:
        for tick in axarr[i,j].get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axarr[i,j].get_yticklabels():
            tick.set_fontname("Times New Roman")

axarr[0,0].set_title('                                                                                          Results for acquisition date 2020-06-10', fontsize=13, fontname='Times New Roman')
axarr[1,0].set_title('                                                                                                Average results for all dates in the quantitative analysis', fontsize=13, fontname='Times New Roman')

plt.tight_layout()
# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------

f.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)


path_results_average = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df_average = pd.read_csv(path_results_average, decimal=',')

"""


"""
# ------------------------------------------------------------------------
# ----------------------- SUBPLOT 2020-06-10
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.csv')
df = pd.read_csv(path_results, decimal=',')

epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

# Create figure
f, axarr = plt.subplots(2,3, figsize=(10, 6),gridspec_kw={'width_ratios': [3, 1, 1]})

# First plot (whole epsilon range)
axarr[0,0].plot(epsilon_vector, RSIC_results, marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,0].plot(epsilon_vector, 81.5*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,0].plot(epsilon_vector, RGMM_results, marker='x', linewidth = 0.8, color='blue')
#axarr[0,0].plot(epsilon_vector, 80.9*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,0].plot(epsilon_vector, RLR_results, marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,0].plot(epsilon_vector, 82.01*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,0].grid(alpha=0.2)

axarr[0,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')

# Plot small range
limit = 12
axarr[0,1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
axarr[0,1].set_yticks([85, 90, 95])
axarr[0,2].set_yticks([75, 80, 85])


# Plot small range (second)
limit_1 = 14
limit_2 = 19
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RSIC_results[limit_1:limit_2], marker='.', linewidth = 0.8, color='orange', zorder=10)
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 81.5*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RGMM_results[limit_1:limit_2], marker='x', linewidth = 0.8, color='blue')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 80.9*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], RLR_results[limit_1:limit_2], marker='*', linewidth = 0.8, color= '#00CC00')
axarr[0,2].plot(epsilon_vector[limit_1:limit_2], 82.01*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,2].grid(alpha=0.2)


#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,2].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[0,1].set_xticks([0.0, 0.1, 0.2, 0.3])


#axarr[0,1].xaxis.set_ticks([])
#axarr[0,1].set_xlabel('', fontsize=11, fontname='Times New Roman')
#axarr[0,0].xaxis.set_ticks([])
#axarr[0,0].set_xlabel('', fontsize=11, fontname='Times New Roman')

axarr[0,1].set_xlabel('', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('', fontsize=11, fontname='Times New Roman')
axarr[0,2].set_xlabel('', fontsize=11, fontname='Times New Roman')

# Remove axis
for i in [0,1]:
    for j in [0,1,2]:
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[i,j].spines[axis].set_linewidth(0)


i=0
for j in [0,1,2]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[i,j].spines[axis].set_linewidth(0)
        axarr[i,j].tick_params(axis='x', colors='white')

axarr[0,0].axvline(x=0.5, color='black', linewidth='0.3', alpha=0.3, linestyle='--')
axarr[1,0].axvline(x=0.5, color='black', linewidth='0.3', alpha=0.3, linestyle='--')

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT AVERAGE
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df = pd.read_csv(path_results, decimal=',')


epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

axarr[1,0].plot(epsilon_vector, RSIC_results, marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1,0].plot(epsilon_vector, 88.928*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,0].plot(epsilon_vector, RGMM_results, marker='x', linewidth = 0.8, color='blue')
#axarr[1,0].plot(epsilon_vector, 89.038*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,0].plot(epsilon_vector, RLR_results, marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[1,0].plot(epsilon_vector, 88.71*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,0].grid(alpha=0.2)

axarr[1,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[1].set_ylim([70,95])

# Plot small range
limit = 12
axarr[1,1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')




# Plot small range (second)
limit_1 = 14
limit_2 = 19
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RSIC_results[limit_1:limit_2], marker='.', linewidth = 0.8, color='orange', zorder=10)
axarr[1,2].plot(epsilon_vector[limit_1:limit_2],88.928*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RGMM_results[limit_1:limit_2], marker='x', linewidth = 0.8, color='blue')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], 89.038*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], RLR_results[limit_1:limit_2], marker='*', linewidth = 0.8, color= '#00CC00')
axarr[1,2].plot(epsilon_vector[limit_1:limit_2], 88.71*np.ones(epsilon_vector[limit_1:limit_2].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,2].grid(alpha=0.2)


#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,2].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[0,1].set_xticks([0.0, 0.1, 0.2, 0.3])

font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

legend_strings=['RSIC', 'SIC', 'RGMM', 'GMM', 'RLR', 'LR']
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)

for i in [0,1]:
    for j in [0, 1,2]:
        for tick in axarr[i,j].get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axarr[i,j].get_yticklabels():
            tick.set_fontname("Times New Roman")

axarr[0,0].set_title('                                                                                          Results for acquisition date 2020-06-10', fontsize=13, fontname='Times New Roman')
axarr[1,0].set_title('                                                                                                Average results for all dates in the quantitative analysis', fontsize=13, fontname='Times New Roman')

plt.tight_layout()
# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------

f.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)


path_results_average = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df_average = pd.read_csv(path_results_average, decimal=',')

"""


# ------------------------------------------------------------------------
# ----------------------- SUBPLOT 2020-06-10
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.csv')
df = pd.read_csv(path_results, decimal=',')

epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

# Create figure
f, axarr = plt.subplots(2,2, figsize=(13, 6),gridspec_kw={'width_ratios': [3, 1.2]})

last_value_index = 19
# First plot (whole epsilon range)
axarr[0,0].plot(epsilon_vector[0:last_value_index], RSIC_results[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,0].plot(epsilon_vector, 81.5*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,0].plot(epsilon_vector[0:last_value_index], RGMM_results[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[0,0].plot(epsilon_vector, 80.9*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,0].plot(epsilon_vector[0:last_value_index], RLR_results[0:last_value_index], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,0].plot(epsilon_vector, 82.01*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,0].grid(alpha=0.2)

axarr[0,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')

# Plot small range
limit = 12
axarr[0,1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
axarr[0,1].set_xlabel('', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('', fontsize=11, fontname='Times New Roman')

# Remove axis
for i in [0,1]:
    for j in [0,1]:
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[i,j].spines[axis].set_linewidth(0)


i=0
for j in [0,1]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[i,j].spines[axis].set_linewidth(0)
        axarr[i,j].tick_params(axis='x', colors='white')

# Ticks
#axarr[0,1].set_yticks([80, 85, 90, 95])
axarr[0,0].set_yticks([80, 82.5, 85, 87.5, 90, 92.5, 95])
#axarr[1,1].set_yticks([80, 85, 90, 95])
#axarr[0,0].set_yticks([80, 85, 90, 95])

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT AVERAGE
# ------------------------------------------------------------------------
path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df = pd.read_csv(path_results, decimal=',')


epsilon_vector = df.iloc[0].to_numpy()[1:]
RSIC_results = df.iloc[1].to_numpy()[1:]
RGMM_results = df.iloc[2].to_numpy()[1:]
RLR_results = df.iloc[3].to_numpy()[1:]

last_value_index = 19
axarr[1,0].plot(epsilon_vector[0:last_value_index], RSIC_results[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1,0].plot(epsilon_vector, 88.928*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,0].plot(epsilon_vector[0:last_value_index], RGMM_results[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[1,0].plot(epsilon_vector, 89.038*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,0].plot(epsilon_vector[0:last_value_index], RLR_results[0:last_value_index], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[1,0].plot(epsilon_vector, 88.71*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,0].grid(alpha=0.2)

axarr[1,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[1].set_ylim([70,95])

# Plot small range
limit = 12
axarr[1,1].plot(epsilon_vector[0:limit], RSIC_results[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RGMM_results[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RLR_results[0:limit], marker='*', linewidth = 0.8, color= '#00CC00')
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')



font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

#legend_strings=['RSIC', 'SIC', 'RGMM']
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
#leg.get_frame().set_alpha(0.4)

for i in [0,1]:
    for j in [0, 1]:
        for tick in axarr[i,j].get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axarr[i,j].get_yticklabels():
            tick.set_fontname("Times New Roman")

axarr[0,0].set_title('                                                            Results for acquisition date 2020-06-10', fontsize=13, fontname='Times New Roman')
axarr[1,0].set_title('                                                                 Average results for all dates in the quantitative analysis', fontsize=13, fontname='Times New Roman')


# Colors graphs
axarr[0,0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr[1,0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr[0,0].axvspan(0.5, 0.61, color='red', alpha=0.08)
axarr[1,0].axvspan(0.5, 0.61, color='red', alpha=0.08)
axarr[0,0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[1,0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[0,1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[1,1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[0,0].text(0.505, 88, 'Performance below \nbenchmark', fontsize=13, fontname='Times New Roman', color='r')

# Position labels
axarr[0,0].yaxis.set_label_coords(-0.07,0.5)
axarr[1,0].yaxis.set_label_coords(-0.07,0.5)

plt.tight_layout()

legend_strings=['RSIC', 'SIC', 'RGMM']
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)
# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------

f.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)


path_results_average = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df_average = pd.read_csv(path_results_average, decimal=',')

