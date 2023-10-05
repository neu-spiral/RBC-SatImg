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



epsilon_evaluation_vector = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06,
                             0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

#def plot_results():
pickle_file_path = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.pkl')

results_sensitivity_analysis = pickle.load(open(pickle_file_path, 'rb'))

epsilon_vector = np.array(list(results_sensitivity_analysis.keys()))

RSIC_results_date2 = []
RGMM_results_date2 = []
RLR_results_date2 = []

RSIC_results_avg = []
RGMM_results_avg = []
RLR_results_avg = []

for eps_i in epsilon_evaluation_vector:
    results_i = results_sensitivity_analysis[eps_i]

    RSIC_results_date2.append(results_i[1]['RSIC'])
    RGMM_results_date2.append(results_i[1]['RGMM'])
    RLR_results_date2.append(results_i[1]['RLR'])

    RSIC_results_avg_aux = []
    RGMM_results_avg_aux = []
    RLR_results_avg_aux = []
    for label in results_i.keys():
        RSIC_results_avg_aux.append(results_i[label]['RSIC'])
        RGMM_results_avg_aux.append(results_i[label]['RGMM'])
        RLR_results_avg_aux.append(results_i[label]['RLR'])
    RSIC_results_avg.append(np.mean(RSIC_results_avg_aux))
    RGMM_results_avg.append(np.mean(RGMM_results_avg_aux))
    RLR_results_avg.append(np.mean(RLR_results_avg_aux))

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT 2020-06-10
# ------------------------------------------------------------------------
#path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.csv')
#df = pd.read_csv(path_results, decimal=',')

#epsilon_vector = df.iloc[0].to_numpy()[1:]
#RSIC_results = df.iloc[1].to_numpy()[1:]
#RGMM_results = df.iloc[2].to_numpy()[1:]
#RLR_results = df.iloc[3].to_numpy()[1:]

RSIC_results_date2 = np.array(RSIC_results_date2)*100
RGMM_results_date2 = np.array(RGMM_results_date2)*100
RLR_results_date2 = np.array(RLR_results_date2)*100

RSIC_results_avg = np.array(RSIC_results_avg)*100
RGMM_results_avg = np.array(RGMM_results_avg)*100
RLR_results_avg = np.array(RLR_results_avg)*100

# Create figure
f1, axarr = plt.subplots(1,2, figsize=(14, 4.5),gridspec_kw={'width_ratios': [3, 1]})

last_value_index = 36
# First plot (whole epsilon range)
axarr[0].plot(epsilon_vector[0:last_value_index], RSIC_results_date2[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,0].plot(epsilon_vector, 81.5*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0].plot(epsilon_vector[0:last_value_index], RGMM_results_date2[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[0,0].plot(epsilon_vector, 80.9*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0].plot(epsilon_vector[0:last_value_index], RLR_results_date2[0:last_value_index], marker='d', linewidth = 0.8, color= '#00CC00', markersize=3)
#axarr[0,0].plot(epsilon_vector, 82.01*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0].grid(alpha=0.2)

# Plot small range
limit = 22
axarr[1].plot(epsilon_vector[0:limit], RSIC_results_date2[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1].plot(epsilon_vector[np.argmax(RSIC_results_date2)], RSIC_results_date2[np.argmax(RSIC_results_date2)], marker='.', linewidth = 0.8, color='orange', zorder=10, markersize=14)
axarr[1].scatter(epsilon_vector[np.argmax(RSIC_results_date2)], RSIC_results_date2[np.argmax(RSIC_results_date2)], edgecolors='magenta', facecolors='none', s=90, zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1].plot(epsilon_vector[0:limit], RGMM_results_date2[0:limit], marker='x', linewidth = 0.8, color='blue')
axarr[1].scatter(epsilon_vector[np.argmax(RGMM_results_date2)], RGMM_results_date2[np.argmax(RGMM_results_date2)], edgecolors='magenta', facecolors='none', s=90, zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1].plot(epsilon_vector[0:limit], RLR_results_date2[0:limit], marker='d', linewidth = 0.8, color= '#00CC00', markersize=3)
axarr[1].scatter(epsilon_vector[np.argmax(RLR_results_date2)], RLR_results_date2[np.argmax(RLR_results_date2)], edgecolors='magenta', facecolors='none', s=90, zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1].grid(alpha=0.2)

# Labels and ticks
axarr[0].set_ylabel('Classification accuracy (%)', fontsize=18, fontname='Times New Roman')
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


#i=0
"""
for j in [0,1]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr[j].spines[axis].set_linewidth(0)
        axarr[j].tick_params(axis='x', colors='white')
"""

# Colors graphs
axarr[0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr[0].axvspan(0.5, 0.8, color='red', alpha=0.08)
axarr[0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[0].text(0.52, 88, 'Performance below \nbenchmark', fontsize=20, fontname='Times New Roman', color='r', weight='normal')
# Position labels
axarr[0].yaxis.set_label_coords(-0.05,0.5)
plt.tight_layout()

"""
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)
legend_strings=['RSIC', 'RGMM', 'RLR']
leg = axarr[0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5), prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)
"""
plt.subplots_adjust(wspace=0.001)
f1.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth_newdesign_a.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)


# ------------------------------------------------------------------------
# ----------------------- SUBPLOT AVERAGE
# ------------------------------------------------------------------------
#path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
#df = pd.read_csv(path_results, decimal=',')


#epsilon_vector = df.iloc[0].to_numpy()[1:]
#RSIC_results = df.iloc[1].to_numpy()[1:]
#RGMM_results = df.iloc[2].to_numpy()[1:]
#RLR_results = df.iloc[3].to_numpy()[1:]

# Create figure
f2, axarr2 = plt.subplots(1,2, figsize=(14, 4.5),gridspec_kw={'width_ratios': [3, 1]})


last_value_index = 36
axarr2[0].plot(epsilon_vector[0:last_value_index], RSIC_results_avg[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1,0].plot(epsilon_vector, 88.928*np.o nes(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr2[0].plot(epsilon_vector[0:last_value_index], RGMM_results_avg[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[1,0].plot(epsilon_vector, 89.038*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr2[0].plot(epsilon_vector[0:last_value_index], RLR_results_avg[0:last_value_index], marker='d', linewidth = 0.8, color= '#00CC00', markersize=3)
#axarr[1,0].plot(epsilon_vector, 88.71*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr2[0].grid(alpha=0.2)

#axarr2[0].set_ylabel('Classification accuracy (%)', fontsize=18, fontname='Times New Roman')
#axarr2[0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[1].set_ylim([70,95])

# Plot small range
limit = 22
axarr2[1].plot(epsilon_vector[0:limit], RSIC_results_avg[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr2[1].scatter(epsilon_vector[np.argmax(RSIC_results_avg)], RSIC_results_avg[np.argmax(RSIC_results_avg)], edgecolors='magenta', facecolors='none', s=90, zorder=10)
axarr2[1].plot(epsilon_vector[0:limit], RGMM_results_avg[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr2[1].scatter(epsilon_vector[np.argmax(RGMM_results_avg)], RGMM_results_avg[np.argmax(RGMM_results_avg)], edgecolors='magenta', facecolors='none', s=90, zorder=10)
axarr2[1].plot(epsilon_vector[0:limit], RLR_results_avg[0:limit], marker='d', linewidth = 0.8, color= '#00CC00', markersize=3)
#axarr[0,1].plot(epsilon_vector[0:limit], 82.01*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr2[1].scatter(epsilon_vector[np.argmax(RLR_results_avg)], RLR_results_avg[np.argmax(RLR_results_avg)], edgecolors='magenta', facecolors='none', s=90, zorder=10)

axarr2[1].grid(alpha=0.2)

#axarr[0,1].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
#axarr2[1].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')



font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=10)

#legend_strings=['RSIC', 'SIC', 'RGMM']
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
#leg.get_frame().set_alpha(0.4)


# Labels and ticks
axarr2[0].set_ylabel('Classification accuracy (%)', fontsize=18, fontname='Times New Roman')
axarr2[0].set_xlabel('$\epsilon$', fontsize=15.5, fontname='Times New Roman')
axarr2[1].set_xlabel('$\epsilon$', fontsize=15.5, fontname='Times New Roman')

for j in [0,1]:
    for axis in ['top', 'bottom', 'left', 'right']:
        axarr2[j].spines[axis].set_linewidth(0)
        for tick in axarr2[j].get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(15.5)
        for tick in axarr2[j].get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(15.5)


#i=0
#for j in [0,1]:
    #for axis in ['top', 'bottom', 'left', 'right']:
        #axarr2[j].spines[axis].set_linewidth(0)
        #axarr2[j].tick_params(axis='x', colors='white')

# Colors graphs
axarr2[0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr2[0].axvspan(0.5, 0.8, color='red', alpha=0.08)
axarr2[0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr2[1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
#axarr2[0].text(0.52, 88, 'Performance below \nbenchmark area', fontsize=14, fontname='Times New Roman', color='r', weight='normal')
# Position labels
axarr2[0].yaxis.set_label_coords(-0.05,0.5)
plt.tight_layout()
plt.subplots_adjust(wspace=0.025)


# Ticks
#axarr[0,1].set_yticks([80, 85, 90, 95])
#axarr[1,0].set_yticks(np.arange(84,96,2))
#axarr[1,1].set_yticks([80, 85, 90, 95])
#axarr[0,0].set_yticks([80, 85, 90, 95])

plt.tight_layout()



plt.subplots_adjust(wspace=0.02)
f2.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth_newdesign_b.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)

# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------
legend_strings=['RSIC', 'SIC', 'RGMM']
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)



path_results_average = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df_average = pd.read_csv(path_results_average, decimal=',')



def compute_results():
    """ --

    """
    # Initialize random seed
    random.seed(1)

    # Set logging path
    Debug.set_logging_file(time_now=datetime.now())

    # Instance of Image Reader object
    image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                 Config.image_dimensions[Config.scenario]['dim_y'])

    # Training Stage
    labels, gmm_densities, trained_lr_model = training_main(image_reader)

    results_sensitivity_analysis = dict()

    for eps_i in epsilon_evaluation_vector:
        Config.eps = eps_i
        Config.eps_LR = eps_i
        Config.eps_GMM = eps_i

        # Evaluation Stage
        # This stage includes the plotting of plot_results
        print(f"EVALUATION STARTS for epsilon = {Config.eps}")
        results_sensitivity_analysis[eps_i] = evaluation_main(gmm_densities, trained_lr_model, image_reader)

    pickle.dump(results_sensitivity_analysis, open(pickle_file_path, 'wb'))


"""
pickle_file_path = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.pkl')

results_sensitivity_analysis = pickle.load(open(pickle_file_path, 'rb'))

epsilon_vector = np.array(list(results_sensitivity_analysis.keys()))

RSIC_results_date2 = []
RGMM_results_date2 = []
RLR_results_date2 = []

RSIC_results_avg = []
RGMM_results_avg = []
RLR_results_avg = []

for eps_i in epsilon_evaluation_vector:
    results_i = results_sensitivity_analysis[eps_i]

    RSIC_results_date2.append(results_i[1]['RSIC'])
    RGMM_results_date2.append(results_i[1]['RGMM'])
    RLR_results_date2.append(results_i[1]['RLR'])

    RSIC_results_avg_aux = []
    RGMM_results_avg_aux = []
    RLR_results_avg_aux = []
    for label in results_i.keys():
        RSIC_results_avg_aux.append(results_i[label]['RSIC'])
        RGMM_results_avg_aux.append(results_i[label]['RGMM'])
        RLR_results_avg_aux.append(results_i[label]['RLR'])
    RSIC_results_avg.append(np.mean(RSIC_results_avg_aux))
    RGMM_results_avg.append(np.mean(RGMM_results_avg_aux))
    RLR_results_avg.append(np.mean(RLR_results_avg_aux))

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT 2020-06-10
# ------------------------------------------------------------------------
#path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth.csv')
#df = pd.read_csv(path_results, decimal=',')

#epsilon_vector = df.iloc[0].to_numpy()[1:]
#RSIC_results = df.iloc[1].to_numpy()[1:]
#RGMM_results = df.iloc[2].to_numpy()[1:]
#RLR_results = df.iloc[3].to_numpy()[1:]

RSIC_results_date2 = np.array(RSIC_results_date2)*100
RGMM_results_date2 = np.array(RGMM_results_date2)*100
RLR_results_date2 = np.array(RLR_results_date2)*100

RSIC_results_avg = np.array(RSIC_results_avg)*100
RGMM_results_avg = np.array(RGMM_results_avg)*100
RLR_results_avg = np.array(RLR_results_avg)*100

# Create figure
f, axarr = plt.subplots(2,2, figsize=(14, 6),gridspec_kw={'width_ratios': [3, 1]})

last_value_index = 36
# First plot (whole epsilon range)
axarr[0,0].plot(epsilon_vector[0:last_value_index], RSIC_results_date2[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,0].plot(epsilon_vector, 81.5*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,0].plot(epsilon_vector[0:last_value_index], RGMM_results_date2[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[0,0].plot(epsilon_vector, 80.9*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,0].plot(epsilon_vector[0:last_value_index], RLR_results_date2[0:last_value_index], marker='d', linewidth = 0.8, color= '#00CC00')
#axarr[0,0].plot(epsilon_vector, 82.01*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[0,0].grid(alpha=0.2)

axarr[0,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[0,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')

# Plot small range
limit = 22
axarr[0,1].plot(epsilon_vector[0:limit], RSIC_results_date2[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RGMM_results_date2[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[0,1].plot(epsilon_vector[0:limit], RLR_results_date2[0:limit], marker='d', linewidth = 0.8, color= '#00CC00')
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

# ------------------------------------------------------------------------
# ----------------------- SUBPLOT AVERAGE
# ------------------------------------------------------------------------
#path_results = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
#df = pd.read_csv(path_results, decimal=',')


#epsilon_vector = df.iloc[0].to_numpy()[1:]
#RSIC_results = df.iloc[1].to_numpy()[1:]
#RGMM_results = df.iloc[2].to_numpy()[1:]
#RLR_results = df.iloc[3].to_numpy()[1:]

last_value_index = 36
axarr[1,0].plot(epsilon_vector[0:last_value_index], RSIC_results_avg[0:last_value_index], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[1,0].plot(epsilon_vector, 88.928*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,0].plot(epsilon_vector[0:last_value_index], RGMM_results_avg[0:last_value_index], marker='x', linewidth = 0.8, color='blue')
#axarr[1,0].plot(epsilon_vector, 89.038*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,0].plot(epsilon_vector[0:last_value_index], RLR_results_avg[0:last_value_index], marker='d', linewidth = 0.8, color= '#00CC00')
#axarr[1,0].plot(epsilon_vector, 88.71*np.ones(epsilon_vector.size), marker='', linewidth = 0.6, color= '#00CC00', linestyle='--')
axarr[1,0].grid(alpha=0.2)

axarr[1,0].set_ylabel('Classification accuracy (%)', fontsize=11, fontname='Times New Roman')
axarr[1,0].set_xlabel('$\epsilon$', fontsize=11, fontname='Times New Roman')
#axarr[1].set_ylim([70,95])

# Plot small range
limit = 22
axarr[1,1].plot(epsilon_vector[0:limit], RSIC_results_avg[0:limit], marker='.', linewidth = 0.8, color='orange', zorder=10)
#axarr[0,1].plot(epsilon_vector[0:limit], 81.5*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='orange', zorder=10, linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RGMM_results_avg[0:limit], marker='x', linewidth = 0.8, color='blue')
#axarr[0,1].plot(epsilon_vector[0:limit], 80.9*np.ones(epsilon_vector[0:limit].size), marker='', linewidth = 0.6, color='blue', linestyle='--')
axarr[1,1].plot(epsilon_vector[0:limit], RLR_results_avg[0:limit], marker='d', linewidth = 0.8, color= '#00CC00')
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

axarr[0,0].set_title('                                                                          Results for acquisition date 2020-06-10', fontsize=14, fontname='Times New Roman')
axarr[1,0].set_title('                                                                        Average results for all dates', fontsize=14, fontname='Times New Roman')


# Colors graphs
axarr[0,0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr[1,0].axvline(x=0.5, color='red', linewidth='0.8', alpha=1, linestyle='--')
axarr[0,0].axvspan(0.5, 0.8, color='red', alpha=0.08)
axarr[1,0].axvspan(0.5, 0.8, color='red', alpha=0.08)
axarr[0,0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[1,0].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[0,1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[1,1].axvspan(0, 0.1, color='#CDF7FF', alpha=0.35)
axarr[0,0].text(0.52, 88, 'Performance below \nbenchmark area', fontsize=14, fontname='Times New Roman', color='r', weight='normal')

# Position labels
axarr[0,0].yaxis.set_label_coords(-0.05,0.5)
axarr[1,0].yaxis.set_label_coords(-0.05,0.5)

# Ticks
#axarr[0,1].set_yticks([80, 85, 90, 95])
axarr[1,0].set_yticks(np.arange(84,96,2))
#axarr[1,1].set_yticks([80, 85, 90, 95])
#axarr[0,0].set_yticks([80, 85, 90, 95])

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)


legend_strings=['RSIC', 'SIC', 'RGMM']
#leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.67,1.5),prop=font, ncol=3)
leg = axarr[0,0].legend(legend_strings, loc='upper center', bbox_to_anchor=(0.9,1.5),prop=font, ncol=len(legend_strings))
leg.get_frame().set_alpha(0.4)
# ------------------------------------------------------------------------
# ----------------------- SAVE FIGURE
# ------------------------------------------------------------------------

f.savefig(os.path.join(Config.path_results_figures,
                                      f'sensitivity_analysis_multiearth_newdesign.pdf'),
                         format="pdf", bbox_inches="tight", dpi=1000)


path_results_average = os.path.join(Config.path_evaluation_results, 'sensitivity_analysis_multiearth_average_summary.csv')
df_average = pd.read_csv(path_results_average, decimal=',')
"""