from matplotlib import pyplot as plt
import numpy as np
from tools.spectral_index import get_mean_std_scaled_index_model


def gaussian(x, mu, sig):
    return (
            1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


"""
# PLOT 3 SCENARIOS (DEPRECATED)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
thresholds = {'charles_river': [-1, -0.05, 0.35, 1], 'oroville_dam': [-1, 0.13, 1], 'multiearth': [-1, 0.65, 1]}
classes = {'charles_river': ['$C_t$ = 1 (water)', '$C_t$ = 2 (land)', '$C_t$ = 3 (vegetation)'], 'oroville_dam': ['$C_t$ = 1', '$C_t$ = 2'],
           'multiearth': ['$C_t$ = 1 (water)', '$C_t$ = 2 (!forest)']}
tag_experiment = {'charles_river': 'Experiment 2', 'oroville_dam': 'Experiment 1', 'multiearth': 'Experiment 3'}

fig, axs = plt.subplots(3,1)
scenarios = ['oroville_dam','charles_river',  'multiearth']
x_values = np.linspace(-3, 3, 120)
for scenario_idx, scenario in enumerate(scenarios):
    mu_vector, sig_vector = get_mean_std_scaled_index_model(thresholds[scenario][1:-1])
    for class_idx, _ in enumerate(classes[scenario]):
        mu =  mu_vector[class_idx]
        sig = sig_vector[class_idx]
        axs[scenario_idx].plot(x_values, gaussian(x_values, mu, sig))
    axs[scenario_idx].legend(classes[scenario])
    axs[scenario_idx].set_title(tag_experiment[scenario])
    axs[scenario_idx].set_ylabel('$p(y(z_t)|C_t)$', )
    for threshold_idx, threshold_i in enumerate(thresholds[scenario]):
        axs[scenario_idx].axvline(x=threshold_i, linewidth=0.8, linestyle=':')
        axs[scenario_idx].text(threshold_i+0.03, 0.8, f'$\\tau_{threshold_idx}$', rotation=0, verticalalignment='center', fontsize=12)
plt.show()
plt.tight_layout()
"""

# PLOT 3 SCENARIOS
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
thresholds = {'oroville_dam': [-1, 0.13, 1], 'multiearth': [-1, 0.65, 1]}
classes = {'oroville_dam': ['$C_t$ = 1 (land)', '$C_t$ = 2 (water)'],
           'multiearth': ['$C_t$ = 1 (land)', '$C_t$ = 2 (forest)']}
tag_experiment = {'oroville_dam': 'Water Mapping (Test sites 1a, 1b, 2)',
                  'multiearth': 'Deforestation Detection (Test site 3)'}

fig, axs = plt.subplots(2, 1, figsize=(6, 3.2))
scenarios = ['oroville_dam', 'multiearth']
x_values = np.linspace(-3, 3, 1000)
for scenario_idx, scenario in enumerate(scenarios):
    mu_vector, sig_vector = get_mean_std_scaled_index_model(thresholds[scenario][1:-1])
    for class_idx, _ in enumerate(classes[scenario]):
        mu = mu_vector[class_idx]
        sig = sig_vector[class_idx]
        axs[scenario_idx].plot(x_values, gaussian(x_values, mu, sig))
    axs[scenario_idx].legend(classes[scenario])
    axs[scenario_idx].set_title(tag_experiment[scenario])
    axs[scenario_idx].set_ylabel('$f_{C_t}(y(z_t)|C_t)$', )
    for threshold_idx, threshold_i in enumerate(thresholds[scenario]):
        axs[scenario_idx].axvline(x=threshold_i, linewidth=0.8, linestyle=':')
        axs[scenario_idx].text(threshold_i + 0.03, 0.8, f'$\\tau_{threshold_idx}$', rotation=0,
                               verticalalignment='center', fontsize=12)
plt.show()
plt.tight_layout()

path_save_image = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/paper_figures/thresholds_wm_dd.svg"
plt.savefig(path_save_image, format="svg", bbox_inches="tight")
