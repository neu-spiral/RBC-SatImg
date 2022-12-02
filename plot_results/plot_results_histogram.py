import pickle
import os

import matplotlib.pyplot as plt

from image_reader import ReadSentinel2
from configuration import Config, Debug
from matplotlib import colors


Config.scenario='oroville_dam'
Config.scene_id=2
epsilon_value = 0.05

index_results_to_plot = [1]
n_image_to_plot = 1
path_results = Config.path_evaluation_results
models_to_plot = ["WatNet", "DeepWaterMap", "Scaled Index", "Logistic Regression", "GMM"]

cmap = colors.ListedColormap(Config.cmap[Config.scenario])

# Instance of Image Reader object
image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                             Config.image_dimensions[Config.scenario]['dim_y'])
path_results = os.path.join(r"/Users/helena/Documents/test/evaluation_results/histogram_prediction", f'{Config.scenario}_{Config.scene_id}')

# Get coordinates of evaluated pixels
x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']

# Plot plot_results for Oroville Dam scenario
print(f"Results for Oroville Dam, k = {Config.scene_id}")

# Create figure
f, axarr= plt.subplots(len(models_to_plot), 2, figsize=(9, 12))

# Read stored evaluation plot_results to reproduce published figure
pickle_file_path = os.path.join(path_results, f"oroville_dam_{Config.scene_id}_image_{n_image_to_plot}_epsilon_{epsilon_value}_norm_constant_{Config.norm_constant}.pkl")
[prediction_float, date_string] = pickle.load(open(pickle_file_path, 'rb'))
print(pickle_file_path)

# Loop through models
for idx, model_i in enumerate(models_to_plot):
    prediction_i = prediction_float[model_i]
    axarr[idx, 0].hist(prediction_i[:, 0], bins=100)
    axarr[idx, 1].hist(prediction_i[:, 1], bins=100)
    axarr[idx, 0].set_ylabel(model_i)
    axarr[idx, 0].grid(alpha=0.2)
    axarr[idx, 1].grid(alpha=0.2)

axarr[0, 0].title.set_text('p(class 0|pixel)')
axarr[0, 1].title.set_text('p(class 1|pixel)')
f.suptitle(f"Histogram posterior all samples date {date_string}, epsilon {epsilon_value}, {Config.scenario} {Config.scene_id}, norm constant = {Config.norm_constant}")

# Save figure as pdf
if Debug.save_figures:
    path_image_save = os.path.join(Config.path_figures, f'histogram_y_pred_{Config.scenario}_{Config.scene_id}_epsilon_{epsilon_value}_norm_constant_{Config.norm_constant}.pdf')
    plt.savefig(path_image_save, format="pdf", bbox_inches="tight")