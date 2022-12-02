import pickle
import os

import matplotlib.pyplot as plt

from image_reader import ReadSentinel2
from configuration import Config, Debug
from matplotlib import colors
from figures import get_rgb_image

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------CONFIGURATION (user must check)-----------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# Selection of the scenario and scene
Config.scenario = "oroville_dam"  # charles_river | oroville_dam # TODO: CHANGE
Config.scene_id = 2  # 0 | 1 | 2 | 3 # TODO: CHANGE
#   - scene_id = 0 if wanting to process the whole image (DEPRECATED)
#   - scene_id = 1 for scene A in Oroville Dam (water stream)
#   - scene_id = 2 for scene B in Oroville Dam (peninsula)
#   - scene_id = 3 for scene C (Charles River)
# Results Visualization Options
plot_images_manuscript = True  # TODO: CHANGE
# set to True if wanting to reproduce the same results presented in the manuscript
# set to False if wanting to reproduce the same results presented at the end of the arxiv version (containing all dates)

# Path with results
path_results = os.path.join(Config.path_evaluation_results, "classification", f'{Config.scenario}_{Config.scene_id}')
path_images = os.path.join(Config.path_sentinel_images, f'{Config.scenario}', 'evaluation')

# offset evaluation images to plot
offset_eval_images = 1  # set to 1 because we want to skip the first evaluation dates in both scenarios

# Epsilon value for this setup (epsilon determines the transition probability value)
epsilon_value = 0.05

# Scaling RGB
scaling_rgb = {2: 1.2, 1: 1.9, 3: 1}

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ---------------------------PLOTS----------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# Index of images plotted after evaluation
if plot_images_manuscript:
    # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
    index_plot = {1: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 2: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
                  3: [0, 4, 8, 12, 16, 20, 24]}  # if wanting to plot the dates shown in the manuscript
else:
    # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
    index_plot = {2: [*range(0, 41, 1)], 1: [*range(0, 41, 1)], 3: [*range(0, 27, 1)]}

# Index of images to plot is updated according to the image index offset
index_results_to_plot = [x + Config.offset_eval_images for x in index_plot[Config.scene_id]]
n_results_to_plot = len(index_results_to_plot)

# Read cmap from configuration file
cmap = colors.ListedColormap(Config.cmap[Config.scenario])

# Instance of Image Reader object
image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                             Config.image_dimensions[Config.scenario]['dim_y'])
# Get coordinates of evaluated pixels
x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']

# Plot plot_results for Oroville Dam scenario
if Config.scene_id == 1 or Config.scene_id == 2:
    print(f"Results for Oroville Dam, k = {Config.scene_id}")

    # Vector with models
    plot_legend = ['SIC', 'GMM', 'LR', 'DWM', 'WN', 'RSIC', 'RGMM', 'RLR', 'RDWM', 'RWN', 'RGB']
    models = ['Scaled Index', 'GMM', 'Logistic Regression']  # TODO: Use this vector to clean code

    # Create figure (size changes depending on the amount of plotted images)
    if plot_images_manuscript:
        # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
        f, axarr = plt.subplots(n_results_to_plot, 11, figsize=(9, 10))
    else:
        # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
        f, axarr = plt.subplots(n_results_to_plot, 11, figsize=(9, 30))

    for image_i in range(0, n_results_to_plot):
        # Read image again to be able to get RGB image
        image_all_bands, date_string = image_reader.read_image(path=path_images,
                                                               image_idx=index_results_to_plot[image_i])

        # Get RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)
        # Scaling to increase illumination
        rgb_image = rgb_image * scaling_rgb[Config.scene_id]

        # Read stored evaluation plot_results to reproduce published figure
        pickle_file_path = os.path.join(path_results,
                                        f"oroville_dam_{Config.scene_id}_image_{index_results_to_plot[image_i]}_epsilon_{epsilon_value}_norm_constant_{Config.norm_constant}.pkl")
        [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))
        print(pickle_file_path)

        # Plot plot_results
        axarr[image_i, 0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 0].get_yaxis().set_ticks([])
        axarr[image_i, 0].get_xaxis().set_ticks([])

        # Remove axis
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[image_i, 0].spines[axis].set_linewidth(0)
        axarr[image_i, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

        # The label position must be changed accordingly, considering the amount of images plotted
        if ~plot_images_manuscript:
            # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
            axarr[image_i, 0].yaxis.set_label_coords(-0.9, 0.35)
        else:
            # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
            axarr[image_i, 0].yaxis.set_label_coords(-0.9, 0.4)

        axarr[image_i, 1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 1].axis('off')
        axarr[image_i, 2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 2].axis('off')
        axarr[image_i, 3].imshow(y_pred["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 3].axis('off')
        axarr[image_i, 4].imshow(y_pred["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 4].axis('off')
        axarr[image_i, 5].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]],
                                 cmap, aspect='auto')
        axarr[image_i, 5].axis('off')
        axarr[image_i, 6].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 6].axis('off')
        axarr[image_i, 7].imshow(
            predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
            aspect='auto')
        axarr[image_i, 7].axis('off')
        axarr[image_i, 8].imshow(predicted_image["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]],
                                 cmap, aspect='auto')
        axarr[image_i, 8].axis('off')
        axarr[image_i, 9].imshow(predicted_image["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap,
                                 aspect='auto')
        axarr[image_i, 9].axis('off')
        axarr[image_i, 10].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], aspect='auto')
        axarr[image_i, 10].axis('off')

    # Set figure labels
    for idx, label in enumerate(plot_legend):
        axarr[0, idx].title.set_fontfamily('Times New Roman')
        axarr[0, idx].title.set_fontsize(12)
        axarr[0, idx].title.set_text(plot_legend[idx])

# Plot plot_results for Charles River scenario
elif Config.scene_id == 3:
    print("Results for Charles River, Study Area C")

    # Create figure (size changes depending on the amount of plotted images)
    if plot_images_manuscript:
        # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
        f, axarr = plt.subplots(n_results_to_plot, 7, figsize=(9, 10))
    else:
        # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
        f, axarr = plt.subplots(n_results_to_plot, 7, figsize=(9, 30))

    # Vectors with models to plot
    plot_legend = ['SIC', 'GMM', 'LR', 'RSIC', 'RGMM', 'RLR', 'RGB']
    models = ['Scaled Index', 'GMM', 'Logistic Regression']  # TODO: Use this vector to clean code

    # Vectors with models without recursion version
    models = ['']

    for image_i in range(0, n_results_to_plot):

        # Read image again to be able to get RGB image
        image_all_bands, date_string = image_reader.read_image(path=path_images,
                                                               image_idx=index_results_to_plot[image_i])

        # Get RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)

        # Read stored evaluation plot_results to reproduce published figure
        pickle_file_path = os.path.join(path_results,
                                        f"charles_river_{Config.scene_id}_image_{index_results_to_plot[image_i]}_epsilon_{epsilon_value}.pkl")
        [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))
        print(pickle_file_path)

        # Plot plot_results
        axarr[image_i, 0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 0].get_yaxis().set_ticks([])
        axarr[image_i, 0].get_xaxis().set_ticks([])

        # Remove axis
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[image_i, 0].spines[axis].set_linewidth(0)

        axarr[image_i, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

        # The label position must be changed accordingly, considering the amount of images plotted
        if ~plot_images_manuscript:
            # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
            axarr[image_i, 0].yaxis.set_label_coords(-0.8, 0.25)
        else:
            # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
            axarr[image_i, 0].yaxis.set_label_coords(-0.7, 0.25)

        axarr[image_i, 1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 1].axis('off')
        axarr[image_i, 2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 2].axis('off')
        axarr[image_i, 3].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]],
                                 cmap)
        axarr[image_i, 3].axis('off')
        axarr[image_i, 4].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 4].axis('off')
        axarr[image_i, 5].imshow(
            predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 5].axis('off')
        axarr[image_i, 6].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        axarr[image_i, 6].axis('off')

    # Set figure labels
    for idx, label in enumerate(plot_legend):
        axarr[0, idx].title.set_fontfamily('Times New Roman')
        axarr[0, idx].title.set_fontsize(12)
        axarr[0, idx].title.set_text(plot_legend[idx])

    # Adjust space between subplots
    plt.subplots_adjust(top=0.437, right=0.776)
    # plt.subplot_tool()
    # plt.show()
else:
    print("No plot_results of this scene appearing in the publication.")

# Save figure as pdf
if Debug.save_figures:
    plt.savefig(os.path.join(Config.path_figures,
                             f'classification_{Config.scenario}_{Config.scene_id}_epsilon_{epsilon_value}_norm_constant_{Config.norm_constant}_selected_dates.pdf'),
                format="pdf", bbox_inches="tight", dpi=200)
