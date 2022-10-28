import pickle
import os

import matplotlib.pyplot as plt

from image_reader import ReadSentinel2
from configuration import Config, Debug
from matplotlib import colors
from figures import get_rgb_image


Config.scenario='charles_river'
Config.scene_id=3

index_results_to_plot = Config.index_plot[Config.scene_id]
n_results_to_plot = len(index_results_to_plot)
path_results = Config.path_evaluation_results

cmap = colors.ListedColormap(Config.cmap[Config.scenario])

# Instance of Image Reader object
image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                             Config.image_dimensions[Config.scenario]['dim_y'])
path_evaluation_images = os.path.join(Config.path_sentinel_images, Config.scenario, 'evaluation')

# Get coordinates of evaluated pixels
x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']

# Plot results for Oroville Dam scenario
if Config.scene_id == 1 or Config.scene_id == 2:
    print(f"Results for Oroville Dam, k = {Config.scene_id}")

    # Create figure
    f, axarr= plt.subplots(n_results_to_plot, 11, figsize=(12, 8))
    #f.suptitle("Land Classification Results")

    for image_i in range(0, n_results_to_plot):

        # Read image again to be able to get RGB image
        image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images, image_idx=index_results_to_plot[image_i])

        # Get RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)

        # Read stored evaluation results to reproduce published figure
        pickle_file_path = os.path.join(path_results, f"oroville_dam_{Config.scene_id}_evaluation_results_image_{index_results_to_plot[image_i]}.pkl")
        [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))

        # Plot results
        axarr[image_i, 0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 0].get_yaxis().set_ticks([])
        axarr[image_i, 0].get_xaxis().set_ticks([])
        # Remove axis
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[image_i, 0].spines[axis].set_linewidth(0)
        axarr[image_i, 0].set_ylabel(date_string)
        axarr[image_i, 0].set_ylabel(date_string)
        axarr[image_i, 1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 1].axis('off')
        axarr[image_i, 2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 2].axis('off')
        axarr[image_i, 3].imshow(y_pred["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 3].axis('off')
        axarr[image_i, 4].imshow(y_pred["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 4].axis('off')
        axarr[image_i, 5].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 5].axis('off')
        axarr[image_i, 6].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 6].axis('off')
        axarr[image_i, 7].imshow(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 7].axis('off')
        axarr[image_i, 8].imshow(predicted_image["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 8].axis('off')
        axarr[image_i, 9].imshow(predicted_image["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap, aspect='auto')
        axarr[image_i, 9].axis('off')
        axarr[image_i, 10].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], aspect='auto')
        axarr[image_i, 10].axis('off')

    # Set figure labels
    axarr[0, 0].title.set_text('SIC')
    axarr[0, 1].title.set_text('GMM')
    axarr[0, 2].title.set_text('LR')
    axarr[0, 3].title.set_text('DWM')
    axarr[0, 4].title.set_text('WN')
    axarr[0, 5].title.set_text('RSIC')
    axarr[0, 6].title.set_text('RGMM')
    axarr[0, 7].title.set_text('RLR')
    axarr[0, 8].title.set_text('RDWM')
    axarr[0, 9].title.set_text('RWN')
    axarr[0, 10].title.set_text('RGB')

# Plot results for Charles River scenario
elif Config.scene_id == 3:
    print("Results for Charles River, Study Area C")

    # Create figure
    f, axarr= plt.subplots(n_results_to_plot, 7, figsize=(12, 8))
    #f.suptitle("Land Classification Results")

    for image_i in range(0, n_results_to_plot):

        # Read image again to be able to get RGB image
        image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images, image_idx=index_results_to_plot[image_i])

        # Get RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)

        # Read stored evaluation results to reproduce published figure
        pickle_file_path = os.path.join(path_results, f"charles_river_{Config.scene_id}_evaluation_results_image_{index_results_to_plot[image_i]}.pkl")
        [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))

        # Plot results
        axarr[image_i, 0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 0].get_yaxis().set_ticks([])
        axarr[image_i, 0].get_xaxis().set_ticks([])

        # Remove axis
        for axis in ['top', 'bottom', 'left', 'right']:
            axarr[image_i, 0].spines[axis].set_linewidth(0)

        axarr[image_i, 0].set_ylabel(date_string)
        axarr[image_i, 0].set_ylabel(date_string)
        axarr[image_i, 1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 1].axis('off')
        axarr[image_i, 2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 2].axis('off')
        axarr[image_i, 3].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 3].axis('off')
        axarr[image_i, 4].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 4].axis('off')
        axarr[image_i, 5].imshow(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[image_i, 5].axis('off')
        axarr[image_i, 6].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        axarr[image_i, 6].axis('off')

    # Set figure labels
    axarr[0, 0].title.set_text('SIC')
    axarr[0, 1].title.set_text('GMM')
    axarr[0, 2].title.set_text('LR')
    axarr[0, 3].title.set_text('RSIC')
    axarr[0, 4].title.set_text('RGMM')
    axarr[0, 5].title.set_text('RLR')
    axarr[0, 6].title.set_text('RGB')

else:
    print("No results of this scene appearing in the publication.")

# Save figure as pdf
if Debug.save_figures:
    plt.savefig(os.path.join(Config.path_figures, f'classificacion_{Config.scenario}_{Config.scene_id}].pdf'), format="pdf", bbox_inches="tight")