import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict
from matplotlib import colors
from configuration import Config
from tools.operations import normalize
from collections import Counter


def get_rgb_image(image_all_bands: np.ndarray):
    """ Returns the RGB image of the input image.

    Parameters
    ----------
    image_all_bands : np.ndarray
        array containing the bands of the image

    Returns
    -------
    rgb_image : np.ndarray
        computed rgb image

    """
    # Get image dimensions
    dim_h = Config.image_dimensions[Config.scenario]['dim_x']
    dim_v = Config.image_dimensions[Config.scenario]['dim_y']

    # Get position of RGB bands within the *Config.bands_to_read* list
    band_r_pos = Config.bands_to_read.index('4')  # Red (band 4)
    band_g_pos = Config.bands_to_read.index('3')  # Green (band 3)
    band_b_pos = Config.bands_to_read.index('2')  # Blue (band 2)

    # Get pixel values for all RGB bands
    x = normalize(image_all_bands[:, band_r_pos]).reshape(dim_h, dim_v)
    y = normalize(image_all_bands[:, band_g_pos]).reshape(dim_h, dim_v)
    z = normalize(image_all_bands[:, band_b_pos]).reshape(dim_h, dim_v)

    # Stack the three bands
    rgb = np.dstack((x, y, z)) * 4

    # Reshape to get proper image dimensions
    rgb_image = rgb.reshape(dim_h, dim_v, 3)
    return rgb_image


def plot_results(image_all_bands: np.ndarray, y_pred: Dict[str, np.ndarray], predicted_image: Dict[str, np.ndarray], labels: np.ndarray, time_index: int, date_string : str):
    """ Plots evaluation results when evaluating the target models on the input image.

    Parameters
    ----------
    image_all_bands : np.ndarray
        pixel values for all the bands of the target image
    y_pred : Dict[str, np.ndarray]
        dictionary containing the prior probabilities or likelihood for each model
    predicted_image : Dict[str, np.ndarray]
        dictionary containing the posterior probabilities for each model
    labels : np.ndarray
        array containing the predicted labels
    time_index : int
        bayesian recursion time index

    Returns
    -------
    None (plots the obtained results)

    """
    # Get RGB Image
    rgb_image = get_rgb_image(image_all_bands=image_all_bands)

    # Create figure
    f, axarr = plt.subplots(1, 8, figsize=(12,4))

    # Define colors to use for the plots
    cmap = colors.ListedColormap(Config.cmap[Config.scenario])

    # Reshape the labels
    labels = labels.reshape(Config.image_dimensions[Config.scenario]['dim_x'], Config.image_dimensions[Config.scenario]['dim_y'])

    # The plotting area depends on which scene has been selected in the configuration file
    # If the scene_id is 0, the whole image wants to be evaluated
    if Config.scene_id == 0:
        # Plot results
        axarr[0].imshow(labels, cmap)
        axarr[1].imshow(y_pred["GMM"], cmap)
        axarr[2].imshow(y_pred["Scaled Index"], cmap)
        axarr[3].imshow(y_pred["Logistic Regression"], cmap)
        axarr[4].imshow(predicted_image["GMM"], cmap)
        axarr[5].imshow(predicted_image["Scaled Index"], cmap)
        axarr[6].imshow(predicted_image["Logistic Regression"], cmap)
        axarr[7].imshow(rgb_image*Config.enhance[Config.scenario]   , alpha=1)
    else:
        # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
        # in the configuration file.
        x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
        y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']
        # Plot results
        axarr[0].imshow(labels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[2].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[3].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[4].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[5].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[6].imshow(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
        axarr[7].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])

    # Remove the axis from the figure
    for axx in range(8):
        axarr[axx].axis('off')

    # Set figure labels
    if time_index == 0:
        axarr[0].title.set_text('Index')
        axarr[1].title.set_text('GMM')
        axarr[2].title.set_text('Scaled Index')
        axarr[3].title.set_text('LR')
        axarr[4].title.set_text('GMM RBC')
        axarr[5].title.set_text('Scaled Index RBC')
        axarr[6].title.set_text('LR RBC')
        axarr[7].title.set_text('RGB')

    # Adjust subplots
    f.subplots_adjust(wspace=0.1, hspace=0)

    # Show subplots
    f.suptitle(date_string)
    plt.show()
    plt.close()
