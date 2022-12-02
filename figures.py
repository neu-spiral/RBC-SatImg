import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from typing import List, Dict
from matplotlib import colors
from configuration import Config, Debug
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
    rgb = np.dstack((x, y, z)) * Config.enhance_rgb[Config.scenario]

    # Reshape to get proper image dimensions
    rgb_image = rgb.reshape(dim_h, dim_v, 3)
    return rgb_image


def plot_results(image_all_bands: np.ndarray, y_pred: Dict[str, np.ndarray], predicted_image: Dict[str, np.ndarray], labels: np.ndarray, time_index: int, date_string : str, image_idx: int):
    """ Plots evaluation plot_results when evaluating the target models on the input image.

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
    date_string : str
        string with current image date (for debugging purposes)
    image_idx : int
        index of image (for debugging purposes)

    Returns
    -------
    None (plots the obtained plot_results)

    """
    # Get RGB Image
    rgb_image = get_rgb_image(image_all_bands=image_all_bands)


    # Define colors to use for the plots
    cmap = colors.ListedColormap(Config.cmap[Config.scenario])

    # Reshape the labels
    labels = labels.reshape(Config.image_dimensions[Config.scenario]['dim_x'], Config.image_dimensions[Config.scenario]['dim_y'])

    # The plotting area depends on which scene has been selected in the configuration file
    # If the scene_id is 0, the whole image wants to be evaluated
    if Config.scenario=="oroville_dam":
        # Create figure
        f, axarr = plt.subplots(1, 11, figsize=(12, 4))
        # We need to plot the baseline_models deep learning model plot_results as well
        if Config.scene_id == 0:
            # Plot plot_results
            axarr[0].imshow(y_pred["Scaled Index"], cmap)
            axarr[1].imshow(y_pred["GMM"], cmap)
            axarr[2].imshow(y_pred["Logistic Regression"], cmap)
            axarr[3].imshow(y_pred["DeepWaterMap"], cmap)
            axarr[4].imshow(y_pred["WatNet"], cmap)
            axarr[5].imshow(predicted_image["Scaled Index"], cmap)
            axarr[6].imshow(predicted_image["GMM"], cmap)
            axarr[7].imshow(predicted_image["Logistic Regression"], cmap)
            axarr[8].imshow(predicted_image["DeepWaterMap"], cmap)
            axarr[9].imshow(predicted_image["WatNet"], cmap)
            axarr[10].imshow(rgb_image)
        else:
            # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
            # in the configuration file.
            x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
            y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']
            # Plot plot_results
            #axarr[0].imshow(labels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[3].imshow(y_pred["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[4].imshow(y_pred["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[5].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[6].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[7].imshow(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[8].imshow(predicted_image["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[9].imshow(predicted_image["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[10].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        # Set figure labels
        axarr[0].title.set_text('SIC')
        axarr[1].title.set_text('GMM')
        axarr[2].title.set_text('LR')
        axarr[3].title.set_text('DWM')
        axarr[4].title.set_text('WN')
        axarr[5].title.set_text('RSIC')
        axarr[6].title.set_text('RGMM')
        axarr[7].title.set_text('RLR')
        axarr[8].title.set_text('RDWM')
        axarr[9].title.set_text('RWN')
        axarr[10].title.set_text('RGB')
        # Remove the axis from the figure
        for axx in range(11):
            axarr[axx].axis('off')

        # Save vector with number of water pixels for sensitivity analysis
        # For the three models (GMM, LR, Scaled Index) and their recursive versions
        # With name according to epsilon value
        # We will store one pickle file per epsilon value
        if Debug.pickle_sensitivity and Config.scene_id==2:
            print("Saving sensitivity plot_results")
            pickle_file_path = os.path.join(Config.path_evaluation_results,
                                            f'sensitivity_analysis_epsilon_{Config.eps}_image_index_{image_idx}.pkl')
            # Get water pixels dictionary
            water_pixels = {"GMM": np.sum(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "LR": np.sum(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "SIC": np.sum(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RGMM": np.sum(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RLR": np.sum(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RSIC": np.sum(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])}
            # Dump data into pickle
            pickle.dump([date_string, water_pixels], open(pickle_file_path, 'wb'))
            print(water_pixels)

    else: # Config.scenario=="charles_river":
        # Create figure
        f, axarr = plt.subplots(1, 8, figsize=(12, 4))
        if Config.scene_id == 0:
            # Plot plot_results
            axarr[0].imshow(y_pred["Scaled Index"], cmap)
            axarr[1].imshow(y_pred["GMM"], cmap)
            axarr[2].imshow(y_pred["Logistic Regression"], cmap)
            axarr[3].imshow(predicted_image["Scaled Index"], cmap)
            axarr[4].imshow(predicted_image["GMM"], cmap)
            axarr[5].imshow(predicted_image["Logistic Regression"], cmap)
            axarr[6].imshow(rgb_image)
        else:
            # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
            # in the configuration file.
            x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
            y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']
            # Plot plot_results
            #axarr[0].imshow(labels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[0].imshow(y_pred["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[1].imshow(y_pred["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[2].imshow(y_pred["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[3].imshow(predicted_image["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[4].imshow(predicted_image["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[5].imshow(predicted_image["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[6].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        # Remove the axis from the figure
        for axx in range(8):
            axarr[axx].axis('off')
        # Set figure labels
        axarr[0].title.set_text('SIC')
        axarr[1].title.set_text('GMM')
        axarr[2].title.set_text('LR')
        axarr[3].title.set_text('RSIC')
        axarr[4].title.set_text('RGMM')
        axarr[5].title.set_text('RLR')
        axarr[6].title.set_text('RGB')

    # Adjust subplots
    f.subplots_adjust(wspace=0.1, hspace=0)

    # Show subplots
    f.suptitle(date_string)
    plt.show()
    plt.close()
