import numpy as np

from matplotlib import colors

from configuration import Config, Visual
from tools.operations import normalize


def plot_image(axes: np.ndarray, image: np.ndarray, indices_axes: list):
    """ Provides a plot of the input image within the specified axes.

    """
    cmap = colors.ListedColormap(Visual.cmap[Config.scenario])
    axes[indices_axes[0], indices_axes[1]].imshow(image, vmin=0, cmap=cmap)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes[indices_axes[0], indices_axes[1]].spines[axis].set_linewidth(0)
    axes[indices_axes[0], indices_axes[1]].get_yaxis().set_ticks([])
    axes[indices_axes[0], indices_axes[1]].get_xaxis().set_ticks([])


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
    # Try without normalization:
    x = image_all_bands[:, band_r_pos].reshape(dim_h, dim_v)
    y = image_all_bands[:, band_g_pos].reshape(dim_h, dim_v)
    z = image_all_bands[:, band_b_pos].reshape(dim_h, dim_v)

    # Stack the three bands
    rgb = np.dstack((x, y, z))

    # Reshape to get proper image dimensions
    rgb_image = rgb.reshape(dim_h, dim_v, 3)
    return rgb_image


# DEPRECATED FUNCTIONS:
'''
def plot_results(image_all_bands: np.ndarray, likelihood: Dict[str, np.ndarray], posterior: Dict[str, np.ndarray], labels: np.ndarray, time_index: int, date_string : str, image_idx: int):
    """ Plots evaluation plot_figures when evaluating the target models on the input image.

    Parameters
    ----------
    image_all_bands : np.ndarray
        pixel values for all the bands of the target image
    likelihood : Dict[str, np.ndarray]
        dictionary containing the prior probabilities or likelihood for each model
    posterior : Dict[str, np.ndarray]
        dictionary containing the posterior probabilities for each model
    labels : np.ndarray
        array containing the predicted labels
    time_index : int
        bayesian recursion time index
    date_string : str
        string with current image date (for multiearth purposes)
    image_idx : int
        index of image (for multiearth purposes)

    Returns
    -------
    None (plots the obtained plot_figures)

    """
    # Get RGB Image
    rgb_image = get_rgb_image(image_all_bands=image_all_bands)


    # Define colors to use for the plots
    cmap = colors.ListedColormap(Visual.cmap[Config.scenario])

    # Reshape the labels
    labels = labels.reshape(Config.image_dimensions[Config.scenario]['dim_x'], Config.image_dimensions[Config.scenario]['dim_y'])

    # The plotting area depends on which scene has been selected in the configuration file
    # If the scene_id is 0, the whole image wants to be evaluated
    if Config.scenario=="oroville_dam":
        # Create figure
        f, axarr = plt.subplots(1, 11, figsize=(12, 4))
        # We need to plot the benchmark_models deep learning model plot_figures as well
        if Config.test_site == 0:
            # Plot plot_figures
            axarr[0].imshow(likelihood["Scaled Index"], cmap)
            axarr[1].imshow(likelihood["GMM"], cmap)
            axarr[2].imshow(likelihood["Logistic Regression"], cmap)
            axarr[3].imshow(likelihood["DeepWaterMap"], cmap)
            axarr[4].imshow(likelihood["WatNet"], cmap)
            axarr[5].imshow(posterior["Scaled Index"], cmap)
            axarr[6].imshow(posterior["GMM"], cmap)
            axarr[7].imshow(posterior["Logistic Regression"], cmap)
            axarr[8].imshow(posterior["DeepWaterMap"], cmap)
            axarr[9].imshow(posterior["WatNet"], cmap)
            axarr[10].imshow(rgb_image)
        else:
            # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
            # in the configuration file.
            x_coords = Config.pixel_coords_to_evaluate[Config.test_site]['x_coords']
            y_coords = Config.pixel_coords_to_evaluate[Config.test_site]['y_coords']
            # Plot plot_figures
            #axarr[0].imshow(labels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[0].imshow(likelihood["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[1].imshow(likelihood["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[2].imshow(likelihood["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[3].imshow(likelihood["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[4].imshow(likelihood["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[5].imshow(posterior["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[6].imshow(posterior["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[7].imshow(posterior["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[8].imshow(posterior["DeepWaterMap"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[9].imshow(posterior["WatNet"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[10].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        # Set figure labels
        axarr[0].title.set_text("Scaled Index")
        axarr[1].title.set_text('GMM')
        axarr[2].title.set_text("Logistic Regression")
        axarr[3].title.set_text('DeepWaterMap')
        axarr[4].title.set_text('WatNet')
        axarr[5].title.set_text('RSIC')
        axarr[6].title.set_text('RGMM')
        axarr[7].title.set_text('RLR')
        axarr[8].title.set_text('RDeepWaterMap')
        axarr[9].title.set_text('RWatNet')
        axarr[10].title.set_text('RGB')
        # Remove the axis from the figure
        for axx in range(11):
            axarr[axx].axis('off')

        # Save vector with number of water pixels for sensitivity analysis
        # For the three models (GMM, LR, Scaled Index) and their recursive versions
        # With name according to epsilon value
        # We will store one pickle file per epsilon value
        if Debug.store_pickle_sensitivity_analysis_water_mapping and Config.test_site==2:
            print("Saving sensitivity plot_figures")
            pickle_file_path = os.path.join(Config.path_evaluation_results,
                                            f'sensitivity_analysis_epsilon_{Config.eps}_image_index_{image_idx}.pkl')
            # Get water pixels dictionary
            water_pixels = {"GMM": np.sum(likelihood["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "Logistic Regression": np.sum(likelihood["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "Scaled Index": np.sum(likelihood["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RGMM": np.sum(posterior["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RLR": np.sum(posterior["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]), "RSIC": np.sum(posterior["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])}
            # Dump data into pickle
            pickle.dump([date_string, water_pixels], open(pickle_file_path, 'wb'))
            print(water_pixels)

    else: # Config.scenario=="charles_river":
        # Create figure
        f, axarr = plt.subplots(1, 8, figsize=(12, 4))
        if Config.test_site == 0:
            # Plot plot_figures
            axarr[0].imshow(likelihood["Scaled Index"], cmap)
            axarr[1].imshow(likelihood["GMM"], cmap)
            axarr[2].imshow(likelihood["Logistic Regression"], cmap)
            axarr[3].imshow(posterior["Scaled Index"], cmap)
            axarr[4].imshow(posterior["GMM"], cmap)
            axarr[5].imshow(posterior["Logistic Regression"], cmap)
            axarr[6].imshow(rgb_image)
        else:
            # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
            # in the configuration file.
            x_coords = Config.pixel_coords_to_evaluate[Config.test_site]['x_coords']
            y_coords = Config.pixel_coords_to_evaluate[Config.test_site]['y_coords']
            # Plot plot_figures
            #axarr[0].imshow(labels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[0].imshow(likelihood["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[1].imshow(likelihood["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[2].imshow(likelihood["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[3].imshow(posterior["Scaled Index"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[4].imshow(posterior["GMM"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[5].imshow(posterior["Logistic Regression"][x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap)
            axarr[6].imshow(rgb_image[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]])
        # Remove the axis from the figure
        for axx in range(8):
            axarr[axx].axis('off')
        # Set figure labels
        axarr[0].title.set_text("Scaled Index")
        axarr[1].title.set_text('GMM')
        axarr[2].title.set_text("Logistic Regression")
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
    

def get_green_image(image_all_bands: np.ndarray):
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
    # Try without normalization:
    x = image_all_bands[:, band_r_pos].reshape(dim_h, dim_v)
    y = image_all_bands[:, band_g_pos].reshape(dim_h, dim_v)
    z = image_all_bands[:, band_b_pos].reshape(dim_h, dim_v)

    # Stack the three bands
    if Config.scenario == 'multiearth':
        rgb = np.dstack((x, y, z))
    else:
        rgb = np.dstack((x, y, z)) * Visual.scaling_rgb[Config.test_site]

    # Reshape to get proper image dimensions
    rgb_image = rgb.reshape(dim_h, dim_v, 3)
    max_channel_2 = np.max(rgb_image, axis=2)
    green_image = np.where((rgb_image[:, :, 1] == max_channel_2), 1, 0)
    return green_image
'''
