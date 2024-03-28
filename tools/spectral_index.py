import os

import numpy as np

from configuration import Config
from typing import List

'''
# The get_pos_condition_index function is DEPRECATED
def get_pos_condition_index(class_idx: int, spectral_index: np.ndarray):
    """ Returns the positions of the vector *index* that match the thresholds defined
    in the configuration file.

    Parameters
    ----------
    class_idx : int
        index of the evaluated class, which determines which thresholds are to be considered
    spectral_index: np.ndarray
        array with spectral indexes to be evaluated

    Returns
    -------
    positions : np.ndarray
        array with the positions of the vector *index* that match the corresponding thresholds

    """
    # Get list with the corresponding threshold values from the configuration file
    # Being N the number of classes, the class index
    threshold_list = Config.threshold_index_labels[Config.scenario]

    # Create empty array to store the target positions
    positions_th_1 = np.empty(shape=0)
    first_comparative_applied = 0  # the application of the two comparatives is independent

    # Apply the necessary threshold constraints to select the target positions of the *indexes* vector
    if class_idx >= 1:
        # Apply 'greater than' constraint
        positions_th_1 = np.where(spectral_index >= threshold_list[class_idx - 1])
        first_comparative_applied = 1
    if class_idx < len(threshold_list):
        # Apply 'lower than' constraint
        positions = np.where(spectral_index < threshold_list[class_idx])
        # Intersect if 'greater than' limit had been applied
        if first_comparative_applied:
            positions = np.intersect1d(positions_th_1, positions)
    else:
        positions = positions_th_1
    return positions[0]
'''


def get_num_images_in_folder(path_folder: str, image_type: str, file_extension: str):
    """ Returns the number of images with type *image_type* and file extension *file_extension*
    in the folder with path *path_folder*.

    Parameters
    ----------
    path_folder : str
        path of the folder from which images are counted
    image_type : str
        type that images must have to be counted
    file_extension : str
        file extension that the image files must have to be counted

    Returns
    -------
    image_counter : int
        number of images with type *image_type* and file extension *file_extension*
        in the folder with path *path_folder*.

    """
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in os.listdir(path_folder):
        if file_name.endswith(image_type + file_extension):
            file_counter = file_counter + 1
    return file_counter


def get_path_image(path_folder: str, image_type: str, file_extension: str, image_index: int):
    """ Returns the path of the image stored in the folder *path_folder*, with type
    *image_type* and file extension *file_extension*. If sorting by file_name in ascending order,
    and only considering the images of the specified type and file extension, the returned
    path is linked to the image with index *image_index*.

    Parameters
    ----------
    path_folder: str
        path of the folder where the target image is stored
    image_type: str
        type of the target image
    file_extension: str
        extension of the target image file
    image_index: int
        index of the target image within the folder with path *path_folder*

    Returns
    -------
    output_path : str
        path of the target image

    """
    output_path = -1  # if image with specified type, file extension and index is not found, -1 is returned
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in os.listdir(path_folder):
        if file_name.endswith(image_type + file_extension):
            if file_counter == image_index:  # the counter of images is compared to the specified image index
                output_path = os.path.join(path_folder, file_name)
                break  # if the corresponding image is found, the loop is automatically stopped
            file_counter = file_counter + 1
    return output_path


def get_broadband_index(data: np.ndarray, bands: List[str]):
    """ Gets spectral index values for the given array of images. To calculate the
    spectral index, the broadband spectral index expression is used. The bands
    considered for the calculation must be specified in the *bands* list.

    Parameters
    ----------
    data : np.ndarray
        array with images from which the index values are calculated
    bands : list
        list with the spectral bands used to calculate the two-band (broadband) index
        - For NDVI, NIR (8) and Red (4) bands are used
        - For NDWI, Green (3) and NIR (8) bands are used
        - For MNDWI, Green (3) and SWIR (11) bands are used

    Returns
    -------
    index_without_nan : np.ndarray
        array with calculated spectral index values after removing np.nan values

    References
    ----------
    Spectral Indices: https://www.l3harrisgeospatial.com/docs/spectralindices.html
    For normalized ratio-based indices such as NDVI and its derivatives, it is not
    necessary to scale the pixel values.

    """
    # Calculate the positions of the identifiers of band[0] and band[1] in the vector that
    # defines all the bands considered when evaluating the algorithm, *Config.bands_to_read*.
    # For example, if *Config.bands_to_read = ['2', '3', '4', '5', '6', '8A', '8', '11']*
    # and *bands = ['3', '8']* (NDWI calculation), *pos_band_1* will be equal to 1 and
    # *pos_band_2* will be equal to 6.
    pos_band_1 = Config.bands_to_read.index(bands[0])
    pos_band_2 = Config.bands_to_read.index(bands[1])
    index = (data[:, pos_band_1] - data[:, pos_band_2]) / (data[:, pos_band_1] + data[:, pos_band_2])
    # nan values are substituted by the mean of spectral index values
    index_without_nan = np.nan_to_num(index, nan=np.nanmean(index.flatten()))  # TODO: check if this can be skipped
    return index_without_nan


def get_labels_from_index(index: np.ndarray, num_classes: int, threshold: float):
    """ Calculates labels from the spectral index values for this data set.

    Parameters
    ----------
    index : np.ndarray
        array with stored spectral index values for this set of images
    threshold : float
        labeling threshold

    Returns
    -------
    labels : np.ndarray
        array with labels calculated considering the spectral index values

    """
    if len(threshold) + 1 == 2:
        labels = np.transpose(index.copy())
        np.place(labels, index < threshold,
                 0)  # labels under threshold are set to 0
        np.place(labels, index >= threshold,
                 1)  # labels over threshold are set to 1
    elif len(threshold) + 1 == 3:
        labels = np.transpose(index.copy())  # TODO: check if this line can be removed
        np.place(labels, index < threshold[0], 0)
        np.place(labels, (index >= threshold[0]) & (index <= threshold[1]), 1)
        np.place(labels, index >= threshold[1], 2)
    return labels


def get_scaled_index(spectral_index: np.ndarray, num_classes: int):
    """ Scaled the spectral index given by the user.

    Parameters
    ----------
    spectral_index : np.ndarray
        array with stored spectral index values for a set of images
    num_classes : int
        number of classes being evaluated

    Returns
    -------
    scaled_index : np.ndarray
        array with calculated values of scaled spectral index

    """
    # Get mean and standard deviation values for the Scaled Index Model with this configuration
    mean_values, std_values = get_mean_std_scaled_index_model(Config.threshold_index_labels[Config.scenario])
    print(mean_values)
    print(std_values)

    list_pdf_values = []
    for i in range(num_classes):
        list_pdf_values.append(
            np.exp(- ((mean_values[i] - spectral_index) / std_values[i]) ** 2 / 2))
    array_pdf_values = np.array(list_pdf_values)
    sum_pdf_values = np.sum(array_pdf_values, axis=0)
    scaled_index = np.divide(array_pdf_values, sum_pdf_values)
    scaled_index = np.transpose(scaled_index)
    '''
    if num_classes == 3:
        list_pdf_values = []
        for i in range(num_classes):
            list_pdf_values.append(np.exp(- ((Config.scaled_index_model[Config.scenario]['mean_values'][i] - spectral_index) /
                                             Config.scaled_index_model[Config.scenario]['std_values'][i]) ** 2 / 2))
        array_pdf_values = np.array(list_pdf_values)
        sum_pdf_values = np.sum(array_pdf_values, axis=0)
        scaled_index = np.divide(array_pdf_values, sum_pdf_values)
        scaled_index = np.transpose(scaled_index)
    else:
        scaled_index = (spectral_index.reshape(-1, 1) + 1) / 2
        probability_water = 1 - scaled_index
        probability_no_water = scaled_index
        scaled_index = np.append(probability_water, probability_no_water, axis=1)
    '''
    return scaled_index


def get_mean_std_scaled_index_model(thresholds: list):
    """ Given the threshold values, computes the scaled index model mean and standard deviation values.

    Parameters
    ----------
    thresholds : list
        includes the thresholds used for the GMM, which are also used to compute the mean and std values of the scaled index model.

    Returns
    -------
    mean_values : list
        includes the mean values used to build the scaled index model
    std_values : list
         includes the standard deviation values used to build the scaled index model

    """
    num_gaussians = len(thresholds) + 1

    # Init lists with mean and standard deviation values
    mean_values = []
    std_values = []

    # Add -1 and +1 threshold
    thresholds_with_limits = [-1]
    for threshold_i in thresholds:
        thresholds_with_limits.append(threshold_i)
    thresholds_with_limits.append(1)

    # Loop through thresholds
    for gaussian_i in range(1, num_gaussians + 1):
        mean_values.append((thresholds_with_limits[gaussian_i] - thresholds_with_limits[gaussian_i - 1]) / 2 +
                           thresholds_with_limits[gaussian_i - 1])
        std_values.append((thresholds_with_limits[gaussian_i] - thresholds_with_limits[gaussian_i - 1]) / 2)

    return mean_values, std_values
