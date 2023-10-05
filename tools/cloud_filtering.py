from configuration import Config, Debug
from figures import get_rgb_image
from typing import List
from tools.spectral_index import get_broadband_index, get_labels_from_index

import matplotlib.pyplot as plt
import numpy as np



def get_ci_2_labels(pixels: np.ndarray):
    """ Returns the ci_2 index labels for cloud detection.

    1 - CI_2 index from:
    Han Zhai, Hongyan Zhang, Liangpei Zhang, Pingxiang Li,
    Cloud/shadow detection based on spectral indices for multi/hyperspectral optical remote sensing imagery,
    ISPRS Journal of Photogrammetry and Remote Sensing, Equations (1-b) and (2)

    Parameters
    ----------
    pixels: np.ndarray
        pixels to be processed for cloud detection (from all the bands)

    Returns
    -------
    labels_ci_2 : np.ndarray
    ci_2 index labels for cloud detection

    """
    index_bands_ci_2 = []
    for band_i in Config.cloud_filtering['ci_2_bands']:
        index_bands_ci_2.append(Config.bands_to_read.index(band_i))
    ci_2 = np.sum(pixels[:, index_bands_ci_2], axis=1) / len(Config.cloud_filtering['ci_2_bands'])
    labels_ci_2 = np.transpose(ci_2.copy())
    T_2 = np.mean(ci_2) + Config.cloud_filtering['t_2'] * (np.max(ci_2) - np.mean(ci_2))
    np.place(labels_ci_2, ci_2 < T_2, 0)  # cloud not detected
    np.place(labels_ci_2, ci_2 >= T_2, 1)  # cloud detected
    return labels_ci_2


def get_csi_labels(pixels: np.ndarray):
    """ Returns the csi index labels for cloud shadow detection.

    1 - CSI index from:
    Han Zhai, Hongyan Zhang, Liangpei Zhang, Pingxiang Li,
    Cloud/shadow detection based on spectral indices for multi/hyperspectral optical remote sensing imagery,
    ISPRS Journal of Photogrammetry and Remote Sensing, Equations (3) and (4)

    Parameters
    ----------
    pixels: np.ndarray
        pixels to be processed for cloud shadow detection (from all the bands)

    Returns
    -------
    labels_csi : np.ndarray
    csi index labels for cloud shadow detection

    """
    index_bands_csi = []
    for band_i in Config.cloud_filtering['csi_bands']:
        index_bands_csi.append(Config.bands_to_read.index(band_i))
    csi = np.sum(pixels[:, index_bands_csi], axis=1) / len(Config.cloud_filtering['csi_bands'])
    labels_csi = np.transpose(csi.copy())
    T_3 = np.min(csi) + Config.cloud_filtering['t_3'] * (np.mean(csi) - np.min(csi))
    np.place(labels_csi, csi <= T_3, 1)  # cloud shadow detected
    np.place(labels_csi, csi > T_3, 0)  # cloud shadow not detected
    return labels_csi


def get_ndvi_cloud_detector_labels(pixels: np.ndarray):
    """ Returns results from an NDVI cloud detector.

    Parameters
    ----------
    pixels: np.ndarray
        pixels to be processed for cloud detection (from all the bands)

    Returns
    -------
    labels : np.ndarray
    ndvi index labels for cloud detection

    """
    index = get_broadband_index(data=pixels, bands=Config.cloud_filtering['ndvi_bands'])
    labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]),
                                   threshold=Config.cloud_filtering['t_ndvi'])
    return labels


def check_cloud_threshold(image: np.ndarray, image_date: str):
    """ Returns True if the image cloud/cloud shadow percentage is under the threshold defined in the configuration file.

    Parameters
    ----------
    image: np.ndarray
    processed image for cloud detection
    image_date: str
    date of processed image

    Returns
    -------
    accepted_image: bool
    True if image cloud/cloud shadow percentage is not above threshold

    """
    # --------------------------------- STEP 1: ACTUAL CLOUD MASK CALCULATION
    # WE CHECK THE PERCENTAGE OF CLOUDS IN THE WHOLE IMAGE
    # Cloud and cloud shadow masks are calculated considering the pixels of the whole image
    Config.cloud_threshold_evaluation = 0.2
    labels_ndvi = get_ndvi_cloud_detector_labels(pixels=image)  # cloud labels
    labels_csi = get_csi_labels(pixels=image)  # cloud shadow labels
    labels_c_i_2 = get_ci_2_labels(pixels=image)
    labels_combined = np.zeros(labels_ndvi.size)
    np.place(labels_combined, np.bitwise_or(labels_c_i_2 == 1, labels_csi == 1),
             1)  # cloud labels + cloud shadow labels
    cloud_percentage = np.sum(labels_combined) / labels_combined.size

    # --------------------------------- STEP 2: PLOTS ONLY FOR DEBUGGING PURPOSES
    # The plot and pixel histograms are compute using the pixels within the coordinates defined in
    # coord_scale_x and coord_scale_y
    #if Debug.plot_cloud_detection_evaluation and cloud_percentage < 0.2:
    if Debug.plot_cloud_detection_evaluation:
        # Get RGB Image
        rgb_image = get_rgb_image(image_all_bands=image)
        # Coordinates to scale images
        coord_scale_x = [70, 150]
        coord_scale_y = [0, 70]
        coord_scale_x = [0, 256]
        coord_scale_y = [0, 256]
        coord_hist_x = coord_scale_x
        coord_hist_y = coord_scale_y
        # Create figure
        fig, axs = plt.subplots(2, 5, figsize=(13, 10))
        dim_x = Config.image_dimensions[Config.scenario]['dim_x']
        dim_y = Config.image_dimensions[Config.scenario]['dim_y']
        axs[0, 0].imshow(8 * rgb_image[coord_scale_x[0]:coord_scale_x[1], coord_scale_y[0]:coord_scale_y[1], :])
        axs[0, 0].title.set_text('Satellite')
        axs[0, 1].imshow(
            labels_csi.reshape(dim_x, dim_y)[coord_scale_x[0]:coord_scale_x[1], coord_scale_y[0]:coord_scale_y[1]]), \
        axs[0, 1].title.set_text('CSI (Cloud shadow detection)')
        axs[0, 3].imshow(
            labels_ndvi.reshape(dim_x, dim_y)[coord_scale_x[0]:coord_scale_x[1], coord_scale_y[0]:coord_scale_y[1]]), \
        axs[0, 3].title.set_text('NDVI (Cloud detection)')
        axs[0, 2].imshow(
            labels_c_i_2.reshape(dim_x, dim_y)[coord_scale_x[0]:coord_scale_x[1], coord_scale_y[0]:coord_scale_y[1]]), \
        axs[0, 2].title.set_text('CI_2 (Cloud detection)')
        image_hist_0 = image[:, 0].reshape(dim_x, dim_y)[coord_hist_x[0]:coord_hist_x[1],
                       coord_hist_y[0]:coord_hist_y[1]].flatten()
        axs[0, 4].hist(image_hist_0, bins=60), axs[0, 4].title.set_text('B' + Config.bands_to_read[0])
        for idx, band_i in enumerate(Config.bands_to_read[1:]):
            image_hist_i = image[:, idx + 1].reshape(dim_x, dim_y)[coord_hist_x[0]:coord_hist_x[1],
                           coord_hist_y[0]:coord_hist_y[1]].flatten()
            axs[1, idx].hist(image_hist_i[:], bins=60), axs[1, idx].title.set_text(
                'B' + Config.bands_to_read[idx + 1])
        fig.suptitle(image_date + ' - Cloud+Cloud Shadow percentage of ' + str(round(cloud_percentage * 100, 2)) + ' %')
        plt.tight_layout()
        # axs[0, 4].xaxis.set_ticks([0.025, 0.04, 0.05, 0.075, 0.1])
        # axs[1, 0].xaxis.set_ticks([0.025, 0.04, 0.05, 0.075, 0.1])
        # axs[1, 1].xaxis.set_ticks([0.025, 0.04, 0.05, 0.075, 0.1])

    # --------------------------------- IMAGE SI ACCEPTED OR REJECTED DEPENDING ON THE COMPUTED CLOUD PERCENTAGE
    if cloud_percentage > Config.cloud_threshold_evaluation:
        accepted_image = False
        print('REJECTED image with date ' + image_date + ' - Cloud+Cloud Shadow percentage of ' + str(
            round(cloud_percentage * 100, 2)) + ' %')
    else:
        accepted_image = True
        print('ACCEPTED image with date ' + image_date + ' - Cloud+Cloud Shadow percentage of ' + str(
            round(cloud_percentage * 100, 2)) + ' %')
    return accepted_image


def scale_image(image: np.ndarray, offset_ref: List[float]):
    """  Returns image after scaling, which uses ref_image as a reference.

    Parameters
    ----------
    image: np.ndarray
    image to be scaled
    offset_ref : List[float]
    pixel mean for clean area in reference image for all bands

    Returns
    -------
    scaled_image: np.ndarray
    image after scaling

    """
    # Coordinates to scale images
    coord_scale_x = Config.coord_scale_x
    coord_scale_y = Config.coord_scale_y

    # Get original image size
    dim_x = Config.image_dimensions[Config.scenario]['dim_x']
    dim_y = Config.image_dimensions[Config.scenario]['dim_y']

    # Store image with shifted/adjusted histogram
    scaled_image = np.ndarray(image.shape)

    hist_offset = []
    # Scale each band
    for idx, band_i in enumerate(Config.bands_to_read):
        pixels_no_clouds = image[:, idx].reshape(dim_x, dim_y)[coord_scale_x[0]:coord_scale_x[1],
                           coord_scale_y[0]:coord_scale_y[1]].flatten()

        # Normalization (method 2 - subtracting the average, suggested by Pau):
        offset_idx = offset_ref[idx] - np.mean(pixels_no_clouds)
        hist_offset.append(offset_idx)
        scaled_image[:, idx] = image[:, idx] + offset_idx

        # Normalization (method 1):
        # array_min, array_max = pixels_no_clouds.min(), pixels_no_clouds.max()
        # scaled_image[:, idx] = (image[:, idx] - array_min) / (array_max - array_min)  # we scale the whole band image
        # but with the values given by the area without clouds
    return scaled_image, hist_offset
