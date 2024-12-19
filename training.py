import os
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from image_reader import ReadSentinel2
from configuration import Config, Debug
from tools.spectral_index import (
    get_num_images_in_folder,
    get_broadband_index,
    get_labels_from_index
)
from tools.cloud_filtering import (
    get_ci_2_labels,
    get_csi_labels,
    get_ndvi_cloud_detector_labels,
    scale_image,
    check_cloud_threshold
)
from figures import get_rgb_image, plot_image


def training_main(image_reader: ReadSentinel2):
    """ Main function of the training stage.

    First, the training images available in the training images folder (defined
    in the configuration file) are read.

    The spectral index defined in the configuration folder is calculated for the
    pixels of the training images.

    The data labels and the Gaussian Mixture Model densities are calculated/generated.
    Finally, the Logistic Regression (LR) model is trained.

    """
    logging.debug("Start Training")

    # Read training images
    # Size training_images is (dim_x*dim_y, num_bands)
    training_images = read_training_images(image_reader)

    # Calculate and add the spectral index for all the images
    # The index will be NDVI, NDWI or MNDWI depending on what has been specified in the configuration file
    index = get_broadband_index(data=training_images, bands=Config.bands_spectral_index[Config.scenario])
    training_images = np.hstack([training_images, index.reshape(-1, 1)])  # one new column with the indices is added
    # to training_images

    # Get labels from the spectral index values
    # The threshold considered to compute the labels is the one specified in the configuration file
    labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]),
                                   threshold=Config.threshold_index_labels[Config.scenario])
    # Discard training image pixels with clouds/cloud shadows by using a mask
    if Config.apply_cloud_mask_training:
        training_images_masked, labels_masked, mask = apply_mask_cloud_det(training_images=training_images, labels=labels)
    else:
        mask = np.zeros(labels.size)
    if Debug.plot_labels_training_images:
        #Config.bands_spectral_index[Config.scenario] = ['3', '8']
        #Config.threshold_index_labels[Config.scenario] = [-0.65 ]
        # Debugging: plot label images (to find the desired threshold)
        plot_labels_training_images(training_images=training_images,
                                    bands_index=Config.bands_spectral_index[Config.scenario],
                                    threshold=Config.threshold_index_labels[Config.scenario], mask=mask)

    # Generate Gaussian Mixtures and train the Logistic Regression (LR) model
    if Config.apply_cloud_mask_training:
        gmm_densities = get_gmm_densities(labels=labels_masked, images=training_images_masked)
        trained_lr_model = get_trained_lr_model(images=training_images_masked, labels=labels_masked)
    else:
        gmm_densities = get_gmm_densities(labels=labels, images=training_images)
        trained_lr_model = get_trained_lr_model(images=training_images, labels=labels)


    logging.debug("Training stage is finished")
    return labels, gmm_densities, trained_lr_model

def read_training_images(image_reader: ReadSentinel2):
    """ Reads available training images.

    Parameters
    ----------
    image_reader : ReadSentinel2
        Sentinel2 image reader object

    Returns
    -------
    training_images : np.ndarray
        all the bands of the available training images

    """
    # Path where the training images are stored
    path_training_images = os.path.join(Config.path_sentinel_images, Config.scenario, 'training')

    # It is assumed that all the band folders have the same number of stored images.
    # Therefore, to count training images we can check the folder associated to any of the bands.
    # For simplicity, we select the first available band.
    path_first_band = os.path.join(path_training_images, f'Sentinel2_B{Config.bands_to_read[0]}')

    # Images with the type and file extension specified in Config are counted:
    # It is necessary to count the amount of training images because there is no a priori information
    # about this value
    num_training_images = get_num_images_in_folder(path_folder=path_first_band,
                                                   image_type=Config.image_types[Config.scenario],
                                                   file_extension=Config.file_extension[Config.scenario])
    logging.debug(f"Number of available images for training = {num_training_images}")

    # Images are read and stored in *images_all_bands*
    size_image = Config.image_dimensions[Config.scenario]['dim_x'] * Config.image_dimensions[Config.scenario]['dim_y']
    # Initialize empty vector with shape
    training_images = np.empty(shape=[size_image * num_training_images, len(Config.bands_to_read)])

    # Loop through images (each image corresponds to one date)
    # *image_idx* goes from 0 to N-1, being N the number of images with type and file name extension
    # specified in the configuration. Images/files with different type/extension are skipped
    for image_idx in range(0, num_training_images):
        # All bands of the image with index *image_idx* are stored in *image_all_bands*
        if Config.scenario == "multiearth":
            image_all_bands, date_string = image_reader.read_image(path=path_training_images, image_idx=image_idx, shift_option="training")
            #image_all_bands, date_string = image_reader.read_image(path=path_training_images, image_idx=image_idx)
        else:
            image_all_bands, date_string = image_reader.read_image(path=path_training_images, image_idx=image_idx)
        # Add all the bands of the image/date that corresponds to this iteration
        training_images[image_idx * size_image: (image_idx + 1) * size_image, :] = image_all_bands

        # Debugging
        _ = check_cloud_threshold(image=image_all_bands, image_date=date_string)
        print('training image has been read')

    return training_images


def get_gmm_densities(images: np.ndarray, labels: np.ndarray):
    """ Gets the value of the Gaussian Mixture Model densities used in the training and evaluation stages.
    - If Config.gmm_dump_pickle = False, the data has already been generated and stored in a pickle file.
    This function therefore loads the available data.
    - If Config.gmm_dump_pickle = True, the data is generated from scratch in this function. After, it is
    stored in a pickle file.

    Parameters
    ----------
    images : np.ndarray
        array containing the available training images
    labels: np.ndarray
        array containing each pixel label

    Returns
    -------
    gmm_densities : list
        returns the trained Gaussian Mixture Model densities in a list

    """
    num_components = Config.gmm_num_components[Config.scenario]
    pickle_file_path = os.path.join(Config.path_trained_models, f'gmm_densities_{Config.scenario}_numcomp_{num_components}.pkl')
    pickle_file_path = os.path.join(Config.path_trained_models,
                                    f'gmm_densities_{Config.scenario}.pkl')

    # If user wants to generate training data from scratch
    if Config.gmm_dump_pickle:

        logging.debug("Trained model is not available --> Generating Gaussian Mixtures")

        # Set empty list where the calculated densities will be stored
        gmm_densities = []

        # Fitting GMMs for the classes
        for class_idx in range(len(Config.classes[Config.scenario])):
            # Get the number of components per gaussian distribution from the configuration file
            num_components = Config.gmm_num_components[Config.scenario][class_idx]

            # The thresholds defined in the configuration file determine which are the positions
            # of the pixels are to be evaluated for each Gaussian Mixture
            # target_positions = get_pos_condition_index(class_idx=class_idx, spectral_index=index)

            # Cut the number of pixels used for training if the code execution is too slow
            # by using the parameter *Config.training_data_crop*
            gmm_densities.append(GaussianMixture(n_components=num_components).fit(
                images[labels == class_idx, :-1][
                0:int(Config.training_data_crop_ratio[Config.scenario] * images.shape[0])]))

        if Debug.plot_gmm_components:
            # Model Selection
            lower_bound_list = dict()
            for class_idx in range(len(Config.classes[Config.scenario])):
                lower_bound_list[class_idx] = []
                for num_components_model_selection in range(1, 10):
                    gmm_loop = GaussianMixture(n_components=num_components_model_selection).fit(
                        images[labels == class_idx, :-1][
                        0:int(Config.training_data_crop_ratio[Config.scenario] * images.shape[0])])
                    lower_bound_list[class_idx].append(gmm_loop.lower_bound_)
                    print(num_components_model_selection)
                plt.figure()
                plt.plot(lower_bound_list[class_idx])

                fig, axs = plt.subplots(len(Config.classes[Config.scenario]), len(Config.bands_to_read), figsize=(15, 7))
                fig.suptitle('Histogram training pixel values (GMM means in red)')
                for class_idx in range(len(Config.classes[Config.scenario])):
                    gmm_means = gmm_densities[class_idx].means_
                    # gmm_cov = gmm_densities[class_idx].covariances_
                    axs[class_idx, 0].set_ylabel(
                        'Class ' + str(class_idx) + ' (' + str(gmm_densities[class_idx].n_components) + ' comp)')
                    for band_idx, band_id in enumerate(Config.bands_to_read):
                        if class_idx == 0:
                            axs[class_idx, band_idx].title.set_text('B' + band_id)
                        image_data_i = images[labels == class_idx, band_idx]
                        axs[class_idx, band_idx].hist(image_data_i, bins=100)
                        for component_i in range(0, gmm_densities[class_idx].n_components):
                            axs[class_idx, band_idx].axvline(gmm_means[component_i, band_idx], color='red', linewidth=1)
                plt.tight_layout()

        # Dump data into pickle
        pickle.dump(gmm_densities, open(pickle_file_path, 'wb'))

    # If user wants to load training data that has already been generated
    else:  # if ~Config.gmm_dump_pickle
        logging.debug("Trained model is already available --> Loading trained Gaussian Mixture Model (GMM)")

        # The GMM densities have already been computed and stored in a pickle file,
        # so it is not needed to calculate them again. They can be directly read.
        gmm_densities = pickle.load(open(pickle_file_path, 'rb'))
    return gmm_densities


def get_trained_lr_model(images: np.ndarray, labels: np.ndarray):
    """ Trains the Logistic Regression (LR) model with the available training images and using the generated
    Gaussian Mixture Model densities.
    - If Config.trained_lr_model_pickle = False, the data has already been generated and stored in a pickle file.
    This function therefore loads the available data.
    - If Config.trained_lr_model_pickle = True, the data is generated from scratch in this function. After, it is
    stored in a pickle file.

    Parameters
    ----------
    images : np.ndarray
        array containing the available training images
    labels : np.ndarray
        array containing each pixel label

    Returns
    -------
    trained_lr_model : LogisticRegression
        trained Logistic Regression (LR) model

    """
    # Get the relative path where the pickle file containing training data is/will be stored
    pickle_file_path = os.path.join(Config.path_trained_models, f'lr_trained_model_{Config.scenario}.pkl')

    # If user wants to generate training data from scratch
    if Config.trained_lr_model_pickle:
        logging.debug("Trained model is not available --> Training Logistic Regression (LR) Model")
        # Train Logistic Regression (LR) model
        # The last column of the images array is not processed because it contains the spectral index values
        trained_lr_model = LogisticRegression().fit(X=images[:, :-1], y=labels)

        # Dump data into pickle
        pickle.dump(trained_lr_model, open(pickle_file_path, 'wb'))

    # If user wants to load training data that has already been generated
    else:  # ~Config.trained_lr_model_pickle
        logging.debug("Trained model is already available --> Loading trained Logistic Regression (LR) Model")
        trained_lr_model = pickle.load(open(pickle_file_path, 'rb'))
    return trained_lr_model


def plot_labels_training_images(training_images: np.ndarray, bands_index: np.ndarray, threshold: float, mask : np.ndarray):
    """ Plot training images and their labels, with pixels discarded by the cloud/cloud shadow mask shown.
    Parameters
    ----------
    training_images : np.ndarray
        array containing the available training images
    bands_index : np.ndarray
        array containing the identifiers of the bands to be read to compute the spectral indices
    threshold : float
        threshold used to calculat the training labels
    mask: np.ndarray
        equals to 1 for the pixels that contain either cloud or cloud shadows

    Returns
    -------
    -
    """
    # Compute spectral index
    index = get_broadband_index(data=training_images, bands=bands_index)
    training_images = np.hstack([training_images[:, :-1], index.reshape(-1, 1)])

    # Get training labels
    labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]), threshold=threshold)

    # Calculate number of training images
    num_pixels_image = Config.image_dimensions[Config.scenario]['dim_x'] * Config.image_dimensions[Config.scenario][
        'dim_y']
    num_training_images = int(labels.size / num_pixels_image)
    #num_training_images = 7

    # Create figure, set titles and labels
    fig, axs = plt.subplots(4, num_training_images, figsize=(13, 7))
    fig.suptitle('Labels training images (values discarded with the cloud/cloud shadow mask)'+'\n1: deforested (yellow), 0: forest (blue)')
    axs[0, 0].set_ylabel('Labels', rotation=0, fontsize=12, fontfamily='Times New Roman')
    axs[0, 0].yaxis.set_label_coords(-.35, .5)
    axs[0, 0].title.set_text('Thresholds: ' + str(
        Config.threshold_index_labels[Config.scenario]) + '\n' + 'Bands to compute index: ' + str(
        Config.bands_spectral_index[Config.scenario]))
    axs[1, 0].title.set_text('Index histograms')
    axs[2, 0].title.set_text('Satellite images')
    axs[3, 0].title.set_text('Cloud+Cloud Shadow Mask')


    histogram_list = []  # store histogram maximum value to adjust y axis after loop

    # Loop training images
    for idx in range(0, num_training_images):
        # Plot Label Images
        pixels_label_image = labels[num_pixels_image * idx:num_pixels_image * (idx + 1)]
        label_image = pixels_label_image.reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                 Config.image_dimensions[Config.scenario]['dim_y'])
        plot_image(axes=axs, image=label_image, indices_axes=[0, idx])

        # Plot index histogram per image
        indices_image = index[num_pixels_image * idx:num_pixels_image * (idx + 1)]
        histogram = axs[1, idx].hist(indices_image, bins=100)
        histogram_list.append(histogram[0])

        # Plot RGB images
        rgb_image = get_rgb_image(
            image_all_bands=training_images[num_pixels_image * idx:num_pixels_image * (idx + 1), 0:-1])
        plot_image(axs, rgb_image, [2, idx])

        # Plot Mask
        mask_image = mask[num_pixels_image * idx:num_pixels_image * (idx + 1)].reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                 Config.image_dimensions[Config.scenario]['dim_y'])
        plot_image(axs, mask_image, [3, idx])

    for idx in range(0, num_training_images):
        axs[1, idx].get_yaxis().set_ticks([0, np.max(histogram_list)])

    plt.tight_layout()
    #plt.savefig('surrogate_labels_charles_river.pdf', bbox_inches='tight', format='pdf')

def apply_mask_cloud_det(training_images: np.ndarray, labels: np.ndarray):
    """ Applies mask for cloud and cloud shadow detection to the input image.
    This function also provides an explanatory plot of the masking process.
    The mask is 1 when cloud and/or cloud shadow are detected.
    ----------
    training_images : np.ndarray
        array containing the available training images
    labels: np.ndarray
        array containing the labels of the available training pixels

    Returns
    -------
    training_images_masked: np.ndarray
        training images after applying the mask (training pixels with clouds/cloud shadows have been discarded)
    labels_masked: np.ndarray
        labels after applying the mask
    """
    # Divide in individual training images
    num_pixels_image = Config.image_dimensions[Config.scenario]['dim_x'] * Config.image_dimensions[Config.scenario][
        'dim_y']
    num_training_images = int(len(training_images) / num_pixels_image)

    if Debug.plot_cloud_detection_training:
        # Create figure
        fig, axs = plt.subplots(5, num_training_images, figsize=(13, 10))

        # Set figure titles
        fig.suptitle('t_2 = ' + str(Config.cloud_filtering['t_2']) + '; t_3 = ' + str(Config.cloud_filtering['t_3']) + '; t_ndvi = ' + str(Config.cloud_filtering['t_ndvi']))
        axs[0, 0].title.set_text('Cloud det. (CI_2)')
        axs[1, 0].title.set_text('Cloud det. (NDVI)')
        axs[2, 0].title.set_text('Satellite images')
        axs[3, 0].title.set_text('Shadow det. (CSI)')
        axs[4, 0].title.set_text('Cloud and Shadow Mask')

    mask = np.zeros(len(training_images))  # init mask (mask = 0 if no cloud/cloud shadow is detected)

    for idx in range(0, num_training_images):
        # Plot ci_2 index (cloud detection)
        labels_ci_2 = get_ci_2_labels(pixels=training_images[num_pixels_image * idx:num_pixels_image * (idx + 1), :])

        # Plot NDVI index (cloud detection)
        labels_ndvi = get_ndvi_cloud_detector_labels(pixels=training_images[num_pixels_image * idx:num_pixels_image * (idx + 1), :])
        labels_ndvi = (labels_ndvi - np.ones(labels_ndvi.size))*-1

        # Plot CSI index (shadow detection)
        labels_csi = get_csi_labels(pixels=training_images[num_pixels_image * idx:num_pixels_image * (idx + 1), :])

        # Update and plot cloud and shadow mask
        mask_idx = np.zeros(num_pixels_image)
        #np.place(mask_idx, np.bitwise_or(labels_ci_2 == 1, labels_csi == 1), 1)  # This mask combines the CI_2 (cloud detection) and CSI (cloud shadow detection) indices
        np.place(mask_idx, np.bitwise_or(labels_ndvi == 1, labels_csi == 1), 1)
        mask[num_pixels_image * idx:num_pixels_image * (idx + 1)] = mask_idx

        if Debug.plot_cloud_detection_training:
            # Plot CI_2 index
            labels_ci_2_image = labels_ci_2.reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                    Config.image_dimensions[Config.scenario]['dim_y'])
            plot_image(axs, labels_ci_2_image, [0, idx])

            # Plot NDVI index
            labels_ndvi_image = labels_ndvi.reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                    Config.image_dimensions[Config.scenario]['dim_y'])
            plot_image(axs, labels_ndvi_image, [1, idx])

            # Plot CSI index
            labels_csi_image = labels_csi.reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                  Config.image_dimensions[Config.scenario]['dim_y'])
            plot_image(axs, labels_csi_image, [3, idx])

            # Plot mask
            mask_image = mask_idx.reshape(
                Config.image_dimensions[Config.scenario]['dim_x'], Config.image_dimensions[Config.scenario]['dim_y'])
            plot_image(axs, mask_image, [4, idx])

            # Plot RGB images
            rgb_image = get_rgb_image(
                image_all_bands=training_images[num_pixels_image * idx:num_pixels_image * (idx + 1), 0: -1])
            rgb_image = 2 * rgb_image  # increase image luminosity
            plot_image(axs, rgb_image, [2, idx])

            plt.tight_layout()

    # Apply the mask to consider only pixels where no clouds/cloud shadows are detected
    # Discard pixels with mask = 1
    indices_to_discard = np.where(mask == 1)
    labels_masked = np.delete(labels, indices_to_discard)
    training_images_masked = np.delete(training_images, indices_to_discard, axis=0)

    # Return training_images after applying masking that discards clouds and cloud shadows
    return training_images_masked, labels_masked, mask