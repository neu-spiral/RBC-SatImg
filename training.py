import os
import pickle
import logging

import numpy as np

from typing import List
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from collections import Counter

from image_reader import ReadSentinel2
from configuration import Config, Debug
from tools.spectral_index import get_num_images_in_folder, get_broadband_index, get_labels_from_index


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
    training_images = read_training_images(image_reader)

    # Calculate and add the spectral index for all the images
    index = get_broadband_index(data=training_images, bands=Config.bands_spectral_index[Config.scenario])
    training_images = np.hstack([training_images, index.reshape(-1, 1)])
    # Get labels from the spectral index values
    labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]))

    # Generate Gaussian Mixtures and train the Logistic Regression (LR) model
    gmm_densities = get_gmm_densities(labels=labels, images=training_images)
    trained_lr_model = get_trained_lr_model(images=training_images, labels=labels)

    logging.debug("Training stage is finished")
    return labels, gmm_densities, trained_lr_model

'''
def generate_new_labels(images: np.ndarray, gmm_densities: List[GaussianMixture], classes: List[str]):
    """ Generates labels for the training data of the Logistic Regression Model. The likelihood is obtained 
    for each Gaussian Mixture and then normalization is applied.
    
    Parameters
    ----------
    images : np.ndarray
        array containing the available training images
    gmm_densities: List[GaussianMixture]
        list containing the trained Gaussian Mixture Model densities
    classes : List[str]
        list containing the classes to be evaluated for the current scenario

    Returns
    -------
    labels : np.ndarray
        array containing the generated labels
    
    """
    # Initialize vector of new calculated labels
    labels = np.zeros((images.shape[0], len(classes)))

    # Use of the Gaussian Mixture score_samples function to obtain the likelihood of each Gaussian Mixture
    for enum, label in enumerate(classes):
        labels[:, enum] = np.exp(gmm_densities[enum].score_samples(images))

    # Normalization of the obtained label values
    sum_den = np.sum(labels, axis=1)
    sum_den_nonzero_positions = np.where(sum_den > 0)
    labels[sum_den_nonzero_positions, :] = np.divide(labels[sum_den_nonzero_positions, :],
                                                     sum_den.reshape(sum_den.shape[0], 1)[sum_den_nonzero_positions, :])
    return labels.argmax(axis=1)
'''

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
                                                   file_extension='.tif')
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
        image_all_bands, _ = image_reader.read_image(path=path_training_images, image_idx=image_idx)

        # Add all the bands of the image/date that corresponds to this iteration
        training_images[image_idx * size_image: (image_idx + 1) * size_image, :] = image_all_bands

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
    pickle_file_path = os.path.join(Config.path_trained_models, f'gmm_densities_{Config.scenario}.pkl')

    # If user wants to generate training data from scratch
    if Config.gmm_dump_pickle:

        logging.debug("Trained model is not available --> Generating Gaussian Mixtures")

        # Set empty list where the calculated densities will be stored
        gmm_densities = []

        # Fitting GMMs for the classes
        for class_idx in range(len(Config.classes[Config.scenario])):
            # Get the number of components per gaussian distribution from the configuration file
            num_components = Config.gm_model_selection[Config.scenario]['num_components'][class_idx]

            # The thresholds defined in the configuration file determine which are the positions
            # of the pixels are to be evaluated for each Gaussian Mixture
            # target_positions = get_pos_condition_index(class_idx=class_idx, spectral_index=index)

            # Cut the number of pixels used for training if the code execution is too slow
            # by using the parameter *Config.training_data_crop*
            gmm_densities.append(GaussianMixture(n_components=num_components).fit(
                images[labels==class_idx, :-1][0:int(Config.training_data_crop_ratio[Config.scenario] * images.shape[0])]))

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
        """
        new_labels = generate_new_labels(images=np.array(images)[:, :-1], gmm_densities=gmm_densities,
                                         classes=Config.classes[
                                             Config.scenario])  # only used for training the LR
        """
        trained_lr_model = LogisticRegression().fit(X=images[:, :-1], y=labels)

        # Dump data into pickle
        pickle.dump(trained_lr_model, open(pickle_file_path, 'wb'))

    # If user wants to load training data that has already been generated
    else:  # ~Config.trained_lr_model_pickle
        logging.debug("Trained model is already available --> Loading trained Logistic Regression (LR) Model")
        trained_lr_model = pickle.load(open(pickle_file_path, 'rb'))
    return trained_lr_model
