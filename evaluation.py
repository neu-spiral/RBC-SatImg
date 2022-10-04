import logging
import os

import numpy as np

from image_reader import ReadSentinel2
from configuration import Config
from bayesian_recursive import RBC, get_rbc_objects
from figures import plot_results
from typing import List
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from typing import Dict

from tools.spectral_index import get_broadband_index, get_labels_from_index
from tools.path_operations import get_num_images_in_folder


def evaluate_models(image_all_bands: np.ndarray, rbc_objects: Dict[str, RBC], time_index: int):
    """ Returns the prior/likelihood and posterior probabilities for each evaluated model.

    Parameters
    ----------
    image_all_bands : np.ndarray
        array containing the bands of the image under evaluation
    rbc_objects : Dict[str, RBC]
        dictionary containing one RBC object per model to evaluate
    time_index : int
        index of the current time instant of the Bayesian process

    Returns
    -------
    y_pred : Dict[str, np.ndarray]
        dictionary containing the prior probabilities or likelihood for each model
    predicted_image : Dict[str, np.ndarray]
        dictionary containing the posterior probabilities for each model

    """
    # Initialize empty dictionaries with the prior and posterior probabilities
    y_pred = {}
    predicted_image = {}

    # Reshape the image posterior probabilities
    # Reshape the labels (*y_pred*) to compare with the image posterior probabilities
    dim_h = Config.image_dimensions[Config.scenario]['dim_x']
    dim_v = Config.image_dimensions[Config.scenario]['dim_y']

    # GMM Model
    y_pred["GMM"], predicted_image["GMM"] = rbc_objects["GMM"].update_labels(image_all_bands=image_all_bands,
                                                                             time_index=time_index)
    predicted_image["GMM"] = predicted_image["GMM"].reshape(dim_h, dim_v)
    y_pred["GMM"] = y_pred["GMM"].reshape(dim_h, dim_v)

    # Scaled Index Model
    y_pred["Scaled Index"], predicted_image["Scaled Index"] = rbc_objects["Scaled Index"].update_labels(
        image_all_bands=image_all_bands, time_index=time_index)
    predicted_image["Scaled Index"] = predicted_image["Scaled Index"].reshape(dim_h, dim_v)
    y_pred["Scaled Index"] = y_pred["Scaled Index"].reshape(dim_h, dim_v)

    # Logistic Regression Model
    y_pred["Logistic Regression"], predicted_image["Logistic Regression"] = rbc_objects[
        "Logistic Regression"].update_labels(
        image_all_bands=image_all_bands, time_index=time_index)
    predicted_image["Logistic Regression"] = predicted_image["Logistic Regression"].reshape(dim_h, dim_v)
    y_pred["Logistic Regression"] = y_pred["Logistic Regression"].reshape(dim_h, dim_v)

    return y_pred, predicted_image


def evaluation_main(gmm_densities: List[GaussianMixture], trained_lr_model: LogisticRegression,
                    image_reader: ReadSentinel2):
    """ Main function of the evaluation stage.

    First, one RBC object per model to evaluate is created.

    Next, the available images for evaluation are counted so that a loop through the evaluation
    images can be started.

    Finally, the models under study are evaluated on each image.

    Parameters
    ----------
    gmm_densities: List[GaussianMixture]
        list containing the trained Gaussian Mixture Model densities
    trained_lr_model : LogisticRegression
        logistic regression (LR) model after training in the training stage
    image_reader : ReadSentinel2
        instance of a ReadSentinel2 object

    Returns
    -------
    None (plots results for the time indexes specified in the configuration file)

    """

    # Initialize one RBC object for each model to be compared
    rbc_objects = get_rbc_objects(gmm_densities=gmm_densities, trained_lr_model=trained_lr_model)

    # Read the images in the evaluation folder
    # Path where the training images are stored
    logging.debug("Start Evaluation")
    path_evaluation_images = os.path.join(Config.path_sentinel_images, Config.scenario, 'evaluation')

    # It is assumed that all the band folders have the same number of stored images.
    # Therefore, to count training images we can check the folder associated to any of the bands.
    # For simplicity, we select the first available band.
    path_first_band = os.path.join(path_evaluation_images, f'Sentinel2_B{Config.bands_to_read[0]}')

    # Calculate how many images to be read for evaluation purposes
    num_evaluation_images = get_num_images_in_folder(path_folder=path_first_band,
                                                     image_type=Config.image_types[Config.scenario],
                                                     file_extension='.tif')
    logging.debug(f"Number of available images for evaluation = {num_evaluation_images}")

    # Initialize the time instant parameter *time_index*
    time_index = 0

    # The classifier is applied to all the bands of all the images to be read
    # Each image is linked to one specific date
    for image_idx in range(0, num_evaluation_images):
        # All bands of the image with index *image_idx* are stored in *image_all_bands*
        image_all_bands = image_reader.read_image(path=path_evaluation_images, image_idx=image_idx)

        # Calculate and add the spectral index for all bands
        index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])
        image_all_bands = np.hstack([image_all_bands, index.reshape(-1, 1)])

        # Get labels from the spectral index values
        labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]))

        # Evaluate the 3 models for one date
        y_pred, predicted_image = evaluate_models(image_all_bands=image_all_bands, rbc_objects=rbc_objects,
                                                  time_index=time_index)

        # Plot Results at each Image
        # For each Image, we have read as many bands as configured in *Config.bands_to_read*
        # Each Image is linked to one specific Date
        plot_results(y_pred=y_pred, predicted_image=predicted_image, labels=labels, time_index=time_index,
                     image_all_bands=image_all_bands)

        # Update the *time_index* value
        time_index = time_index + 1
