import logging

import numpy as np

from typing import List, Dict
from sklearn.mixture import GaussianMixture
from configuration import Config
from benchmark.benchmark import main_deepwater
from benchmark.watnet.watnet_infer import watnet_infer_main
from tools.spectral_index import get_broadband_index, get_scaled_index
from tools.operations import get_index_pixels_of_interest
from sklearn.linear_model import LogisticRegression


class RBC:
    """ Class containing the core of the recursive Bayesian classification algorithm. RBC stands for
    Recursive Bayesian Classifier.

    Class Attributes
    ----------------
    transition_matrix : np.ndarray
        transition matrix, hardcoded by the user in the configuration file
    gmm_densities: List[GaussianMixture]
        list containing the trained Gaussian Mixture Model densities
    classes : List[str]
        list containing the classes to be evaluated for the current scenario
    classes_prior : list
        list containing the prior probability for each evaluated class
    model : str
        model string identifier

    Class Methods
    -------------
    set_lr_trained_model(cls, trained_model)
    get_rbc_objects(self, gmm_densities, trained_lr_model)
    update_labels(self, image_all_bands, time_index)
    calculate_prediction(self, image_all_bands, time_index)
    calculate_posterior(self, bands, y_pred, model, pixel_position)

    """

    def __init__(self, transition_matrix: np.ndarray, gmm_densities: List[GaussianMixture], classes: list,
                 classes_prior: list, model: str):
        """ Initializes instance of RBC object with the corresponding class attributes.

        """
        self.transition_matrix = transition_matrix
        self.classes = classes
        self.classes_prior = classes_prior
        self.num_classes = len(classes)
        self.model = model
        self.densities = gmm_densities
        print(f"model name {self.model} and transition probability matrix is {self.transition_matrix}")

    @classmethod
    def set_lr_trained_model(cls, trained_model: LogisticRegression):
        """ Sets the corresponding value of the trained Logistic Regression (LR) model.

        """
        cls.trained_model = trained_model

    def update_labels(self, image_all_bands: np.ndarray, time_index: int = 0):
        """ Returns the prior/likelihood and posterior probabilities for each evaluated model.

        Parameters
        ----------
        image_all_bands : np.ndarray
            pixel values for all the bands of the target image
            This parameter must have size (dim_x*dim_y, n_bands):
                - dim_x = Config.image_dimensions[Config.scenario]['dim_x']
                - dim_y = Config.image_dimensions[Config.scenario]['dim_y']
                - n_bands = len(Config.bands_to_read))
        time_index : int
            bayesian recursion time index

        Returns
        -------
        y_pred : Dict[str, np.ndarray]
            dictionary containing the prior probabilities or likelihood for each model
        predicted_image : Dict[str, np.ndarray]
            dictionary containing the posterior probabilities for each model
        
        """
        # Calculate the index of pixels of interest
        self.index_pixels_of_interest = get_index_pixels_of_interest(image_all_bands=image_all_bands,
                                                                     scene_id=Config.scene_id)

        # Calculation of the total number of pixels in the processed images
        self.total_num_pixels = Config.image_dimensions[Config.scenario]['dim_x'] * \
                                Config.image_dimensions[Config.scenario]['dim_y']

        # Calculate the prior probability (time_index = 0)
        # or the likelihood or base model posterior (time_index > 0)
        # The manner to calculate this depends on the model under evaluation (if-clause)
        y_pred = self.calculate_prediction(image_all_bands=image_all_bands)

        #  At time instant 0, the prior probability is equal to:
        #   - the model prediction, in the case of having a pretrained model
        #   - the softmax probability, otherwise
        # *y_pred* corresponds to the likelihood or the base model posterior
        if time_index == 0:
            self.prior_probability = y_pred

        # Calculate the posterior probability
        # Initialize array where the posterior probabilities will be stored
        # Later, this array is updated analysing only the pixels in the subscene
        self.posterior_probability = np.zeros(shape=[self.total_num_pixels, self.num_classes])

        # TODO: This line was added to avoid a dimensional error, but it should be checked whether it could be removed
        if np.ndim(y_pred) == 3:
            y_pred = y_pred.reshape(self.total_num_pixels, self.num_classes)

        # Image with the posterior probabilities for each pixel is calculated
        # Only the pixels from the chosen scene are considered
        # The calculate_posterior function is applied to all the rows
        # Each row corresponds to all the bands values for a specific pixel
        """
        del_l = np.apply_along_axis(self.calculate_posterior, 1, image_all_bands[self.index_pixels_of_interest,], y_pred, self.model)
        del_l = del_l.reshape((del_l.shape[0], self.num_classes))
        self.posterior_probability[self.index_pixels_of_interest] = del_l  # we just save the subscene pixels

        """
        del_l = np.ndarray(shape=[self.index_pixels_of_interest.shape[0], self.num_classes])
        for pixel_i in range(self.index_pixels_of_interest.shape[0]):
            del_l[pixel_i, :] = self.calculate_posterior(bands=image_all_bands[self.index_pixels_of_interest[pixel_i],],
                                                         y_pred=y_pred,
                                                         pixel_position=self.index_pixels_of_interest[pixel_i])
        self.posterior_probability[self.index_pixels_of_interest] = del_l  # we just save the subscene pixels

        # Calculate the class predictions
        predict_image = self.posterior_probability.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)  # position with maximum values

        return y_pred, predict_image

    def calculate_prediction(self, image_all_bands: np.ndarray):
        """ Returns the prior/likelihood and posterior probabilities for each evaluated model.

        Parameters
        ----------
        image_all_bands : np.ndarray
            pixel values for all the bands of the target image

        Returns
        -------
        y_pred : np.ndarray
            prior probabilities or likelihood for the currently evaluated model

        """
        # Scaled Index Model
        if self.model == "Scaled Index":

            # Calculate spectral index, which has range [-1, 1]
            spectral_index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])

            # Scale the spectral index to range [0, 1] so that it can be seen as a probability
            scaled_index = get_scaled_index(spectral_index=spectral_index, num_classes=self.num_classes)
            y_pred = scaled_index

        # DeepWaterNet Model, used in this project for benchmarking
        elif self.model == "DeepWaterMap":  # first version

            # Select only the bands used in the case of this algorithm
            image_deepwaternet_bands = image_all_bands[:, Config.bands_deepwaternet].reshape(
                Config.image_dimensions[Config.scenario]['dim_x'],
                Config.image_dimensions[Config.scenario]['dim_y'], len(Config.bands_watnet))

            # Values are clipped between 0 and 1
            image_deepwaternet_bands = np.float32(
                np.clip(image_deepwaternet_bands * Config.scaling_factor_watnet, a_min=0, a_max=1))

            # The model published by the authors is called within the main_deepwater function
            water_map = main_deepwater(checkpoint_path=Config.path_checkpoint_deepwatermap, image=image_deepwaternet_bands)
            water_map = water_map.reshape(-1, 1)
            y_pred = np.concatenate((water_map, 1-water_map), axis=1)

        # WatNet Model, used in this project for benchmarking
        # The WatNet model is an improved version of the DeepWaterNet model provided by their authors
        elif self.model == "WatNet":

            # Select only the bands used in the case of this algorithm
            image_watnet_bands = image_all_bands[:, Config.bands_watnet].reshape(
                Config.image_dimensions[Config.scenario]['dim_x'],
                Config.image_dimensions[Config.scenario]['dim_y'], len(Config.bands_watnet))

            # Values are clipped between 0 and 1
            image_watnet_bands = np.float32(
                np.clip(image_watnet_bands * Config.scaling_factor_watnet, a_min=0, a_max=1))

            # The model published by the authors is called within the watnet_infer function
            water_map = watnet_infer_main(rsimg=image_watnet_bands,
                                     path_model=Config.path_watnet_pretrained_model).reshape(-1, 1)
            y_pred = np.concatenate((water_map, 1-water_map), axis=1)

        # Logistic Regression Model
        elif self.model == "Logistic Regression":

            # The last column is not considered because it contains the spectral index values, which
            # are not required for the probability prediction
            y_pred = self.trained_model.predict_proba(image_all_bands[:, :-1])

        # GMMs Model
        elif self.model == "GMM":
            y_pred = np.zeros(shape=[self.total_num_pixels, self.num_classes])
            # Calculate the likelihood for each Gaussian Mixture
            # the score_samples function takes into account all the rows
            for label_index, label in enumerate(self.classes):
                # Read *image_all_bands[:, :-1]*, not considering the last column because it correspond to the
                # spectral index values
                y_pred[:, label_index] = np.exp(self.densities[label_index].score_samples(image_all_bands[:, :-1]))
            sum_den = np.sum(y_pred, axis=1)
            y_pred = np.divide(y_pred, sum_den.reshape(self.total_num_pixels, 1))
            y_pred = y_pred.astype('float32')
        return y_pred

    def calculate_posterior(self, bands: np.ndarray, y_pred: np.ndarray, pixel_position: int):
        """ Returns the posterior probabilities for each evaluated model. This function is called row by
        row with the np.apply_along_axis function.

        Parameters
        ----------
        bands: np.ndarray
            band values for the evaluated pixel
        y_pred : np.ndarray
            prior probabilities/likelihood for each class
        pixel_position: int
            position of the pixel being evaluated (as the function is called row by row)

        Returns
        -------
        posterior : np.ndarray
            array containing the posterior probability for each class

        """
        # Posterior probability is calculated here. eq1.
        # self.classes_prior[ct_p] = marginal probability
        post_return = np.zeros(shape=[self.num_classes, self.num_classes])
        if self.model != "GMM":  # TODO: Change this
            lik_lb = y_pred[pixel_position]
            #             for each class posterior probability p(water) and p(zt/non water)
            for ct in range(self.num_classes):
                # for each class E(summation outer)
                for ct_1 in range(self.num_classes):
                    b = 0
                    #                     for each class inner summation
                    for ct_p in range(self.num_classes):
                        b += (lik_lb[ct_p] / self.classes_prior[ct_p]) * self.transition_matrix[ct_p, ct_1]
                    a = self.transition_matrix[ct, ct_1] * self.prior_probability[pixel_position, ct_1]
                    post_return[ct, ct_1] = a / b
                post_return[ct, :] = post_return[ct, :] * (lik_lb[ct] / self.classes_prior[ct])
            p = np.sum(post_return, axis=1).reshape(-1, )
            self.prior_probability[pixel_position] = (p / np.sum(p)).reshape(self.prior_probability[pixel_position].shape)
        else:
            lik_lb = y_pred[pixel_position] / np.sum(y_pred[pixel_position])
            for ct in range(self.num_classes):  # label
                for ct_1 in range(self.num_classes):
                    b = 0
                    for ct_p in range(self.num_classes):
                        b += (lik_lb[ct_p]) * self.transition_matrix[ct_p, ct_1]
                    a = self.transition_matrix[ct, ct_1] * self.prior_probability[pixel_position, ct_1]
                    post_return[ct, ct_1] = a / b
                post_return[ct, :] = post_return[ct, :] * (lik_lb[ct])
            p = np.sum(post_return, axis=1)
            self.prior_probability[pixel_position] = (p / np.sum(p))
        posterior = p / np.sum(p)
        return posterior


def get_rbc_objects(gmm_densities: List[GaussianMixture], trained_lr_model: LogisticRegression):
    """ Returns the RBC objects containing the models to be evaluated.

    Parameters
    ----------
    gmm_densities : List[GaussianMixture]
        gaussian mixture densities calculated in the training stage
    trained_lr_model : LogisticRegression
        logistic regression (LR) model after training in the training stage

    Returns
    -------
    rbc_objects : Dict[str, RBC]
        dictionary containing one RBC object per model to evaluate

    """
    logging.debug("Creating instance of RBC objects for Gaussian Mixture, Logistic Regression (LR) and Scaled "
                  "Index models")

    # Extract the desired settings from the configuration file
    transition_matrix = Config.transition_matrix[Config.scenario]
    classes = Config.classes[Config.scenario]
    classes_prior = Config.prior_probabilities[Config.scenario]

    # Create dictionary object to store the RBC objects
    rbc_objects = {}

    # RBC object for GMM model
    rbc_objects["GMM"] = RBC(transition_matrix=transition_matrix, classes=classes, gmm_densities=gmm_densities,
                             classes_prior=classes_prior, model="GMM")

    # RBC object for Scaled Index model
    rbc_objects["Scaled Index"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                      gmm_densities=gmm_densities,
                                      classes_prior=classes_prior, model="Scaled Index")

    # RBC object for Logistic Regression model
    rbc_objects["Logistic Regression"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                             gmm_densities=gmm_densities, classes_prior=classes_prior,
                                             model="Logistic Regression")
    rbc_objects["Logistic Regression"].set_lr_trained_model(trained_model=trained_lr_model)

    # Benchmark Deep Learning models for the water mapping experiment
    if Config.scenario == "oroville_dam":
        # RBC object for DeepWaterMap algorithm
        rbc_objects["DeepWaterMap"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                                 gmm_densities=gmm_densities, classes_prior=classes_prior,
                                                 model="DeepWaterMap")
        rbc_objects["DeepWaterMap"].set_lr_trained_model(trained_model=trained_lr_model)

        # RBC object for WatNet algorithm
        rbc_objects["WatNet"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                                 gmm_densities=gmm_densities, classes_prior=classes_prior,
                                                 model="WatNet")
        rbc_objects["WatNet"].set_lr_trained_model(trained_model=trained_lr_model)

    # Returns the dictionary with type Dict[RBC]
    return rbc_objects
