import logging

import numpy as np
import pickle
import os

from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from configuration import Config
from benchmark_models.benchmark import main_deepwater
from benchmark_models.watnet.watnet_infer import watnet_infer_main
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
    class_marginal_probabilities : list
        list containing the prior probability for each evaluated class
    model : str
        model string identifier

    Class Methods
    -------------
    set_lr_trained_model(cls, trained_model)
    get_rbc_objects(self, gmm_densities, trained_lr_model)
    update_labels(self, image_all_bands, time_index)
    calculate_prediction(self, image_all_bands, time_index)
    calculate_posterior(self, bands, likelihood, model, pixel_position)

    """

    def __init__(self, transition_matrix: np.ndarray, gmm_densities: List[GaussianMixture], classes: list,
                 class_marginal_probabilities: list, model: str):
        """ Initializes instance of RBC object with the corresponding class attributes.

        """
        # Calculate the total number of pixels in the image (whole image - training area)
        self.total_num_pixels = Config.image_dimensions[Config.scenario]['dim_x'] * \
                                Config.image_dimensions[Config.scenario]['dim_y']
        self.prior_probability = None  # defined when updating the labels for the first time
        self.transition_matrix = transition_matrix
        self.classes = classes
        self.class_marginal_probabilities = class_marginal_probabilities
        self.num_classes = len(classes)
        self.model = model
        self.densities = gmm_densities
        # print(f"model name {self.model} and transition probability matrix is {self.transition_matrix}")

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
        likelihood : Dict[str, np.ndarray]
            dictionary containing the prior probabilities or likelihood for each model
        posterior : Dict[str, np.ndarray]
            dictionary containing the posterior probabilities for each model
        
        """
        # Get the index of the pixels that define the scene selected in the configuration file
        index_pixels_of_interest = get_index_pixels_of_interest(image_all_bands=image_all_bands,
                                                                scene_id=Config.scene_id)

        # Calculate the prior probability (time_index = 0)
        # or the likelihood or base model posterior (time_index > 0)
        # The way to calculate this depends on the model under evaluation
        # A likelihood value is calculated for each pixel and class
        base_model_predicted_probabilities = self.calculate_prediction(image_all_bands=image_all_bands)  # return this value for histogram
        # analysis 

        #  At time instant 0, the prior probability is equal to:
        #   - the model prediction, in the case of having a pretrained model
        #   - the softmax probability, otherwise (this is what is returned when calculating the prediction)
        # *likelihood* corresponds to the likelihood or the base model posterior
        if time_index == 0:
            self.prior_probability = base_model_predicted_probabilities

        """
        # TODO: This line was added to avoid a dimensional error, but it should be checked whether it could be removed
        if np.ndim(likelihood) == 3:
            likelihood = likelihood.reshape(self.total_num_pixels, self.num_classes)
        """

        # Image with the posterior probabilities for each pixel is calculated
        # Only the pixels from the chosen scene are considered
        # The calculate_posterior function is applied to all the rows
        # Each row corresponds to all the bands values for a specific pixel
        """
        del_l = np.apply_along_axis(self.calculate_posterior, 1, image_all_bands[index_pixels_of_interest,], likelihood, self.model)
        del_l = del_l.reshape((del_l.shape[0], self.num_classes))
        self.posterior_probability[index_pixels_of_interest] = del_l  # we just save the sub-scene pixels

        """
        # Initialize posterior probability
        posterior_probability = np.zeros(shape=[self.total_num_pixels, self.num_classes])
        # A posterior value is calculated for each pixel and class
        # Only pixels belonging to the sub-scene are considered
        for pixel_i in index_pixels_of_interest:
            posterior_probability[pixel_i, :] = self.calculate_posterior(bands=image_all_bands[pixel_i, :], base_model_predicted_probabilities=base_model_predicted_probabilities, pixel_position=pixel_i)

        # Calculate the class predictions
        # Values in the predictions correspond to the index of the class that is more probable
        recursive_class_prediction = posterior_probability.argmax(axis=1)
        base_model_class_prediction = base_model_predicted_probabilities.argmax(axis=1)

        return base_model_class_prediction, recursive_class_prediction, base_model_predicted_probabilities, self.prior_probability

    def calculate_prediction(self, image_all_bands: np.ndarray):
        """ Returns the prior/likelihood and posterior probabilities for each evaluated model.

        Parameters
        ----------
        image_all_bands : np.ndarray
            pixel values for all the bands of the target image

        Returns
        -------
        likelihood : np.ndarray
            prior probabilities or likelihood for the currently evaluated model

        """
        # --------------------------------------------------------------------
        # Scaled Index Classifier (SIC)
        # --------------------------------------------------------------------
        if self.model == "Scaled Index":

            # Calculate spectral index, which has range [-1, 1]
            spectral_index = get_broadband_index(data=image_all_bands,
                                                 bands=Config.bands_spectral_index[Config.scenario])

            # Scale the spectral index to range [0, 1] so that it can be seen as a probability
            likelihood = get_scaled_index(spectral_index=spectral_index, num_classes=self.num_classes)

            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant
                y = y / sum(y)
                likelihood[idx] = y

        # --------------------------------------------------------------------
        # DeepWaterNet Model, used in this project for benchmarking
        # --------------------------------------------------------------------
        elif self.model == "DeepWaterMap":  # first version

            # Select only the bands used in the case of this algorithm
            image_deepwaternet_bands = image_all_bands[:, Config.bands_deepwaternet].reshape(
                Config.image_dimensions[Config.scenario]['dim_x'],
                Config.image_dimensions[Config.scenario]['dim_y'], len(Config.bands_watnet))

            # Values are clipped between 0 and 1
            image_deepwaternet_bands = np.float32(
                np.clip(image_deepwaternet_bands * Config.scaling_factor_watnet, a_min=0, a_max=1))

            # The model published by the authors is called within the main_deepwater function
            water_map = main_deepwater(checkpoint_path=Config.path_checkpoints_deepwatermap,
                                       image=image_deepwaternet_bands)
            water_map = water_map.reshape(-1, 1)
            likelihood = np.concatenate((water_map, 1 - water_map), axis=1)

            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant
                y = y / sum(y)
                likelihood[idx] = y

        # --------------------------------------------------------------------
        # WatNet Model, used in this project for benchmarking
        # The WatNet model is an improved version of the DeepWaterNet model provided by their authors
        # --------------------------------------------------------------------
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
            likelihood = np.concatenate((water_map, 1 - water_map), axis=1)
            likelihood_copy = likelihood.copy().astype(np.float)
            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant
                y = y / sum(y)
                likelihood_copy[idx] = y
            likelihood = likelihood_copy

        # --------------------------------------------------------------------
        # Logistic Regression (LR)
        # --------------------------------------------------------------------
        elif self.model == "Logistic Regression":

            # The last column is not considered because it contains the spectral index values, which
            # are not required for the probability prediction
            likelihood = self.trained_model.predict_proba(image_all_bands[:, :-1])

            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant_LR
                y = y / sum(y)
                likelihood[idx] = y

        # --------------------------------------------------------------------
        # Gaussian Mixture Model (GMM)
        # --------------------------------------------------------------------
        elif self.model == "GMM":
            likelihood = np.zeros(shape=[self.total_num_pixels, self.num_classes])
            # Calculate the likelihood for each Gaussian Mixture
            # the score_samples function takes into account all the rows
            for class_idx in range(len(self.classes)):
                # Read *image_all_bands[:, :-1]*, not considering the last column because it corresponds   to the
                # spectral index values
                likelihood[:, class_idx] = np.exp(self.densities[class_idx].score_samples(image_all_bands[:, :-1]))
            sum_den = np.sum(likelihood, axis=1).reshape(self.total_num_pixels, 1)  # reshape to transpose vector
            likelihood = np.divide(likelihood, sum_den).astype('float32')
            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant_GMM
                y = y / sum(y)
                likelihood[idx] = y

        else:
            print('No prediction can be computed for the selected model.')

        return likelihood

    def calculate_posterior(self, bands: np.ndarray, base_model_predicted_probabilities: np.ndarray, pixel_position: int):
        """ Returns the posterior probabilities for each evaluated model.
        This function is called pixel by pixel.
        # This function is called row by row with the np.apply_along_axis function.

        Parameters
        ----------
        bands: np.ndarray
            band values for the evaluated pixel
        likelihood : np.ndarray
            prior probabilities/likelihood for each class
        pixel_position: int
            position of the pixel being evaluated (as the function is called row by row)

        Returns
        -------
        posterior : np.ndarray
            array containing the posterior probability for each class

        """
        # Posterior probability is calculated here. eq1.
        # self.class_marginal_probabilities[ct_p] = marginal probability
        post_return = np.zeros(shape=[self.num_classes, self.num_classes])
        if self.model != "GMM":
            # --------------------------------------------------------------------
            # Recursive Bayesian Discriminative Model (RBDM) - Paper Equation (5)
            # Logistic Regression, Spectral Index Classifier (SIC), Deep Learning Models
            # --------------------------------------------------------------------
            # Discriminative prediction
            prediction_pixel = base_model_predicted_probabilities[pixel_position]  # in this case, corresponds to p(class|pixel)
            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):
                    denominator_sum = 0
                    # Loop to compute second summation (over C_t^prime)
                    for c_t_p in range(self.num_classes):
                        denominator_sum += (prediction_pixel[c_t_p] / self.class_marginal_probabilities[c_t_p]) * self.transition_matrix[c_t_p, c_t_1]
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[pixel_position, c_t_1]
                    post_return[c_t, c_t_1] = numerator / denominator_sum
                post_return[c_t, :] = post_return[c_t, :] * (prediction_pixel[c_t] / self.class_marginal_probabilities[c_t])
            # These reshaping operations might be necessary only for some of the methods
            # TODO: check
            posterior = np.sum(post_return, axis=1).reshape(-1, )
            self.prior_probability[pixel_position] = (posterior / np.sum(posterior)).reshape(self.prior_probability[pixel_position].shape)
        else:
            # --------------------------------------------------------------------
            # Recursive Bayesian Generative Model (RBGM) - Paper Equation (3)
            # Gaussian Mixture Model (GMM)
            # --------------------------------------------------------------------
            #lik_lb = likelihood[pixel_position] / np.sum(likelihood[pixel_position])
            likelihood_pixel = base_model_predicted_probabilities[pixel_position] / np.sum(base_model_predicted_probabilities[pixel_position])  # in this case, corresponds to p(pixel|class)
            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):
                    denominator_sum = 0
                    # Loop to compute second summation (over C_t^prime)
                    for c_t_p in range(self.num_classes):
                        denominator_sum += (likelihood_pixel[c_t_p]) * self.transition_matrix[c_t_p, c_t_1]
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[pixel_position, c_t_1]
                    post_return[c_t, c_t_1] = numerator / denominator_sum
                post_return[c_t, :] = post_return[c_t, :] * (likelihood_pixel[c_t])
            posterior = np.sum(post_return, axis=1)  # First summation is computed here
            self.prior_probability[pixel_position] = (posterior / np.sum(posterior))  # The prior for the future time instant is the current posterior
        posterior = posterior / np.sum(posterior)
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
    class_marginal_probabilities = Config.class_marginal_probabilities[Config.scenario]

    # Create dictionary object to store the RBC objects
    rbc_objects = {}

    # RBC object for GMM model
    transition_matrix =np.array([1 - Config.eps_GMM, Config.eps_GMM, Config.eps_GMM, 1 - Config.eps_GMM]).reshape(2, 2)
    rbc_objects["GMM"] = RBC(transition_matrix=transition_matrix, classes=classes, gmm_densities=gmm_densities,
                             class_marginal_probabilities=class_marginal_probabilities, model="GMM")

    # RBC object for Scaled Index model
    transition_matrix =np.array([1 - Config.eps, Config.eps, Config.eps, 1 - Config.eps]).reshape(2, 2)
    rbc_objects["Scaled Index"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                      gmm_densities=gmm_densities,
                                      class_marginal_probabilities=class_marginal_probabilities, model="Scaled Index")

    # RBC object for Logistic Regression model
    transition_matrix = np.array([1 - Config.eps_LR, Config.eps_LR, Config.eps_LR, 1 - Config.eps_LR]).reshape(2, 2)
    rbc_objects["Logistic Regression"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                             gmm_densities=gmm_densities, class_marginal_probabilities=class_marginal_probabilities,
                                             model="Logistic Regression")
    rbc_objects["Logistic Regression"].set_lr_trained_model(trained_model=trained_lr_model)

    # Benchmark Deep Learning models for the water mapping experiment
    if Config.scenario == "oroville_dam":
        # RBC object for DeepWaterMap algorithm
        rbc_objects["DeepWaterMap"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                          gmm_densities=gmm_densities, class_marginal_probabilities=class_marginal_probabilities,
                                          model="DeepWaterMap")
        rbc_objects["DeepWaterMap"].set_lr_trained_model(trained_model=trained_lr_model)

        # RBC object for WatNet algorithm
        rbc_objects["WatNet"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                    gmm_densities=gmm_densities, class_marginal_probabilities=class_marginal_probabilities,
                                    model="WatNet")
        rbc_objects["WatNet"].set_lr_trained_model(trained_model=trained_lr_model)

    # Returns the dictionary with type Dict[RBC]
    return rbc_objects
