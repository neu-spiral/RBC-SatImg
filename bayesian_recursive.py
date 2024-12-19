import logging
import time

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

    def update_labels(self, image_all_bands: np.ndarray, time_index: int = 0, image_idx: int = 0):
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
                                                                scene_id=Config.test_site)

        pickle_file_path_likelihood = os.path.join(Config.path_evaluation_results, "classification",
                                                   f"{Config.scenario}_{Config.test_site}", "likelihoods",
                                                   f"image_{image_idx}_model_{self.model}_likelihood")
        if Config.evaluation_generate_results_likelihood:
            # Calculate the prior probability (time_index = 0)
            # or the likelihood or base model posterior (time_index > 0)
            # The way to calculate this depends on the model under evaluation
            # A likelihood value is calculated for each pixel and class
            [base_model_predicted_probabilities, prediction_execution_time] = self.calculate_prediction(
                image_all_bands=image_all_bands, index_pixels_of_interest=index_pixels_of_interest)  # return this value for histogram
            # analysis
            pickle.dump(base_model_predicted_probabilities, open(pickle_file_path_likelihood, 'wb'))
        else:
            base_model_predicted_probabilities = pickle.load(open(pickle_file_path_likelihood, 'rb'))
            prediction_execution_time = 'N/A'

        #  At time instant 0, the prior probability is equal to:
        #   - the model prediction, in the case of having a pretrained model
        #   - the softmax probability, otherwise (this is what is returned when calculating the prediction)
        # *likelihood* corresponds to the likelihood or the base model posterior
        if time_index == 0:
            marginal_probs = np.array(Config.class_marginal_probabilities[Config.scenario])
            marginal_probs = np.repeat(marginal_probs.reshape(1, 2), base_model_predicted_probabilities.shape[0], axis=0)
            self.prior_probability = marginal_probs


        # Initialize posterior probability
        #posterior_probability = np.zeros(shape=[self.total_num_pixels, self.num_classes])
        # A posterior value is calculated for each pixel and class
        # Only pixels belonging to the sub-scene are considered

        # Deprecated because it is not efficient
        #for pixel_i in index_pixels_of_interest:
        #    [posterior_probability[pixel_i, :], time] = self.calculate_posterior(bands=image_all_bands[pixel_i, :], base_model_predicted_probabilities=base_model_predicted_probabilities, pixel_position=pixel_i)


        [posterior_probability, recursion_execution_time] = self.calculate_posterior_efficient(bands=image_all_bands, base_model_predicted_probabilities=base_model_predicted_probabilities, index_pixels_of_interest=index_pixels_of_interest)
        #total_time = recursion_execution_time + prediction_execution_time
        #print(f"Total execution time {self.model}: {total_time} seconds")
        #print(f"Recursion percentage {self.model}: {recursion_execution_time/total_time*100} seconds")
        print(f"Recursion time {self.model}: {recursion_execution_time} s")
        print(f"Baseline time {self.model}: {prediction_execution_time} s")

        # Calculate the class predictions
        # Values in the predictions correspond to the index of the class that is more probable
        recursive_class_prediction = posterior_probability.argmax(axis=1)
        base_model_class_prediction = base_model_predicted_probabilities.argmax(axis=1)

        # Return
        # base_model_class_prediction = classification result from instantaneous classifier
        # recursive_class_prediction = classification result from RBC
        # base_model_predicted_probabilities = probability values from instantaneous classifier
        # self.prior_probability = probability values from RBC (posterior probabilities)
        return base_model_class_prediction, recursive_class_prediction, base_model_predicted_probabilities, self.prior_probability

    def calculate_prediction(self, image_all_bands: np.ndarray, index_pixels_of_interest: np.ndarray):
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
            likelihood_return = np.zeros(shape=[self.total_num_pixels, self.num_classes])
            start_time = time.time()
            # Calculate spectral index, which has range [-1, 1]
            spectral_index = get_broadband_index(data=image_all_bands[index_pixels_of_interest,:],
                                                 bands=Config.bands_spectral_index[Config.scenario])

            # Scale the spectral index to range [0, 1] so that it can be seen as a probability
            likelihood = get_scaled_index(spectral_index=spectral_index, num_classes=self.num_classes)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time LIKELIHOOD {self.model}: {execution_time} seconds")
            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant_SIC[Config.test_site]
                y = y / sum(y)
                likelihood[idx] = y
            likelihood_return[index_pixels_of_interest,:] = likelihood
            likelihood = likelihood_return

        # --------------------------------------------------------------------
        # DeepWaterNet Model, used in this project for benchmarking
        # --------------------------------------------------------------------
        elif self.model == "DeepWaterMap":  # first version

            # Extract coordinates and dimensions
            coords = Config.pixel_coords_to_evaluate[Config.test_site]
            dim_x = coords['x_coords'][1] - coords['x_coords'][0]
            dim_y = coords['y_coords'][1] - coords['y_coords'][0]

            # Select only the bands of interest
            image_of_interest = image_all_bands[index_pixels_of_interest][:, Config.bands_watnet]

            # Reshape and scale image
            image_deepwaternet_bands = image_of_interest.reshape(dim_x, dim_y, len(Config.bands_watnet))
            image_deepwaternet_bands = np.clip(image_deepwaternet_bands * Config.scaling_factor_watnet, 0, 1).astype(
                np.float32)

            # Model inference
            start_time = time.time()
            water_map = main_deepwater(checkpoint_path=Config.path_checkpoints_deepwatermap,
                                       image=image_deepwaternet_bands)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time LIKELIHOOD {self.model}: {execution_time} seconds")

            # Reshape water_map and compute likelihood
            water_map = water_map.reshape(-1, 1)
            likelihood = np.concatenate((water_map, 1 - water_map), axis=1)

            # Convert norm_constant_DL to a numpy array for broadcasting
            norm_constant = np.array(Config.norm_constant_DL[Config.test_site], dtype=np.float32)

            # Normalize likelihood in a vectorized manner
            likelihood += norm_constant
            likelihood /= likelihood.sum(axis=1, keepdims=True)

            # Prepare likelihood return array
            likelihood_return = np.zeros((self.total_num_pixels, self.num_classes), dtype=np.float32)
            likelihood_return[index_pixels_of_interest] = likelihood

            # Assign final likelihood
            likelihood = likelihood_return


        # --------------------------------------------------------------------
        # WatNet Model, used in this project for benchmarking
        # The WatNet model is an improved version of the DeepWaterNet model provided by their authors
        # --------------------------------------------------------------------
        elif self.model == "WatNet":


            # Extract coordinates and dimensions
            coords = Config.pixel_coords_to_evaluate[Config.test_site]
            dim_x = coords['x_coords'][1] - coords['x_coords'][0]
            dim_y = coords['y_coords'][1] - coords['y_coords'][0]

            # Select only the bands of interest
            image_of_interest = image_all_bands[index_pixels_of_interest, :]
            image_of_interest = image_of_interest[:,Config.bands_watnet]

            # Reshape and scale image
            image_watnet_bands = image_of_interest.reshape(dim_x, dim_y, len(Config.bands_watnet))
            image_watnet_bands = np.clip(image_watnet_bands * Config.scaling_factor_watnet, 0, 1).astype(np.float32)

            # Call the model inference function
            start_time = time.time()
            water_map = watnet_infer_main(rsimg=image_watnet_bands,
                                          path_model=Config.path_watnet_pretrained_model).reshape(-1, 1)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time LIKELIHOOD {self.model}: {execution_time} seconds")

            # Compute likelihood
            likelihood = np.concatenate((water_map, 1 - water_map), axis=1)

            # Normalize likelihood
            norm_constant = np.array(Config.norm_constant_DL[Config.test_site])
            likelihood = likelihood + norm_constant
            likelihood /= np.sum(likelihood, axis=1, keepdims=True)

            # Initialize likelihood return array
            likelihood_return = np.zeros((self.total_num_pixels, likelihood.shape[1]), dtype=np.float32)

            # Assign likelihood to the relevant indices
            likelihood_return[index_pixels_of_interest] = likelihood

            # Final likelihood
            likelihood = likelihood_return

        # --------------------------------------------------------------------
        # Logistic Regression (LR)
        # --------------------------------------------------------------------
        elif self.model == "Logistic Regression":
            likelihood = np.zeros(shape=[self.total_num_pixels, self.num_classes])
            # The last column is not considered because it contains the spectral index values, which
            # are not required for the probability prediction
            start_time = time.time()
            likelihood[index_pixels_of_interest, :] = self.trained_model.predict_proba(image_all_bands[index_pixels_of_interest, :-1])
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time LIKELIHOOD {self.model}: {execution_time} seconds")

            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant_LR[Config.test_site]
                y = y / sum(y)
                likelihood[idx] = y

        # --------------------------------------------------------------------
        # Gaussian Mixture Model (GMM)
        # --------------------------------------------------------------------
        elif self.model == "GMM":
            likelihood = np.zeros(shape=[self.total_num_pixels, self.num_classes])
            # Calculate the likelihood for each Gaussian Mixture
            # the score_samples function takes into account all the rows
            start_time = time.time()
            for class_idx in range(len(self.classes)):
                # Read *image_all_bands[:, :-1]*, not considering the last column because it corresponds   to the
                # spectral index values
                likelihood[index_pixels_of_interest, class_idx] = np.exp(self.densities[class_idx].score_samples(image_all_bands[index_pixels_of_interest, :-1]))
            sum_den = np.sum(likelihood, axis=1).reshape(self.total_num_pixels, 1)  # reshape to transpose vector
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time LIKELIHOOD {self.model}: {execution_time} seconds")
            likelihood = np.divide(likelihood, sum_den).astype('float32')
            for idx in range(likelihood.shape[0]):
                y = likelihood[idx] + Config.norm_constant_GMM[Config.test_site]
                y = y / sum(y)
                likelihood[idx] = y

        else:
            print('No prediction can be computed for the selected model.')

        return likelihood, execution_time

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
            prediction_pixel = base_model_predicted_probabilities[pixel_position] / np.sum(base_model_predicted_probabilities[pixel_position])
            start_time = time.time()

            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):

                    # Denominator for C_{t-1}:
                    denominator_sum = 0
                    # Loop to compute second summation (over C_t^prime)
                    for c_t_p in range(self.num_classes):
                        third_summation = 0
                        for c_t_1_p in range(self.num_classes):
                            third_summation += self.transition_matrix[c_t_p, c_t_1_p] * self.prior_probability[pixel_position, c_t_1_p]
                        denominator_sum += (prediction_pixel[c_t_p] / self.class_marginal_probabilities[c_t_p]) * third_summation
                    # Numerator for C_{t-1}:
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[pixel_position, c_t_1]

                    # Put all together
                    post_return[c_t, c_t_1] = numerator / denominator_sum
                post_return[c_t, :] = post_return[c_t, :] * (prediction_pixel[c_t] / self.class_marginal_probabilities[c_t])

           # These reshaping operations might be necessary only for some of the methods
            # TODO: check
            posterior = np.sum(post_return, axis=1).reshape(-1, )
            end_time = time.time()
            self.prior_probability[pixel_position] = (posterior / np.sum(posterior)).reshape(self.prior_probability[pixel_position].shape)
        else:
            # --------------------------------------------------------------------
            # Recursive Bayesian Generative Model (RBGM) - Paper Equation (3)
            # Gaussian Mixture Model (GMM)
            # --------------------------------------------------------------------

            #lik_lb = likelihood[pixel_position] / np.sum(likelihood[pixel_position])
            likelihood_pixel = base_model_predicted_probabilities[pixel_position] / np.sum(base_model_predicted_probabilities[pixel_position])  # in this case, corresponds to p(pixel|class)
            start_time = time.time()
            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):
                    # numerator
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[pixel_position, c_t_1]
                    # denominator
                    denominator_sum = 0
                    for c_t_p in range(self.num_classes):
                        third_summation = 0
                        for c_t_1_p in range(self.num_classes):
                            third_summation += self.transition_matrix[c_t_p, c_t_1_p] * self.prior_probability[pixel_position, c_t_1_p]
                        denominator_sum += (likelihood_pixel[c_t_p]) * third_summation
                    # Put together
                    post_return[c_t, c_t_1] = numerator / denominator_sum  # fraction
                post_return[c_t, :] = post_return[c_t, :] * (likelihood_pixel[c_t])  # multiply fraction by the likelihood
            posterior = np.sum(post_return, axis=1)  # First summation is computed here
            end_time = time.time()
            self.prior_probability[pixel_position] = (posterior / np.sum(posterior))  # The prior for the future time instant is the current posterior
        posterior = posterior / np.sum(posterior)
        return posterior, end_time-start_time

    def calculate_posterior_efficient_old(self, bands: np.ndarray, base_model_predicted_probabilities: np.ndarray, index_pixels_of_interest: np.ndarray):
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
        num_pixels_of_interest = index_pixels_of_interest.shape[0] # pixels of the scene under evaluation
        post_return = np.zeros((self.total_num_pixels, self.num_classes, self.num_classes))
        likelihood = base_model_predicted_probabilities / np.sum(base_model_predicted_probabilities, axis=-1, keepdims=True)
        if self.model != "GMM": # Discriminative Model
            # --------------------------------------------------------------------
            # Recursive Bayesian Discriminative Model (RBDM) - Paper Equation (5)
            # Logistic Regression, Spectral Index Classifier (SIC), Deep Learning Models
            # --------------------------------------------------------------------
            # Discriminative prediction

            start_time = time.time()
            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):

                    # Denominator for C_{t-1}:
                    denominator_sum = np.zeros(num_pixels_of_interest)  # 0

                    # Loop to compute second summation (over C_t^prime)
                    for c_t_p in range(self.num_classes):
                        third_summation = np.zeros(num_pixels_of_interest)  # 0
                        for c_t_1_p in range(self.num_classes):
                            third_summation[:] += self.transition_matrix[c_t_p, c_t_1_p] * self.prior_probability[index_pixels_of_interest, c_t_1_p]
                        denominator_sum[:] += (likelihood[index_pixels_of_interest, c_t_p] / self.class_marginal_probabilities[
                            c_t_p]) * third_summation[:]
                    # Numerator for C_{t-1}:
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[index_pixels_of_interest, c_t_1]

                    # Put all together
                    post_return[index_pixels_of_interest, c_t, c_t_1] = numerator / denominator_sum
                # post_return[:, c_t, :] = post_return[:, c_t, :] * (prediction_pixel[:,c_t] / class_marginal_probabilities[c_t])
                post_return[index_pixels_of_interest, c_t, :] = post_return[index_pixels_of_interest, c_t, :] * (
                            np.tile(np.expand_dims(likelihood[index_pixels_of_interest, c_t], 1), (1, self.num_classes)) /
                            self.class_marginal_probabilities[c_t])

            # These reshaping operations might be necessary only for some of the methods
            # TODO: check
            posterior = np.sum(post_return, axis=2)
            end_time = time.time()
            print(end_time - start_time)

        else: # Generative Model
            # --------------------------------------------------------------------
            # Recursive Bayesian Generative Model (RBGM) - Paper Equation (3)
            # Gaussian Mixture Model (GMM)
            # --------------------------------------------------------------------

            start_time = time.time()
            # Loop to compute posterior for each class
            for c_t in range(self.num_classes):
                # Loop to compute first summation (over C_{t-1})
                for c_t_1 in range(self.num_classes):
                    # numerator
                    numerator = self.transition_matrix[c_t, c_t_1] * self.prior_probability[index_pixels_of_interest, c_t_1]
                    # denominator
                    denominator_sum = np.zeros(num_pixels_of_interest)
                    for c_t_p in range(self.num_classes):
                        third_summation = np.zeros(num_pixels_of_interest)
                        for c_t_1_p in range(self.num_classes):
                            third_summation[:] += self.transition_matrix[c_t_p, c_t_1_p] * self.prior_probability[index_pixels_of_interest, c_t_1_p]
                        denominator_sum[:] += (likelihood[index_pixels_of_interest, c_t_p]) * third_summation[:]
                    # Put together
                    post_return[index_pixels_of_interest, c_t, c_t_1] = numerator / denominator_sum  # fraction
                post_return[index_pixels_of_interest, c_t, :] = post_return[index_pixels_of_interest, c_t, :] * (
                            np.tile(np.expand_dims(likelihood[index_pixels_of_interest, c_t], 1), (1, self.num_classes)))
            posterior = np.sum(post_return, axis=2)  # First summation is computed here
            end_time = time.time()

        # Update self.prior
        normalized_posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        self.prior_probability = normalized_posterior

        return normalized_posterior, end_time-start_time

    def calculate_posterior_efficient(self, bands: np.ndarray, base_model_predicted_probabilities: np.ndarray, index_pixels_of_interest: np.ndarray):
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
        num_pixels_of_interest = index_pixels_of_interest.shape[0] # pixels of the scene under evaluation
        post_return = np.zeros((num_pixels_of_interest, self.num_classes, self.num_classes))
        likelihood = base_model_predicted_probabilities / np.sum(base_model_predicted_probabilities, axis=-1, keepdims=True)
        if self.model != "GMM": # Discriminative Model
            # --------------------------------------------------------------------
            # Recursive Bayesian Discriminative Model (RBDM) - Paper Equation (5)
            # Logistic Regression, Spectral Index Classifier (SIC), Deep Learning Models
            # --------------------------------------------------------------------
            start_time = time.time()

            # Extract relevant slices based on the indices
            prior_prob = self.prior_probability[index_pixels_of_interest, :]  # Shape: (num_pixels, num_classes)
            likelihood_pixels = likelihood[index_pixels_of_interest, :]  # Shape: (num_pixels, num_classes)

            # Compute the third summation for each pixel: Shape (num_pixels, num_classes)
            # Efficiently compute with einsum
            third_summation = np.einsum('ij,kj->ki', self.transition_matrix, prior_prob)

            # Compute the denominator for all pixels: Shape (num_pixels,)
            denominator_sum = np.einsum('ki,ki->k', likelihood_pixels, third_summation)

            # Reshape denominator_sum for broadcasting in the next step: Shape (num_pixels, 1)
            denominator_sum = denominator_sum[:, np.newaxis]  # Shape: (num_pixels, 1)

            # Compute the numerator for all pixels and classes: Shape (num_pixels, num_classes, num_classes)
            numerator_all = np.einsum('ij,kj->kij', self.transition_matrix, prior_prob)

            # Compute post_return by dividing numerator_all by denominator_sum: Shape (num_pixels, num_classes, num_classes)
            post_return = numerator_all / denominator_sum[:, np.newaxis, :]

            # Expand dimensions of likelihood_pixels for multiplication: Shape (num_pixels, num_classes, 1)
            likelihood_expanded = np.expand_dims(likelihood_pixels, axis=2)

            # Perform element-wise multiplication
            post_return *= likelihood_expanded  # Shape: (num_pixels, num_classes, num_classes)

            # Sum over the last axis to get the posterior: Shape (num_pixels, num_classes)
            posterior = np.sum(post_return, axis=2)

            end_time = time.time()



        else: # Generative Model
            # --------------------------------------------------------------------
            # Recursive Bayesian Generative Model (RBGM) - Paper Equation (3)
            # Gaussian Mixture Model (GMM)
            # --------------------------------------------------------------------
            start_time = time.time()

            # Extract relevant indices for efficiency
            prior_prob = self.prior_probability[index_pixels_of_interest, :]  # Shape: (868700, 2)
            likelihood_pixels = likelihood[index_pixels_of_interest, :]  # Shape: (868700, 2)

            # Compute the third summation
            # Shape of third_summation should be (868700, 2)
            third_summation = np.einsum('ij,kj->ki', self.transition_matrix, prior_prob)

            # Compute the denominator for each pixel
            # Shape of denominator_sum should be (868700,)
            denominator_sum = np.einsum('ki,ki->k', likelihood_pixels, third_summation)

            # Ensure that denominator_sum has the correct shape for broadcasting
            denominator_sum = denominator_sum[:, np.newaxis]  # Shape: (868700, 1)

            # Compute the numerator for all classes and pixels
            # Shape of numerator_all should be (868700, 2, 2)
            numerator_all = np.einsum('ij,kj->kij', self.transition_matrix, prior_prob)

            # Compute the post_return by dividing numerator_all by denominator_sum
            # denominator_sum should be reshaped to match (868700, 2, 2) for broadcasting
            post_return = numerator_all / denominator_sum[:, np.newaxis, :]  # Shape: (868700, 2, 2)

            # Multiply by the likelihood for each class
            # Expand dimensions of likelihood to match post_return
            likelihood_expanded = np.expand_dims(likelihood_pixels, axis=2)  # Shape: (868700, 2, 1)

            # Perform the multiplication
            post_return *= likelihood_expanded  # Shape: (868700, 2, 2)

            # Sum over the last axis to get the posterior
            # Shape of posterior should be (868700, 2)
            posterior = np.sum(post_return, axis=2)  # Shape: (868700, 2)

            end_time = time.time()

        # Update self.prior
        normalized_posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        posterior_return = np.zeros((self.total_num_pixels, self.num_classes))
        posterior_return[index_pixels_of_interest,:] = normalized_posterior
        self.prior_probability = posterior_return
        print(end_time - start_time)

        return posterior_return, end_time-start_time


    def calculate_posterior_old(self, bands: np.ndarray, base_model_predicted_probabilities: np.ndarray, pixel_position: int):
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

                    # Denominator for C_{t-1}:
                    denominator_sum = 0
                    # Loop to compute second summation (over C_t^prime)
                    for c_t_p in range(self.num_classes):
                        denominator_sum += (prediction_pixel[c_t_p] / self.class_marginal_probabilities[c_t_p]) * self.transition_matrix[c_t_p, c_t_1]

                    # Numerator for C_{t-1}:
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
    # transition_matrix = Config.transition_matrix[Config.scenario]
    classes = Config.classes[Config.scenario]
    class_marginal_probabilities = Config.class_marginal_probabilities[Config.scenario]

    # Create dictionary object to store the RBC objects
    rbc_objects = {}

    # RBC object for GMM model
    # we change the transition matrix when changing the algorithm because the epsilon is different
    eps=Config.eps_GMM[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site == 2:
        eps_aux = Config.eps_GMM_adaptive[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["GMM"] = RBC(transition_matrix=transition_matrix, classes=classes, gmm_densities=gmm_densities,
                             class_marginal_probabilities=class_marginal_probabilities, model="GMM")

    # RBC object for Scaled Index model and Deep learning models
    # we change the transition matrix when changing the algorithm because the epsilon is different
    eps = Config.eps_SIC[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site == 2:
        eps_aux = Config.eps[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["Scaled Index"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                      gmm_densities=gmm_densities,
                                      class_marginal_probabilities=class_marginal_probabilities, model="Scaled Index")

    # RBC object for Logistic Regression model
    # we change the transition matrix when changing the algorithm because the epsilon is different
    eps = Config.eps_LR[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site == 2:
        eps_aux = Config.eps_LR_adaptive[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["Logistic Regression"] = RBC(transition_matrix=transition_matrix, classes=classes,
                                             gmm_densities=gmm_densities, class_marginal_probabilities=class_marginal_probabilities,
                                             model="Logistic Regression")
    rbc_objects["Logistic Regression"].set_lr_trained_model(trained_model=trained_lr_model)

    # Benchmark Deep Learning models for the water mapping experiment
    if Config.scenario == "oroville_dam" or Config.scenario =="charles_river":
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
