
import logging
import os
import pickle
from typing import List, Dict

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from image_reader import ReadSentinel2
from configuration import Config, Debug, Visual
from bayesian_recursive import RBC, get_rbc_objects
from tools.spectral_index import get_broadband_index
from tools.path_operations import get_num_images_in_folder
from plot_figures.classification_figure import ClassificationFigure
from plot_figures.appendix_figure import AppendixFigure
from plot_figures.box_plot import plot_qa_boxplot
from plot_figures.plot_sensitivity_analysis_results import plot_results

def evaluate_models(image_all_bands: np.ndarray, rbc_objects: Dict[str, RBC], time_index: int, image_index: int,
                    date_string: str):
    """ Returns the prior/likelihood and posterior probabilities for each evaluated model.

    Parameters
    ----------
    image_all_bands : np.ndarray
        array containing the bands of the image under evaluation
    rbc_objects : Dict[str, RBC]
        dictionary containing one RBC object per model to evaluate
    time_index : int
        index of the current time instant of the Bayesian process
    imange_index : int
        index of the current image used for evaluation (associated with one date)
    date_string : str
        contains the date of the read image

    Returns
    -------
    likelihood : Dict[str, np.ndarray]
        dictionary containing the prior probabilities or likelihood for each model
    posterior : Dict[str, np.ndarray]
        dictionary containing the posterior probabilities for each model

    """
    # Get the relative path where the pickle file containing evaluation plot_figures
    pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                    f'{Config.scenario}_{Config.test_site}', 'posteriors',
                                    f'{Config.scenario}_{Config.test_site}_image_{image_index}_confID_{Config.conf_id}.pkl')

    logging.debug(f"Evaluation results are generated for image {image_index}")

    # Initialize empty dictionaries with the prior and posterior probabilities
    base_model_predicted_class = {}
    posterior = {}
    prediction_float = {}
    posterior_probabilities = {}

    # Reshape the image posterior probabilities
    # Reshape the labels (*likelihood*) to compare with the image posterior probabilities
    dim_x = Config.image_dimensions[Config.scenario]['dim_x']
    dim_y = Config.image_dimensions[Config.scenario]['dim_y']

    # GMM Model
    # Update transition matrix
    eps = Config.eps_GMM[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site in [1, 2, 3]:
        eps_aux = Config.eps_GMM_adaptive[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["GMM"].transition_matrix = transition_matrix
    # Evaluate
    base_model_predicted_class["GMM"], posterior["GMM"], prediction_float["GMM"], posterior_probabilities["GMM"] = \
        rbc_objects["GMM"].update_labels(
            image_all_bands=image_all_bands,
            time_index=time_index, image_idx=image_index)
    posterior["GMM"] = posterior["GMM"].reshape(dim_x, dim_y)
    base_model_predicted_class["GMM"] = base_model_predicted_class["GMM"].reshape(dim_x, dim_y)
    # plt.figure(), plt.imshow(likelihood["GMM"])
    # plt.figure(), plt.imshow(posterior["GMM"])

    # Scaled Index Model
    # Update transition matrix
    eps = Config.eps_SIC[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site in [1, 2, 3]:
        eps_aux = Config.eps[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["Scaled Index"].transition_matrix = transition_matrix
    # Evaluate
    base_model_predicted_class["Scaled Index"], posterior["Scaled Index"], prediction_float["Scaled Index"], \
    posterior_probabilities["Scaled Index"] = rbc_objects[
        "Scaled Index"].update_labels(
        image_all_bands=image_all_bands, time_index=time_index, image_idx=image_index)
    posterior["Scaled Index"] = posterior["Scaled Index"].reshape(dim_x, dim_y)
    base_model_predicted_class["Scaled Index"] = base_model_predicted_class["Scaled Index"].reshape(dim_x, dim_y)
    # plt.figure(), plt.imshow(likelihood["Scaled Index"])
    # plt.figure(), plt.imshow(posterior["Scaled Index"])

    # Logistic Regression Model
    # Update transition matrix
    eps = Config.eps_LR[Config.test_site]
    if Config.scenario == "charles_river_3classes":
        transition_matrix = np.array(
            [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
            3, 3)
    elif Config.test_site in [1, 2, 3]:
        eps_aux = Config.eps_LR_adaptive[Config.test_site]
        transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
    else:
        transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
    rbc_objects["Logistic Regression"].transition_matrix = transition_matrix
    # Evaluate
    base_model_predicted_class["Logistic Regression"], posterior["Logistic Regression"], prediction_float[
        "Logistic Regression"], posterior_probabilities["Logistic Regression"] = \
        rbc_objects[
            "Logistic Regression"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index, image_idx=image_index)
    posterior["Logistic Regression"] = posterior["Logistic Regression"].reshape(dim_x, dim_y)
    base_model_predicted_class["Logistic Regression"] = base_model_predicted_class["Logistic Regression"].reshape(dim_x,
                                                                                                                  dim_y)
    # Benchmark Deep Learning models for the water mapping experiment
    if Config.scenario == "oroville_dam" or Config.scenario == "charles_river":

        # WatNet Algorithm
        # Update transition matrix
        eps = Config.eps_WN[Config.test_site]
        if Config.scenario == "charles_river_3classes":
            transition_matrix = np.array(
                [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
                3, 3)
        elif Config.test_site in [1, 2, 3]:
            eps_aux = Config.eps_DWM_adaptive[Config.test_site]
            transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
        else:
            transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
        rbc_objects["WatNet"].transition_matrix = transition_matrix
        # Evaluate
        base_model_predicted_class["WatNet"], posterior["WatNet"], prediction_float["WatNet"], posterior_probabilities[
            "WatNet"] = rbc_objects["WatNet"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index, image_idx=image_index)
        posterior["WatNet"] = posterior["WatNet"].reshape(dim_x, dim_y)
        base_model_predicted_class["WatNet"] = base_model_predicted_class["WatNet"].reshape(dim_x, dim_y)

        # DeepWaterMap Algorithm
        # Update transition matrix
        eps = Config.eps_DWM[Config.test_site]
        if Config.scenario == "charles_river_3classes":
            transition_matrix = np.array(
                [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
                3, 3)
        elif Config.test_site in [1, 2, 3]:
            eps_aux = Config.eps_DWM_adaptive[Config.test_site]
            transition_matrix = np.array([1 - eps_aux, eps_aux, eps_aux, 1 - eps_aux]).reshape(2, 2)
        else:
            transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)
        rbc_objects["DeepWaterMap"].transition_matrix = transition_matrix
        # Evaluate
        base_model_predicted_class["DeepWaterMap"], posterior["DeepWaterMap"], prediction_float["DeepWaterMap"], \
        posterior_probabilities["DeepWaterMap"] = rbc_objects[
            "DeepWaterMap"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index, image_idx=image_index)
        posterior["DeepWaterMap"] = posterior["DeepWaterMap"].reshape(dim_x, dim_y)
        base_model_predicted_class["DeepWaterMap"] = base_model_predicted_class["DeepWaterMap"].reshape(dim_x, dim_y)

    # Dump data into pickle if this image index belongs to the list containing indices of images to store
    if image_index in Config.index_images_to_store[Config.test_site] and Config.evaluation_store_results:
        pickle.dump([base_model_predicted_class, posterior], open(pickle_file_path, 'wb'))

    # Dump data into pickle for analysis of the prediction histogram
    if Debug.pickle_histogram:
        pickle_file_path = os.path.join(Config.path_histogram_prediction, f'{Config.scenario}_{Config.test_site}',
                                        f'{Config.scenario}_{Config.test_site}_image_{image_index}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl')
        pickle.dump([prediction_float, date_string], open(pickle_file_path, 'wb'))

    """
    logging.debug(f"Evaluation results are already available --> Loading evaluation data for image {image_index}")
    [likelihood, posterior] = pickle.load(open(pickle_file_path, 'rb'))
    """
    return base_model_predicted_class, posterior, prediction_float, posterior_probabilities


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
    None (plots plot_figures for the time indexes specified in the configuration file)

    """

    # Initialize one RBC object for each model to be compared
    rbc_objects = get_rbc_objects(gmm_densities=gmm_densities, trained_lr_model=trained_lr_model)

    # Read the images in the evaluation folder
    # Path where evaluation images are stored
    logging.debug("Start Evaluation")

    # Path where the evaluation images are stored
    path_evaluation_images = os.path.join(Config.path_sentinel_images, Config.scenario, 'evaluation')
    # path_evaluation_images = "/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset/sent2/-54.58_-3.43"

    # It is assumed that all the band folders have the same number of stored images.
    # Therefore, to count training images we can check the folder associated to any of the bands.
    # For simplicity, we select the first available band.
    path_first_band = os.path.join(path_evaluation_images, f'Sentinel2_B{Config.bands_to_read[0]}')

    # Get the number of images that match format requirements in the evaluation dataset folder
    if Config.num_evaluation_images_hardcoded == 0:
        num_evaluation_images = get_num_images_in_folder(path_folder=path_first_band,
                                                         image_type=Config.image_types[Config.scenario],
                                                         file_extension=Config.file_extension[Config.scenario])
    else:
        num_evaluation_images = Config.num_evaluation_images_hardcoded
    logging.debug(f"Number of available images for evaluation = {num_evaluation_images}")

    time_index = 0
    date_string_list = []  # initialize list to store Evaluation dates

    if Visual.class_fig_settings['plot']:
        results_figure = ClassificationFigure()
    if Visual.appendix_fig_settings['plot']:
        appendix_figure = AppendixFigure()

    # Loop over every image (i.e., date)
    for image_idx in Config.index_images_to_evaluate[Config.test_site]:

        # All bands of the image with index *image_idx* are stored in *image_all_bands*
        image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images, image_idx=image_idx,
                                                               shift_option="None")

        # If Debug.check_dates is False, we Evaluate the models.
        # Otherwise, we only print image dates
        if not Debug.check_dates:

            # Calculate spectral index for all bands and add it to image_all_bands vector
            index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])
            image_all_bands = np.hstack([image_all_bands, index.reshape(-1, 1)])

            # [GENERATE or LOAD results]
            if Config.evaluation_generate_results:
                predicted_class, posterior, predicted_probabilities, posterior_probabilities = evaluate_models(
                    image_all_bands=image_all_bands, rbc_objects=rbc_objects,
                    time_index=time_index, image_index=image_idx,
                    date_string=date_string)
            else:
                [predicted_class, posterior] = load_results(image_idx=image_idx)

            # [APPENDIX] Plot one date results in Appendix figure
            if Visual.appendix_fig_settings['plot']:
                result_idx = Config.index_images_to_evaluate[Config.test_site].index(image_idx)
                appendix_figure.plot_results(image_idx=image_idx, image_all_bands=image_all_bands,
                                            base_model_predicted_class=predicted_class,date_string=date_string,
                                            posterior=posterior, result_idx=result_idx)
                print('Appendix Results have been plotted for the current date')

            # [QUANTITATIVE ANALYSIS] Plot one date results in Classification figure
            print(f"plotting results for image with index {image_idx}")
            if Visual.class_fig_settings['plot'] and image_idx in Config.index_images_to_plot[Config.test_site]:
                result_idx = Config.index_images_to_plot[Config.test_site].index(image_idx)
                results_figure.plot_results(image_idx=image_idx, image_all_bands=image_all_bands,
                                            base_model_predicted_class=predicted_class,
                                            posterior=posterior, result_idx=result_idx)
                print('Quantitative Analysis Results have been plotted for current date')
            time_index = time_index + 1
        date_string_list.append(date_string)
    path_results = os.path.join(Config.path_evaluation_results, "classification")

    #
    # Sensitivity Analysis
    if Debug.store_pickle_sensitivity_analysis and Config.test_site in ['1a', '3']:
        #
        # ----- Save Quantitative Analysis (QA) Results - Pickle with accuracies
        path_save_pickle = os.path.join(Config.path_evaluation_results, "sensitivity_analysis",
                                     f"{Config.scenario}", 'results', f"eps_{Config.eps_SIC[Config.test_site]}")
        pickle.dump(results_figure.results_qa, open(path_save_pickle, 'wb'))
        path_save_figure =  os.path.join(Config.path_evaluation_results, "sensitivity_analysis",
                                     f"{Config.scenario}", "sensitivity_analysis.svg")
        # Set up paths and configurations
        results_path = os.path.join(Config.path_evaluation_results, "sensitivity_analysis",
                                     f"{Config.scenario}", 'results')
        plot_results(save_path=path_save_figure, Config=Config, results_path=results_path)

    if Visual.appendix_fig_settings['save']:
        settings = Visual.appendix_fig_settings[Config.test_site]
       # Adjust layout
        appendix_figure.f.subplots_adjust(wspace=settings['wspace'], hspace=settings['hspace'], top=settings['top'],
                                          right=settings['right'], left=settings['left'], bottom=settings['bottom'])
        # Save figure
        path_save_fig = os.path.join(Config.path_evaluation_results, "classification",f"{Config.scenario}_{Config.test_site}",
                                     f"figures", f"fig_config_{Config.conf_id}_appendix.svg")
        appendix_figure.f.savefig(path_save_fig, bbox_inches='tight', format='svg', dpi=1000)

    if Visual.class_fig_settings['save']:
        # Adjust layout
        results_figure.adjust_figure()
        # Save figure
        path_save_fig = os.path.join(Config.path_evaluation_results, "classification", f"{Config.scenario}_{Config.test_site}",
                                     f"figures", f"fig_config_{Config.conf_id}_quantitative.svg")
        results_figure.f.savefig(path_save_fig, bbox_inches='tight', format='svg', dpi=1000)

    # If QA results are saved we can
    # - (1) save balanced accuracy and models
    if Config.qa_settings['save']:
        path_results_metrics = os.path.join(path_results, f'{Config.scenario}_{Config.test_site}', f'accuracy',
                                            f'conf_{Config.conf_id}')
        pickle.dump(results_figure.plot_legend[:-1], open(os.path.join(path_results_metrics, "models.pkl"), "wb"))
        for acc in Config.qa_settings['metrics']:
            path_i = os.path.join(path_results_metrics, f'{acc}.pkl')
            pickle.dump(results_figure.results_qa[acc], open(path_i, "wb"))
    # - (2) compute boxplot
    if Visual.qa_fig_settings['plot']:
        plot_qa_boxplot(Config=Config)
    print('Evaluation Main is FINISHED')


def check_index_threshold(image: np.ndarray, rbc_objects: Dict[str, RBC], date):
    """ Returns True if the instant spectral index classifier (SIC) returns abnormal pixel classification.
    This function has been thought for the experiment with the MultiEarth 2023 dataset, to discard images where a
    suspiciously high percentage of pixels is classified as deforested.

    Parameters
    ----------
    image: np.ndarray
    processed image for cloud detection
    rbc_objects : Dict[str, RBC]
        dictionary containing one RBC object per model to evaluate

    Returns
    -------
    accepted_image: bool
    True if the instant spectral index classifier (SIC) returns normal pixel classification

    """
    # Scaled Index Model
    dim_x = Config.image_dimensions[Config.scenario]['dim_x']
    dim_y = Config.image_dimensions[Config.scenario]['dim_y']
    likelihood_sic, _, _, _ = rbc_objects["Scaled Index"].update_labels(image_all_bands=image, time_index=0)
    likelihood_sic_image = likelihood_sic.reshape(dim_x, dim_y)
    percentage_deforested = np.sum(likelihood_sic) / likelihood_sic.size
    # plt.figure(), plt.imshow(likelihood["Scaled Index"])
    # plt.figure(), plt.imshow(posterior["Scaled Index"])
    if percentage_deforested > 0.5:
        accepted_image = False
        print(
            f"Image with date {date} has been discarded due to {percentage_deforested}% pixels classified as deforested")
    else:
        accepted_image = True
    return accepted_image

def load_results(image_idx: int):
    if Config.test_site == '2':
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.test_site}',
                                        f'{Config.scenario}_3_image_{image_idx}_epsilon_'+r"{1: 0.05, 2: 0.05, 3: 0.05, 4: 0.01}.pkl")
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.test_site}',
                                        f'{Config.scenario}_{Config.test_site}_image_{image_idx}_confID_{Config.conf_id}.pkl')

    elif Config.test_site == '1a':
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.test_site}',
                                        f'{Config.scenario}_1_image_{image_idx}_EPS_?.pkl')
    elif Config.test_site == '3':
        a = {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.01}
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_4_image_{image_idx}_epsilon_{a}.pkl')
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.test_site}',
                                        f'{Config.scenario}_{Config.test_site}_image_{image_idx}_confID_{Config.conf_id}.pkl')
    pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                    f'{Config.scenario}_{Config.test_site}', 'posteriors',
                                    f'{Config.scenario}_{Config.test_site}_image_{image_idx}_confID_{Config.conf_id}.pkl')
    [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))
    return [predicted_class, posterior]
