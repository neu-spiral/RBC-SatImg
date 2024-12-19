import logging
import os
import pickle
from typing import List, Dict

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from tools.evaluate_models import evaluate_models
from image_reader import ReadSentinel2
from configuration import Config, Debug, Visual
from bayesian_recursive import RBC, get_rbc_objects
from tools.spectral_index import get_broadband_index
from tools.path_operations import get_num_images_in_folder
from plot_figures.classification_figure import ClassificationFigure
from plot_figures.appendix_figure import AppendixFigure
from plot_figures.box_plot import plot_qa_boxplot
from plot_figures.plot_sensitivity_analysis_results import plot_results_sensitivity_analysis

"""
evaluation.py

This script handles the evaluation stage of the recursive Bayesian classification framework
for satellite imagery analysis. It includes functions to initialize evaluation processes,
evaluate models, check thresholds, and load previously computed results.

Classification accuracy results are calculated in the process_and_plot_results function from classification_figure.py

Key Functionalities:
- Initialization of Recursive Bayesian Classification (RBC) objects.
- Evaluation of multiple models (e.g., GMM, Scaled Index, Logistic Regression, Deep Learning Models).
- Quantitative and qualitative analysis of model performance.
- Store and plot results

References:
[1] "Recursive classification of satellite imaging time-series: An application to land cover mapping"
    by Helena Calatrava et al.
    Published in ISPRS Journal of Photogrammetry and Remote Sensing, 2024.
    DOI: https://doi.org/10.1016/j.isprsjprs.2024.09.003

At the end of the `evaluation_main` function, options to plot and save the figures discussed in [1]
are provided.

Corresponding Author:
- Helena Calatrava (Northeastern University, Boston, MA, USA)
Last Updated:
- December 2024

"""


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
    logging.debug("Start Evaluation")

    # Path where the evaluation images are stored
    path_evaluation_images = os.path.join(Config.path_sentinel_images, Config.scenario, 'evaluation')
    path_first_band = os.path.join(path_evaluation_images, f'Sentinel2_B{Config.bands_to_read[0]}')

    # Get the number of images that match format requirements in the evaluation dataset folder
    if Config.num_evaluation_images_hardcoded == 0:
        num_evaluation_images = get_num_images_in_folder(
            path_folder=path_first_band,
            image_type=Config.image_types[Config.scenario],
            file_extension=Config.file_extension[Config.scenario]
        )
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
        image_all_bands, date_string = image_reader.read_image(
            path=path_evaluation_images, image_idx=image_idx, shift_option="None"
        )

        # If Debug.check_dates is False, we Evaluate the models.
        # Otherwise, we only print image dates
        if not Debug.check_dates:

            # Calculate spectral index for all bands and add it to image_all_bands vector
            index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])
            image_all_bands = np.hstack([image_all_bands, index.reshape(-1, 1)])

            # [GENERATE or LOAD results]
            if Config.evaluation_generate_results:
                results = evaluate_models(
                    image_all_bands=image_all_bands, rbc_objects=rbc_objects,
                    time_index=time_index, image_index=image_idx,
                    date_string=date_string
                )
                predicted_class = {model: results[model]['base_class'] for model in results}
                posterior = {model: results[model]['posterior'] for model in results}
            else:
                predicted_class, posterior = load_results(image_idx=image_idx)

            # [APPENDIX] Plot one date results in Appendix figure
            if Visual.appendix_fig_settings['plot']:
                result_idx = Config.index_images_to_evaluate[Config.test_site].index(image_idx)
                appendix_figure.plot_results(
                    image_idx=image_idx, image_all_bands=image_all_bands,
                    base_model_predicted_class=predicted_class, date_string=date_string,
                    posterior=posterior, result_idx=result_idx
                )
                print('Appendix Results have been plotted for the current date')

            # [QUANTITATIVE ANALYSIS] Plot one date results in Classification figure
            print(f"plotting results for image with index {image_idx}")
            if Visual.class_fig_settings['plot'] and image_idx in Config.index_images_to_plot[Config.test_site]:
                result_idx = Config.index_images_to_plot[Config.test_site].index(image_idx)
                results_figure.process_and_plot_results(
                    image_idx=image_idx, image_all_bands=image_all_bands,
                    base_model_predicted_class=predicted_class,
                    posterior=posterior, result_idx=result_idx
                )
                print('Quantitative Analysis Results have been plotted for current date')
            time_index += 1
        date_string_list.append(date_string)

    finalize_evaluation_results(results_figure, appendix_figure)
    print('Evaluation Main is FINISHED')

def check_index_threshold(image: np.ndarray, rbc_objects: Dict[str, RBC], date: str) -> bool:
    """Checks if the spectral index classifier detects an abnormal percentage of deforested pixels.

    Parameters
    ----------
    image : np.ndarray
        Processed image data for cloud detection.
    rbc_objects : Dict[str, RBC]
        Dictionary of RBC objects for evaluation.
    date : str
        Date associated with the image.

    Returns
    -------
    bool
        True if the image passes the threshold check; False otherwise.
    """
    dim_x = Config.image_dimensions[Config.scenario]['dim_x']
    dim_y = Config.image_dimensions[Config.scenario]['dim_y']

    likelihood_sic, _, _, _ = rbc_objects["Scaled Index"].update_labels(image_all_bands=image, time_index=0)
    likelihood_sic_image = likelihood_sic.reshape(dim_x, dim_y)

    percentage_deforested = np.sum(likelihood_sic_image) / likelihood_sic_image.size

    if percentage_deforested > 0.5:
        logging.warning(f"Image on {date} discarded: {percentage_deforested * 100:.2f}% pixels classified as deforested.")
        return False

    return True

def load_results(image_idx: int):
    """Loads evaluation results from a pickle file."""
    pickle_file_path = os.path.join(
        Config.path_evaluation_results,
        'classification',
        f'{Config.scenario}_{Config.test_site}',
        'posteriors',
        f'{Config.scenario}_{Config.test_site}_image_{image_idx}_confID_{Config.conf_id}.pkl'
    )

    try:
        with open(pickle_file_path, 'rb') as file:
            predicted_class, posterior = pickle.load(file)
        return predicted_class, posterior
    except FileNotFoundError:
        logging.error(f"Results file not found: {pickle_file_path}")
        return None, None

def finalize_evaluation_results(results_figure, appendix_figure):
    """
    Finalize and save evaluation results, including sensitivity analysis, classification maps,
    quantitative analysis, and boxplots.

    Parameters
    ----------
    results_figure : ClassificationFigure
        Object containing results and methods for classification figures.
    appendix_figure : AppendixFigure
        Object containing results and methods for appendix figures.

    """

    path_results = os.path.join(Config.path_evaluation_results, "classification")

    # [SENSITIVITY ANALYSIS]
    # Sensitivity Analysis Results (Store Results and Plot/Save Figure with Results)
    # Figure 15 from [1]
    if Debug.store_pickle_sensitivity_analysis and Config.test_site in ['1a', '3']:
        logging.debug("Saving sensitivity analysis results and plotting figures.")
        path_save_pickle = os.path.join(
            Config.path_evaluation_results, "sensitivity_analysis",
            f"{Config.scenario}", 'results', f"eps_{Config.eps_SIC[Config.test_site]}"
        )
        # - Store pickle with results for the current configuration
        with open(path_save_pickle, 'wb') as file:
            pickle.dump(results_figure.results_qa, file)

        # - Plot (and save) sensitivity analysis figure for this configuration
        path_save_figure = os.path.join(
            Config.path_evaluation_results, "sensitivity_analysis",
            f"{Config.scenario}", "sensitivity_analysis.svg"
        )
        results_path = os.path.join(
            Config.path_evaluation_results, "sensitivity_analysis",
            f"{Config.scenario}", 'results'
        )
        plot_results_sensitivity_analysis(save_path=path_save_figure, Config=Config, results_path=results_path)

    # [CLASSIFICATION MAPS/QUALITATIVE ANALYSIS]
    # Appendix/Supplemental Material (Save Figure with Results for all Evaluation Dates/Images)
    # Supplemental Material from [1]
    settings = Visual.appendix_fig_settings[Config.test_site]
    appendix_figure.f.subplots_adjust(
        wspace=settings['wspace'], hspace=settings['hspace'], top=settings['top'],
        right=settings['right'], left=settings['left'], bottom=settings['bottom']
    )
    if Visual.appendix_fig_settings['save']:
        logging.debug("Saving appendix classification maps.")
        path_save_fig = os.path.join(
            Config.path_evaluation_results, "classification", f"{Config.scenario}_{Config.test_site}",
            f"figures", f"fig_config_{Config.conf_id}_appendix.svg"
        )
        appendix_figure.f.savefig(path_save_fig, bbox_inches='tight', format='svg', dpi=1000)

    # [QUANTITATIVE ANALYSIS (with labels)]
    # Classification Results Figure (Save Figure with Classification and Error Maps, and Classification Accuracy Results)
    # Figures 10-13 from [1]
    results_figure.adjust_figure()
    if Visual.class_fig_settings['save']:
        logging.debug("Saving classification results and error maps.")
        path_save_fig = os.path.join(
            Config.path_evaluation_results, "classification", f"{Config.scenario}_{Config.test_site}",
            f"figures", f"fig_config_{Config.conf_id}_quantitative.svg"
        )
        results_figure.f.savefig(path_save_fig, bbox_inches='tight', format='svg', dpi=1000)

    # Save Quantitative Analysis Results
    if Config.qa_settings['save']:
        logging.debug("Saving quantitative analysis metrics and models.")
        path_results_metrics = os.path.join(
            path_results, f'{Config.scenario}_{Config.test_site}', 'accuracy', f'conf_{Config.conf_id}'
        )
        os.makedirs(path_results_metrics, exist_ok=True)

        # Save pickle with models that have been evaluated for this configuration
        pickle.dump(
            results_figure.plot_legend[:-1],
            open(os.path.join(path_results_metrics, "models.pkl"), "wb")
        )

        # Save classification accuracy results
        for acc in Config.qa_settings['metrics']:
            path_i = os.path.join(path_results_metrics, f'{acc}.pkl')
            with open(path_i, "wb") as file:
                pickle.dump(results_figure.results_qa[acc], file)

    # [QUANTITATIVE ANALYSIS (BOXPLOT)]
    # Generate (and save) boxplot
    # Figure 14 from [1]
    if Visual.qa_fig_settings['plot']:
        logging.debug("Generating and saving boxplot for quantitative analysis.")
        plot_qa_boxplot(Config=Config)

    logging.info("Evaluation results finalized and saved.")

