import logging
import os
import pickle

import numpy as np

from image_reader import ReadSentinel2
import matplotlib.pyplot as plt
from configuration import Config, Debug
from bayesian_recursive import RBC, get_rbc_objects
from plot_results.plot_results_classification import ClassificationResultsFigure
from typing import List
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from typing import Dict

from tools.spectral_index import get_broadband_index, get_labels_from_index
from tools.path_operations import get_num_images_in_folder
from tools.cloud_filtering import check_cloud_threshold


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
    # Get the relative path where the pickle file containing evaluation plot_results
    if Config.scenario == 'charles_river':
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.scene_id}',
                                        f'{Config.scenario}_{Config.scene_id}_image_{image_index}_epsilon_{Config.eps}.pkl')
    elif Config.scenario == 'multiearth':
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.scene_id}_image_{image_index}_epsilon_{Config.eps}.pkl')
    else:
        pickle_file_path = os.path.join(Config.path_evaluation_results, 'classification',
                                        f'{Config.scenario}_{Config.scene_id}',
                                        f'{Config.scenario}_{Config.scene_id}_image_{image_index}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl')

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
    base_model_predicted_class["GMM"], posterior["GMM"], prediction_float["GMM"], posterior_probabilities["GMM"] = \
    rbc_objects["GMM"].update_labels(
        image_all_bands=image_all_bands,
        time_index=time_index)
    posterior["GMM"] = posterior["GMM"].reshape(dim_x, dim_y)
    base_model_predicted_class["GMM"] = base_model_predicted_class["GMM"].reshape(dim_x, dim_y)
    # plt.figure(), plt.imshow(likelihood["GMM"])
    # plt.figure(), plt.imshow(posterior["GMM"])

    # Scaled Index Model
    base_model_predicted_class["Scaled Index"], posterior["Scaled Index"], prediction_float["Scaled Index"], \
    posterior_probabilities["Scaled Index"] = rbc_objects[
        "Scaled Index"].update_labels(
        image_all_bands=image_all_bands, time_index=time_index)
    posterior["Scaled Index"] = posterior["Scaled Index"].reshape(dim_x, dim_y)
    base_model_predicted_class["Scaled Index"] = base_model_predicted_class["Scaled Index"].reshape(dim_x, dim_y)
    # plt.figure(), plt.imshow(likelihood["Scaled Index"])
    # plt.figure(), plt.imshow(posterior["Scaled Index"])

    # Logistic Regression Model
    base_model_predicted_class["Logistic Regression"], posterior["Logistic Regression"], prediction_float[
        "Logistic Regression"], posterior_probabilities["Logistic Regression"] = \
        rbc_objects[
            "Logistic Regression"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index)
    posterior["Logistic Regression"] = posterior["Logistic Regression"].reshape(dim_x, dim_y)
    base_model_predicted_class["Logistic Regression"] = base_model_predicted_class["Logistic Regression"].reshape(dim_x,
                                                                                                                  dim_y)
    # plt.figure(), plt.imshow(likelihood["Logistic Regression"])
    # plt.figure(), plt.imshow(posterior["Logistic Regression"])

    # Benchmark Deep Learning models for the water mapping experiment
    if Config.scenario == "oroville_dam":
        # DeepWaterMap Algorithm
        base_model_predicted_class["DeepWaterMap"], posterior["DeepWaterMap"], prediction_float["DeepWaterMap"], \
        posterior_probabilities["DeepWaterMap"] = rbc_objects[
            "DeepWaterMap"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index)
        posterior["DeepWaterMap"] = posterior["DeepWaterMap"].reshape(dim_x, dim_y)
        base_model_predicted_class["DeepWaterMap"] = base_model_predicted_class["DeepWaterMap"].reshape(dim_x, dim_y)

        # WatNet Algorithm
        base_model_predicted_class["WatNet"], posterior["WatNet"], prediction_float["WatNet"], posterior_probabilities[
            "WatNet"] = rbc_objects["WatNet"].update_labels(
            image_all_bands=image_all_bands, time_index=time_index)
        posterior["WatNet"] = posterior["WatNet"].reshape(dim_x, dim_y)
        base_model_predicted_class["WatNet"] = base_model_predicted_class["WatNet"].reshape(dim_x, dim_y)

    # Dump data into pickle if this image index belongs to the list containing indices of images to store
    if image_index in Config.index_images_to_store[Config.scene_id] and Config.evaluation_store_results:
        pickle.dump([base_model_predicted_class, posterior], open(pickle_file_path, 'wb'))

    # Dump data into pickle for analysis of the prediction histogram
    if Debug.pickle_histogram:
        pickle_file_path = os.path.join(Config.path_histogram_prediction, f'{Config.scenario}_{Config.scene_id}',
                                        f'{Config.scenario}_{Config.scene_id}_image_{image_index}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl')
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
    None (plots plot_results for the time indexes specified in the configuration file)

    """

    # Initialize one RBC object for each model to be compared
    rbc_objects = get_rbc_objects(gmm_densities=gmm_densities, trained_lr_model=trained_lr_model)

    # Read the images in the evaluation folder
    # Path where evaluation images are stored
    logging.debug("Start Evaluation")
    path_evaluation_images = os.path.join(Config.path_sentinel_images, 'evaluation')
    #path_evaluation_images = "/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset/sent2/-54.58_-3.43"

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

    # Initialize the time instant parameter *time_index*
    time_index = 0

    if Config.scenario == 'multiearth':
        path_label_images = os.path.join(Config.path_sentinel_images, 'deforestation_labels')
        # The classifier is applied to all the bands of all the images to be read
        # Each image is linked to one specific date
        # If specified in the configuration file, we check images that pass the cloud detection and index thresholds,
        # store their indices, and then we process them in the second loop
        if Config.filter_evaluation_images:
            Config.index_images_to_evaluate[Config.scene_id] = []
            for image_idx_loop in range(Config.offset_eval_images[Config.scenario], num_evaluation_images):
                # All bands of the image with index *image_idx* are stored in *image_all_bands*
                image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images,
                                                                       image_idx=image_idx_loop)
                if check_cloud_threshold(image=image_all_bands,
                                         image_date=date_string) and Config.evaluation_check_index_threshold:
                    print('****** ACCEPTED IMAGE with index ' + str(image_idx_loop))
                    # If image cloud percentage is not above the threshold, we consider this image for evaluation
                    # ***** Second check: index threshold
                    if check_index_threshold(image=image_all_bands, rbc_objects=rbc_objects, date=date_string):
                        Config.index_images_to_evaluate[Config.scene_id].append(image_idx_loop)
            # num_evaluation_images = len(Config.index_images_to_evaluate[Config.scene_id])
        Config.index_images_to_store[Config.scene_id] = Config.index_images_to_evaluate[Config.scene_id]

        # Evaluation Loop
        for image_idx in Config.index_images_to_evaluate[Config.scene_id]:
            # All bands of the image with index *image_idx* are stored in *image_all_bands*
            image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images, image_idx=image_idx, shift_option="evaluation")

            if not Debug.check_dates:
                # TODO: check what index and labels are used for
                # Calculate and add the spectral index for all bands
                index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])
                image_all_bands = np.hstack([image_all_bands, index.reshape(-1, 1)])

                # Get labels from the spectral index values
                labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]),
                                               threshold=Config.threshold_index_labels[Config.scenario])

                # Evaluate the 3 models for one date
                if Config.evaluation_generate_results:
                    # If results need to be generated again, we need to evaluate considering all the dates, regardless of
                    # how many dates we want to plot
                    predicted_class, posterior, predicted_probabilities, posterior_probabilities = evaluate_models(
                        image_all_bands=image_all_bands, rbc_objects=rbc_objects,
                        time_index=time_index, image_index=image_idx,
                        date_string=date_string)
                else:
                    print(
                        'No evaluation results are generated - Please make sure sure results have been previously generated')
                    path_results = os.path.join(Config.path_evaluation_results, "classification")
                    pickle_file_path = os.path.join(path_results,
                                                    f"{Config.scenario}_{Config.scene_id}_image_{image_idx}_epsilon_{Config.eps}.pkl")
                    [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))

                # Plot results for this image
                print(f"plotting results for image with index {image_idx}")
                if image_idx in Config.index_images_to_plot[Config.scene_id]:
                    # User wants to plot this image
                    if image_idx == Config.index_images_to_plot[Config.scene_id][0]:
                        # This is the first image to be plotted
                        # Figure is created
                        results_figure = ClassificationResultsFigure()
                        results_figure.create_figure()
                    result_idx = Config.index_images_to_plot[Config.scene_id].index(image_idx)
                    """
                    results_figure.plot_results_one_date(image_idx=image_idx, image_all_bands=image_all_bands,
                                                         date_string=date_string, base_model_predicted_class=predicted_class,
                                                         posterior=posterior, result_idx=result_idx, base_model_predicted_probabilities=predicted_probabilities)
                    """
                    results_figure.plot_results_one_date(image_idx=image_idx, image_all_bands=image_all_bands,
                                                         date_string=date_string,
                                                         base_model_predicted_class=predicted_class,
                                                         posterior=posterior, result_idx=result_idx)
                    # **************************************************************
                    # ********************** QUANTITATIVE ANALYSIS STARTS
                    # **************************************************************

                    if image_idx in Config.index_quant_analysis and Config.conduct_quantitative_analysis:
                        if image_idx == list(Config.index_quant_analysis.keys())[0]:
                            quantitative_results_figure = ClassificationResultsFigure()
                            quantitative_results_figure.create_quantitative_results_figure()
                            quantitative_results_plotted = 0
                            quantitative_results_output = dict()
                        label_idx = Config.index_quant_analysis[image_idx] # index of the label we want to use for comparison
                        #label_image, date_string_label = image_reader.read_image(path=path_label_images, image_idx=label_idx)
                        for file_counter, file_name in enumerate(sorted(os.listdir(path_label_images))):
                            for col in range(0,13):
                                for axis in ['top', 'bottom', 'left', 'right']:
                                    quantitative_results_figure.axarr[quantitative_results_plotted, col].spines[axis].set_linewidth(0)
                                quantitative_results_figure.axarr[quantitative_results_plotted, col].get_yaxis().set_ticks([])
                                quantitative_results_figure.axarr[quantitative_results_plotted, col].get_xaxis().set_ticks([])
                            if file_counter == label_idx:
                                quantitative_results_output[label_idx] = dict()
                                path_label_i = os.path.join(path_label_images, file_name)
                                label_image = image_reader.read_band(path_label_i)
                                label_date_string = file_name[-15:-5]
                                quantitative_results_figure.axarr[quantitative_results_plotted, 0].imshow(label_image)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 0].title.set_text(f'L {label_date_string[2:]}')
                                # SIC
                                quantitative_results_figure.axarr[quantitative_results_plotted, 1].imshow(predicted_class['Scaled Index'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 1].title.set_text(
                                    f'SIC {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=predicted_class['Scaled Index'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 2].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 2].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                # RSIC
                                quantitative_results_figure.axarr[quantitative_results_plotted, 3].imshow(posterior['Scaled Index'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 3].title.set_text(
                                    f'RSIC {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=posterior['Scaled Index'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 4].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 4].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                quantitative_results_output[label_idx]['RSIC'] = class_acc
                                # GMM
                                quantitative_results_figure.axarr[quantitative_results_plotted, 5].imshow(predicted_class['GMM'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 5].title.set_text(
                                    f'GMM {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=predicted_class['GMM'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 6].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 6].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                # RGMM
                                quantitative_results_figure.axarr[quantitative_results_plotted, 7].imshow(posterior['GMM'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 7].title.set_text(
                                    f'RGMM {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=posterior['GMM'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 8].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 8].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                quantitative_results_output[label_idx]['RGMM'] = class_acc
                                # LR
                                quantitative_results_figure.axarr[quantitative_results_plotted, 9].imshow(predicted_class['Logistic Regression'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 9].title.set_text(
                                    f'LR {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=predicted_class['Logistic Regression'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 10].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 10].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                # RLR
                                quantitative_results_figure.axarr[quantitative_results_plotted, 11].imshow(posterior['Logistic Regression'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 11].title.set_text(
                                    f'RLR {date_string[5:]}')
                                class_map, class_acc = quantitative_results_figure.get_quantitative_results(label_image=label_image, class_labels=posterior['Logistic Regression'])
                                quantitative_results_figure.axarr[quantitative_results_plotted, 12].imshow(class_map)
                                quantitative_results_figure.axarr[quantitative_results_plotted, 12].title.set_text(
                                    f'{np.round(class_acc*100,3)}%')
                                quantitative_results_output[label_idx]['RLR'] = class_acc
                                plt.tight_layout()
                        quantitative_results_plotted = quantitative_results_plotted + 1
                    # **************************************************************
                    # ********************** QUANTITATIVE ANALYSIS ENDS
                    # **************************************************************
                time_index = time_index + 1
        # Save figure as pdf
       #return quantitative_results_output
        if Debug.save_figures:
            results_figure.f.savefig(os.path.join(Config.path_results_figures,
                                                  f'july_192_supp.pdf'),
                                     format="pdf", bbox_inches="tight", dpi=1000)
            quantitative_results_figure.f.savefig(os.path.join(Config.path_results_figures,
                                                  f'quant_results_{Config.eps}.pdf'),
                                     format="pdf", bbox_inches="tight", dpi=1000)
    elif Config.scenario != 'multiearth':
        # The classifier is applied to all the bands of all the images to be read
        # Each image is linked to one specific date
        for image_idx in range(Config.offset_eval_images, num_evaluation_images):
            # All bands of the image with index *image_idx* are stored in *image_all_bands*
            image_all_bands, date_string = image_reader.read_image(path=path_evaluation_images, image_idx=image_idx)

            if not Debug.check_dates:
                # Calculate and add the spectral index for all bands
                index = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[Config.scenario])
                image_all_bands = np.hstack([image_all_bands, index.reshape(-1, 1)])

                # Get labels from the spectral index values
                labels = get_labels_from_index(index=index, num_classes=len(Config.classes[Config.scenario]))

                # Evaluate the 3 models for one date
                if Config.evaluation_generate_results:
                    # If results need to be generated again, we need to evaluate considering all the dates, regardless of
                    # how many dates we want to plot
                    likelihood, posterior = evaluate_models(image_all_bands=image_all_bands, rbc_objects=rbc_objects,
                                                            time_index=time_index, image_index=image_idx,
                                                            date_string=date_string)
                else:
                    print('No evaluation results are generated')

                # Plot results for this image
                print(f"plotting results for image with index {image_idx}")
                if image_idx in Config.index_images_to_evaluate[Config.scene_id]:
                    # User wants to plot this image
                    if image_idx == Config.index_images_to_evaluate[Config.scene_id][0]:
                        # This is the first image to be plotted
                        # Figure is created
                        results_figure = ClassificationResultsFigure()
                        results_figure.create_figure()
                    result_idx = Config.index_images_to_evaluate[Config.scene_id].index(image_idx)
                    results_figure.plot_results_one_date(image_idx=image_idx, image_all_bands=image_all_bands,
                                                         date_string=date_string, likelihood=likelihood,
                                                         posterior=posterior, result_idx=result_idx)
                time_index = time_index + 1
        # Save figure as pdf
        if Debug.save_figures:
            results_figure.f.savefig(os.path.join(Config.path_results_figures,
                                                  f'classification_{Config.scenario}_{Config.scene_id}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pdf'),
                                     format="pdf", bbox_inches="tight", dpi=200)
    else:
        print('Error: incorrect scenario has been selected in the configuration file')
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
