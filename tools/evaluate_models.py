import os
import pickle
import logging
import numpy as np
from typing import Dict
from configuration import Config, Debug
from bayesian_recursive import RBC


def calculate_transition_matrix(model: str, test_site: str, scenario: str, eps_key: str) -> np.ndarray:
    """Calculates the transition matrix for a given model based on scenario and epsilon values."""
    eps = getattr(Config, eps_key)[test_site]
    adaptive_eps = getattr(Config, f"{eps_key}_adaptive", {}).get(test_site, None)

    if scenario == "charles_river_3classes":
        return np.array([
            1 - eps, eps / 2, eps / 2,
            eps / 2, 1 - eps, eps / 2,
            eps / 2, eps / 2, 1 - eps
        ]).reshape(3, 3)
    elif adaptive_eps is not None:
        return np.array([1 - adaptive_eps, adaptive_eps, adaptive_eps, 1 - adaptive_eps]).reshape(2, 2)
    else:
        return np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)


def evaluate_single_model(model_name: str, rbc_objects: Dict[str, RBC], image_all_bands: np.ndarray,
                          time_index: int, image_index: int, dim_x: int, dim_y: int, eps_key: str):
    """Evaluates a single model and returns its results."""
    transition_matrix = calculate_transition_matrix(model_name, Config.test_site, Config.scenario, eps_key)
    rbc_objects[model_name].transition_matrix = transition_matrix

    base_class, posterior, pred_float, post_probs = rbc_objects[model_name].update_labels(
        image_all_bands=image_all_bands, time_index=time_index, image_idx=image_index
    )

    return {
        "base_class": base_class.reshape(dim_x, dim_y),
        "posterior": posterior.reshape(dim_x, dim_y),
        "pred_float": pred_float,
        "post_probs": post_probs
    }


def evaluate_models(image_all_bands: np.ndarray, rbc_objects: Dict[str, RBC], time_index: int, image_index: int,
                    date_string: str):
    """
    Returns the prior/likelihood and posterior probabilities for each evaluated model.

    Parameters
    ----------
    image_all_bands : np.ndarray
        Array containing the bands of the image under evaluation.
    rbc_objects : Dict[str, RBC]
        Dictionary containing one RBC object per model to evaluate.
    time_index : int
        Index of the current time instant of the Bayesian process.
    image_index : int
        Index of the current image used for evaluation (associated with one date).
    date_string : str
        Contains the date of the read image.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Results containing base class, posterior, predicted float, and posterior probabilities for each model.
    """
    logging.debug(f"Evaluation results are being generated for image {image_index}")

    # Dimensions of the image
    dim_x = Config.image_dimensions[Config.scenario]['dim_x']
    dim_y = Config.image_dimensions[Config.scenario]['dim_y']

    models = {
        "GMM": "eps_GMM",
        "Scaled Index": "eps_SIC",
        "Logistic Regression": "eps_LR",
        "WatNet": "eps_WN",
        "DeepWaterMap": "eps_DWM"
    }

    results = {}

    # Evaluate each model
    for model_name, eps_key in models.items():
        if model_name in rbc_objects:
            results[model_name] = evaluate_single_model(
                model_name, rbc_objects, image_all_bands, time_index, image_index, dim_x, dim_y, eps_key
            )

    # Save results if required
    if image_index in Config.index_images_to_store[Config.test_site] and Config.evaluation_store_results:
        pickle_file_path = os.path.join(
            Config.path_evaluation_results, 'classification',
            f'{Config.scenario}_{Config.test_site}', 'posteriors',
            f'{Config.scenario}_{Config.test_site}_image_{image_index}_confID_{Config.conf_id}.pkl'
        )
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(results, file)

    # Save histogram for debugging if enabled
    if Debug.pickle_histogram:
        pickle_file_path = os.path.join(
            Config.path_histogram_prediction, f'{Config.scenario}_{Config.test_site}',
            f'{Config.scenario}_{Config.test_site}_image_{image_index}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl'
        )
        with open(pickle_file_path, 'wb') as file:
            pickle.dump({"predicted_probabilities": results, "date_string": date_string}, file)

    return results
