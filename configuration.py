import numpy as np
import logging
import os

from datetime import datetime


class Debug:
    """ Set the pickle related parameters to True if wanting to train again and store the generated
    data in pickle files for future code executions.
    If the data has already been stored in pickle files, it can be read and used by setting the pickle
    related parameters to False.

    """

    # Debugging options
    save_figures = True  # True if wanting to save figures with results when using scripts in the folder `plot_results`
    pickle_sensitivity = False  # True if wanting to perform sensitivity analysis
    pickle_histogram = False  # True if wanting to store results for histogram analysis
    check_dates = False  # If True, we do not evaluate, just check the dates of evaluated images (debugging)
    

    @staticmethod
    def set_logging_file(time_now: datetime):
        """ Creates a log file for the current code execution.


        force = True in logging.basicConfig() because otherwise the file path is not updated (see [https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm])

        Parameters
        ----------
        time_now : datetime
            time at which the log file is created (year, month, day, hours of the day, etc.)

        """
        # Get log file path
        file_name = time_now.strftime("%Y%m%d_%I%M%S.log")
        file_path = os.path.join(Config.path_log_files, file_name)

        # Set log file basic configuration
        logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', force=True)

        # Write first commands in the log file
        logging.debug(f"Log file has been created at {time_now}")
        logging.debug(f"Selected scenario: {Config.scenario}")


class Config:
    """ Some configuration settings must be changed when executing the code with data that is
    different to the one provided by the authors.

    """

    # Offset of evaluation images to consider in the evaluation stage: For the manuscript we have skipped the first
    # date because the image did not provide a good initialization for the recursive framework
    offset_eval_images = 1

    # Configuration Options
    # TODO: Change as desired
    evaluation_generate_results = True  # False if wanting to skip evaluation
    # If evaluation is skipped, previously stored evaluation plot_results are plotted
    evaluation_store_results = True

    # Index of images to be plotted
    # If wanting to plot all dates:
    index_images_to_plot = {2: [*range(offset_eval_images, 42, 1)],
                                   1: [*range(offset_eval_images, 42, 1)],
                                   3: [*range(offset_eval_images, 28, 1)]}
    # If wanting to reproduce Figure 5:
    # Config.index_images_to_plot = {1: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 2: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 3: [0, 4, 8, 12, 16, 20, 24]}

    evaluation_store_results = True
    # If wanting to store results obtained with all dates (consider the previous important note regarding big file sizes):
    index_images_to_store = {2: [*range(offset_eval_images, 42, 1)],
                                    1: [*range(offset_eval_images, 42, 1)],
                                    3: [*range(offset_eval_images, 28, 1)]}
    # If wanting to store only results to reproduce Figure 5 (these results are already provided in the Zenodo folder)
    # Config.index_images_to_store = {1: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 2: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 3: [0, 4, 8, 12, 16, 20, 24]}

    gmm_dump_pickle = False  # False if wanting to use a stored pretrained model for GMM
    # True if wanting to train the GMM model
    trained_lr_model_pickle = False  # False if wanting to use a stored pretrained model for LR
    # True if wanting to train the LR model

    # Paths
    # TODO: Download folder from Zenodo and store it in the path 'path_zenodo'
    path_zenodo = r"/Users/helena/Documents/Zenodo_RBC"  # TODO: CHANGE
    path_evaluation_results = os.path.join(path_zenodo, "evaluation_results")
    path_sentinel_images = os.path.join(path_zenodo, "Sentinel2_data")
    path_results_figures = os.path.join(path_zenodo, 'results_figures')
    path_watnet_pretrained_model = os.path.join(os.getcwd(), r"benchmark_models/watnet/model/pretrained/watnet.h5")
    path_log_files = os.path.join(path_zenodo, 'log')
    path_trained_models = os.path.join(os.getcwd(), "trained_models")
    # TODO: Download deepwatermap checkpoints file and store in the path 'path_checkpoints_deepwatermap'
    path_checkpoints_deepwatermap = r"/Users/helena/Documents/checkpoints_deepwatermap/cp.135.ckpt"  # CHANGE

    # Scenario and scene selection
    scenario = "charles_river"  # charles_river | oroville_dam
    scene_id = 3  # 0 | 1 | 2 | 3
    #   - scene_id = 0 if wanting to process the whole image (DEPRECATED)
    #   - scene_id = 1 for scene A in Oroville Dam (water stream)
    #   - scene_id = 2 for scene B in Oroville Dam (peninsula)
    #   - scene_id = 3 for scene C (Charles River)

    # Scene-dependent parameters
    # Classes to evaluate
    classes = {'charles_river': ['water', 'land', 'vegetation'], 'oroville_dam': ['water', 'no water']}
    # Prior probabilities
    prior_probabilities = {'charles_river': [0.33, 0.33, 0.33], 'oroville_dam': [0.5, 0.5]}
    # Coordinates of the pixels to evaluate depending on the selected scene
    pixel_coords_to_evaluate = {1: {'x_coords': [1700, 1900], 'y_coords': [500, 1000]},
                                2: {'x_coords': [1400, 1550], 'y_coords': [1540, 1650]},
                                3: {'x_coords': [0, 700], 'y_coords': [800, 2041]}}
    # Spectral Bands used to calculate the Spectral Indices
    # For NDVI, NIR (8) and Red (4) bands are used. Used for 'Charles River' scenario by the authors.
    # For NDWI, Green (3) and NIR (8) bands are used
    # For MNDWI, Green (3) and SWIR (11) bands are used. Used for 'Oroville Dam' scenario by the authors.
    bands_spectral_index = {'charles_river': ['8', '4'],
                            'oroville_dam': ['3', '11']}  # band identifiers must be a string
    # Type of the images to be read
    image_types = {'charles_river': 'TCG', 'oroville_dam': 'SFJ'}
    #   - 'charles_river': TCH|TCG
    #   - 'oroville_dam': SFJ|TFK
    # Pixel coordinates of the images to read (dimensions)
    image_dimensions = {'charles_river': {'dim_x': 927, 'dim_y': 2041}, 'oroville_dam': {'dim_x': 2229, 'dim_y': 3341}}

    # Image scaling factor
    scaling_factor_sentinel = 1e-4  # Sentinel-2 image processing, used when reading a Sentinel-2 image
    scaling_factor_watnet = 1

    # Training data cropping: The amount of pixels used for training, which belong to the training regions described
    # in the manuscript, can be cropped to speed up code execution
    training_data_crop_ratio = {'charles_river': 0.7, 'oroville_dam': 0.5}

    # Model Selection (GMM)
    # These values have been set after data inspection
    # The thresholds are used to compute the scaled index model mean and std values, and also for labeling
    gm_model_selection = {'charles_river': {'num_components': [5, 3, 3], 'thresholds': [-0.01, 0.35]},
                          'oroville_dam': {'num_components': [5, 3], 'thresholds': [0.13]}}

    # Normalization constant to make classifier less confident
    # Equation (10) from manuscript
    norm_constant = 0.3

    # Spectral Bands
    # The main selected bands are the following
    bands_to_read = ['2', '3', '4', '8A', '8', '11']  # band identifiers must be a string
    # In the paper associated to the 'watnet' algorithm the authors mention the selection of the following bands
    bands_watnet = [3, 4, 5, 6, 1, 2]
    # In the paper associated to the 'deepwaternet' algorithm the authors mention the selection of the following bands
    bands_deepwaternet = [3, 4, 5, 6, 1, 2]

    # Transition Matrix
    # The transition matrix depends on the transition probability constant, denoted as epsilon
    eps = 0.05
    transition_matrix = {'oroville_dam': np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2),
                         'charles_river': np.array(
                             [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
                             3, 3)}

    # Set the following parameter to a value different from 0 if wanting to read a specific number of images
    # different from the ones available
    num_evaluation_images_hardcoded = 0

    # Results Visualization Options
    # cmap for mapping classes with colors
    cmap = {'oroville_dam': ['yellow', '#440154'], 'charles_river': ['#440154', 'yellow', 'green']}
    # RGB enhance constant
    scaling_rgb = {2: 2.2, 1: 3.2, 3: 12}

