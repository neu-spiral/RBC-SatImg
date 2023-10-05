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
    # --------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------
    plot_gmm_components = True  # True if wanting to plot the GMM components on top of index to check if number of
    # components for training GMM is the proper one
    plot_labels_training_images = False
    plot_cloud_detection_training = False

    # --------------------------------------------------------------------
    # Evaluation Parameters
    # --------------------------------------------------------------------
    plot_cloud_detection_evaluation = False # True if wanting to plot cloud and cloud shadow masks, together with
    pickle_sensitivity = False  # True if wanting to store results for sensitivity analysis
    pickle_histogram = False  # True if wanting to store results for histogram analysis
    save_figures = True  # True if wanting to save figures with results when using scripts in the folder `plot_results`
    check_dates = False  # If True, we do not evaluate, just check the dates of evaluated images (multiearth)

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
    # --------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------
    path_zenodo = r"/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset" \
                  r"/-54.60_-4.05"  # contains the folder downloaded from Zenodo
    path_evaluation_results = os.path.join(path_zenodo, "evaluation_results")
    path_sentinel_images = os.path.join(path_zenodo, "Sentinel2_data")
    path_results_figures = os.path.join(path_zenodo, 'results_figures')
    path_watnet_pretrained_model = os.path.join(os.getcwd(), r"benchmark_models/watnet/model/pretrained/watnet.h5")
    path_log_files = os.path.join(path_zenodo, 'log')
    path_trained_models = os.path.join(os.getcwd(), "trained_models")
    # TODO: Download deepwatermap checkpoints file and store in the path 'path_checkpoints_deepwatermap'
    path_checkpoints_deepwatermap = r"/Users/helena/Documents/checkpoints_deepwatermap/cp.135.ckpt"  # CHANGE

    # --------------------------------------------------------------------
    # Image Processing Parameters
    # --------------------------------------------------------------------

    # Spectral Bands
    # The main selected bands are the following
    bands_to_read = ['2', '3', '4', '8A', '8', '11']  # band identifiers must be a string
    # In the paper associated to the 'watnet' algorithm the authors mention the selection of the following bands
    bands_watnet = [3, 4, 5, 6, 1, 2]
    # In the paper associated to the 'deepwaternet' algorithm the authors mention the selection of the following bands
    bands_deepwaternet = [3, 4, 5, 6, 1, 2]

    # Image scaling factor
    scaling_factor_sentinel = 1e-4  # Sentinel-2 image processing, used when reading a Sentinel-2 image
    scaling_factor_watnet = 1

    # Histogram shifting factor
    shifting_factor_available = True # If True, a previously computed offset is applied to pixels (required for MultiEarth experiment)
    # this should be set to False when running the image scaling tool
    # Coordinates area free of clouds/cloud shadows
    coord_scale_x = [70, 150]
    coord_scale_y = [0, 70]

    # --------------------------------------------------------------------
    # Scene-dependent ParametersA
    # --------------------------------------------------------------------

    # Scenario and scene selection
    scenario = "multiearth"  # charles_river | oroville_dam
    scene_id = 4  # 0 | 1 | 2 | 3
    #   - scene_id = 0 if wanting to process the whole image (DEPRECATED)
    #   - scene_id = 1 for scene A in Oroville Dam (water stream)
    #   - scene_id = 2 for scene B in Oroville Dam (peninsula)
    #   - scene_id = 3 for scene C (Charles River)
    #   - scene_id = 4 for MULTIEARTH DATASET

    # Classes to evaluate
    classes = {'charles_river': ['water', 'land', 'vegetation'], 'oroville_dam': ['water', 'no water'],
               'multiearth': ['forest', 'no forest']}
    # Prior probabilities
    class_marginal_probabilities = {'charles_river': [0.33, 0.33, 0.33], 'oroville_dam': [0.5, 0.5], 'multiearth': [0.5, 0.5]}
    # Coordinates of the pixels to evaluate depending on the selected scene
    pixel_coords_to_evaluate = {1: {'x_coords': [1700, 1900], 'y_coords': [500, 1000]},
                                2: {'x_coords': [1400, 1550], 'y_coords': [1540, 1650]},
                                3: {'x_coords': [0, 700], 'y_coords': [800, 2041]},
                                4: {'x_coords': [0, 256], 'y_coords': [0, 256]}
                                }
    # Spectral Bands used to calculate the Spectral Indices
    # For NDVI, NIR (8) and Red (4) bands are used. Used for 'Charles River' scenario by the authors.
    # For NDWI, Green (3) and NIR (8) bands are used
    # For MNDWI, Green (3) and SWIR (11) bands are used. Used for 'Oroville Dam' scenario by the authors.
    bands_spectral_index = {'charles_river': ['8', '4'],
                            'oroville_dam': ['3', '11'],
                            'multiearth': ['3', '8']}  # band identifiers must be a string
    # Type of the images to be read
    image_types = {'charles_river': 'TCG', 'oroville_dam': 'SFJ', 'multiearth': ''}
    #   - 'charles_river': TCH|TCG
    #   - 'oroville_dam': SFJ|TFK
    # Pixel coordinates of the images to read (dimensions)
    image_dimensions = {'charles_river': {'dim_x': 927, 'dim_y': 2041}, 'oroville_dam': {'dim_x': 2229, 'dim_y': 3341},
                        'multiearth': {'dim_x': 256, 'dim_y': 256}}

    # file extension of images to be read
    file_extension = {'charles_river': '.tif', 'oroville_dam': '.tif', 'multiearth': '.tiff'}

    # --------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------
    gmm_dump_pickle = False  # False if wanting to use a stored pretrained model for GMM
    # True if wanting to train the GMM model
    trained_lr_model_pickle = False  # False if wanting to use a stored pretrained model for LR
    # True if wanting to train the LR model

    # Set plot_training_label_images to True if wanting to plot the training labels
    apply_cloud_mask_training = False
    cloud_threshold_evaluation = 0.25  # maximum cloud percentage so that an image is considered for evaluation
    cloud_filtering = {'ci_2_bands': ['2', '3', '4', '11'], 't_2': 0.2, 'csi_bands': ['8', '11'], 't_3': 0.6,
                       'ndvi_bands': ['8', '4'],
                       't_ndvi': 0.5}

    # Training data cropping: The amount of pixels used for training, which belong to the training regions described
    # in the manuscript, can be cropped to speed up code execution
    training_data_crop_ratio = {'charles_river': 0.7, 'oroville_dam': 0.5, 'multiearth': 1}

    # Model Selection (GMM)
    # These values have been set after data inspection
    # The thresholds are used to compute the scaled index model mean and std values, and also for labeling
    gmm_num_components = {'charles_river': [5, 3, 3],
                          'oroville_dam': [5, 3],
                          'multiearth': [10, 10]}
    threshold_index_labels = {'charles_river': [-0.01, 0.35],
                              'oroville_dam': [0.13],
                              #'multiearth': [-0.67]}
                              'multiearth': [-0.65]}

    # --------------------------------------------------------------------
    # Evaluation Parameters
    # --------------------------------------------------------------------
    # Configuration Options
    evaluation_generate_results = True  # False if wanting to skip evaluation
    # If evaluation is skipped, previously stored evaluation plot_results are plotted
    evaluation_store_results = False  # True if wanting to store evaluation results for
    # further visualization

    # Offset of evaluation images to consider in the evaluation stage: For the manuscript we have skipped the first
    # date because the image did not provide a good initialization for the recursive framework
    offset_eval_images = {'oroville_dam': 1, 'charles_river': 1, 'multiearth': 0}

    # Index of images to be plotted
    # If wanting to plot all dates:
    index_images_to_evaluate = {2: [*range(offset_eval_images[scenario], 42, 1)],
                            1: [*range(offset_eval_images[scenario], 42, 1)],
                            3: [*range(offset_eval_images[scenario], 28, 1)],
                            #4: [0, 2, 3, 5, 7, 9, 11, 21, 49, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78,
                               # 81, 82, 83, 89, 90, 101, 103, 108, 130, 134, 135, 140, 142, 144, 145, 146, 147,
                                #148, 150, 161, 175] # original vector from cloud filter
                            #4: [0, 2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                #82, 90, 101, 103, 108, 134, 145, 146,
                                #150, 161, 175]}
                            4: [2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                82, 90, 101, 103, 108, 134, 145, 146,
                                150, 161, 175] # plot all dates
                                }

    index_images_to_plot = {2: [*range(offset_eval_images[scenario], 42, 1)],
                            1: [*range(offset_eval_images[scenario], 42, 1)],
                            3: [*range(offset_eval_images[scenario], 28, 1)],
                            #4: [0, 2, 3, 5, 7, 9, 11, 21, 49, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78,
                               # 81, 82, 83, 89, 90, 101, 103, 108, 130, 134, 135, 140, 142, 144, 145, 146, 147,
                                #148, 150, 161, 175] # original vector from cloud filter
                            #4: [0, 2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                #82, 90, 101, 103, 108, 134, 145, 146,
                                #150, 161, 175]}
                             4: [5, 66, 76, 134, 150]   # plot dates for qualitative analysis experiment 3
                               }
    #index_images_to_evaluate = index_images_to_plot
    index_images_to_plot=index_images_to_evaluate

    # If wanting to reproduce Figure 5:
    # Config.index_images_to_evaluate = {1: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 2: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 3: [0, 4, 8, 12, 16, 20, 24]}

    # If wanting to store evaluation results obtained with all dates (consider the previous important note regarding big file sizes):
    index_images_to_store = {2: [*range(offset_eval_images[scenario], 42, 1)],
                             1: [*range(offset_eval_images[scenario], 42, 1)],
                             3: [*range(offset_eval_images[scenario], 28, 1)],
                             4: [*range(0, 217)]}

    # If wanting to store only evaluation results to reproduce Figure 5 (these results are already provided in the Zenodo folder)
    # Config.index_images_to_store = {1: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 2: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40], 3: [0, 4, 8, 12, 16, 20, 24]}

    # Transition Matrix
    # The transition matrix depends on the transition probability constant, denoted as epsilon
    eps = 0.01
    eps_GMM = 0.05
    eps_LR = 0.05
    transition_matrix = {'oroville_dam': np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2),
                         'multiearth': np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2),
                         'charles_river': np.array(
                             [1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps, eps / 2, eps / 2, eps / 2, 1 - eps]).reshape(
                             3, 3)}

    # Set the following parameter to a value different from 0 if wanting to read a specific number of images
    # different from the ones available
    num_evaluation_images_hardcoded = 0

    # Normalization constant to make NN classifier less confident
    # Normalization constant to make NN classifier less confident
    # Equation (10) from manuscript
    norm_constant = 0.1
    norm_constant_GMM = 0.2
    norm_constant_LR = 0.2

    filter_evaluation_images = False # True if wanting to check the images passing the cloud detection and SIC
    # classifier thresholds. If False, images used for evaluation will be the ones specified in
    # Config.index_images_to_evaluate

    evaluation_check_index_threshold = False

    # --------------------------------------------------------------------
    # Quantitative Analysis
    # --------------------------------------------------------------------
    index_quant_analysis = {5: 0, 66:1, 76:2, 134:3, 150:4} # image_idx : label_idx
    conduct_quantitative_analysis = False

    # --------------------------------------------------------------------
    # Plotting Parameters
    # --------------------------------------------------------------------

    # Results Visualization Options
    # cmap for mapping classes with colors
    cmap = {'oroville_dam': ['yellow', '#440154'], 'multiearth': ['#440154', 'yellow'],
            'charles_river': ['#440154', 'yellow', 'green']}

    # RGB enhance constant
    scaling_rgb = {2: 2.2, 1: 3.2, 3: 12, 4: 1}
