import numpy as np
import logging
import os

from datetime import datetime


class Debug:
    """ Change the DEBUGGING settings as desired.

    """
    # --------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------
    plot_gmm_components = False  # True if wanting to plot the GMM components on top of the spectral index to check if
    # the number of components for training the GMM is appropriate
    plot_labels_training_images = False  # True if wanting to plot the images used for training together with their
    # corresponding labels (generated with spectral index results)

    # We use the cloud detection module to select the images to be considered for training and evaluation. After
    # this, we discard images accordingly by removing them from the folder or by changing 'index_images_to_plot' or
    # 'index_images_to_evaluate'. Meaning, cloud detection is not applied when running main.py in the current
    # implementation.
    plot_cloud_detection_training = False
    plot_cloud_detection_evaluation = False

    # --------------------------------------------------------------------
    # Evaluation Parameters - Sensitivity Analysis
    # --------------------------------------------------------------------
    store_pickle_sensitivity_analysis_water_mapping = False  # True if wanting to store results for sensitivity
    # analysis of the transition probability parameter in the water mapping experiments (Oroville Dam and Charles River)
    store_pickle_sensitivity_analysis = True  # True if wanting to store results for sensitivity
    # analysis of the transition probability parameter in the deforestation detection experiment (Amazon rainforest)
    pickle_histogram = False  # True if wanting to store results for histogram analysis
    check_dates = False  # If True, we do not evaluate, and just print the dates of the evaluation images

    # --------------------------------------------------------------------
    # Generate ground truth labels
    # --------------------------------------------------------------------
    # CODE TO GENERATE THE GROUND TRUTH LABELS HAS NOT BEEN ADDED IN THIS CODE

    # --------------------------------------------------------------------
    # Adaptive Transition Probability Parameter - [DEPRECATED FOR THIS IMPLEMENTATION]
    # --------------------------------------------------------------------
    update_table = False
    epsilon_adaptive = False
    adaptive_epsilon_debug = False
    calculate_SAM = False
    get_date_statistic = False
    # epsilon adaptive TIME
    epsilon_adaptive_time = False

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


class Visual:
    """ Some VISUALIZATION settings must be changed when executing the code with data that is
    different to the one provided by the authors.

    """

    #
    # General Settings
    cmap = {'charles_river': ['yellow', '#440154'],
            'oroville_dam': ['yellow', '#440154'],
            'multiearth': ['#440154', 'yellow'],
            'charles_river_3class': ['#440154', 'yellow', 'green'],
            'error_map': ['#D6D4D4', 'red'],
            'label': ['black', 'white']}
    scaling_rgb = {'1b': 2.2, '1a': 3.2, '2': 2.7, '3': 3}  # RGB enhance constant

    #
    # Classification figure (with accuracy metric, classification error maps and labels)
    class_fig_settings = {
        # Options
        'save': False,
        'plot': True,
        'fontsize': {'1a': 10, '1b': 15, '2': 13, '3': 15},
        'font_highest_value':
            {'fontweight': 'normal', 'fontcolor': 'black'},
        # The following parameters are used by the function
        # 'adjust_figure()' in classification_figure.py to scale the figure with classification results Test site 1a
        '1a': {'dist_aux': 0.052,  # space to read accuracy
               'height_image': 0.018,  # this needs to be changed when adjusting hspace
               'dist_separation': 0.005,  # nonrecursive-recursive-rgblabel separation
               'tuned_wspace': 0.015,
               'tuned_hspace': -0.9},
        # Test site 1b
        '1b': {'dist_aux': 0.14,
               'height_image': 0.025,
               'dist_separation': 0.001,
               'tuned_wspace': 0.015,
               'tuned_hspace': -0.7},
        # Test site 2
        '2': {'dist_aux': 0.073,
              'height_image': 0.025,
              'dist_separation': 0.005,
              'tuned_wspace': 0.015,
              'tuned_hspace': -0.985},
        # Test site 3
        '3': {'dist_aux': 0.15,
              'height_image': 0.02,
              'dist_separation': 0.005,
              'tuned_wspace': -0.09,
              'tuned_hspace': -0.2},
    }

    #
    # Appendix figure
    appendix_fig_settings = {
        # Options
        'save': False,
        'plot': True,
        # Test site 1a
        '1a': {'wspace': 0.05,
               'hspace': 0.1,
               'top': 1,
               'right': 0.9,
               'left': 0,
               'bottom': 0},
        # Test site 1b
        '1b': {'wspace': 0.05,
               'hspace': 0.1,
               'top': 1,
               'right': 0.9,
               'left': 0,
               'bottom': 0},
        # Test site 2
        '2': {'wspace': 0.08,
               'hspace': 0.1,
               'top': 1,
               'right': 1.4,
               'left': 0,
               'bottom': 0},
        # Test site 3
        '3': {'wspace': 0.1,
               'hspace': 0.1,
               'top': 1,
               'right': 0.8,
               'left': 0.15,
               'bottom': 0},
    }

    #
    # Quantitative Analysis (QA) figure (boxplot)
    water_mapping_models = ["SIC", 'GMM', "LR", 'DWM', 'WN', 'RSIC', 'RGMM', 'RLR', 'RDWM', 'RWN']
    deforestation_detection_models = ["SIC", 'GMM', "LR", 'RSIC', 'RGMM', 'RLR']
    qa_fig_settings = {
        # Options
        'save': False,
        'plot': False,
        # Legend models
        'legend_models': {'1a': water_mapping_models,
                          '1b': water_mapping_models,
                          '2': water_mapping_models,
                          '3': deforestation_detection_models
                          },
    }


class Config:
    """ Some CONFIGURATION settings must be changed when executing the code with data that is
    different to the one provided by the authors.

    """

    # TODO: Change conf ID as desired (directory and file names might be created using this ID)
    conf_id = "00"  # Running ts '2': eps 0.1, norm_constant 0.3 for DL and 0 for others
    if conf_id == "00":
        # has been processed with test site '2'
        # conf_id 03 is the same but with offset 2
        eps = {'1a': 0.05, '1b': 0.05, '2': 0.1, '3': 0.01}  # SIC and WN models
        eps_DWM = {'1a': 0.05, '1b': 0.05, '2': 0.1, '3': None}  # WN model
        eps_GMM = {'1a': 0.05, '1b': 0.05, '2': 0.1, '3': 0.005}  # GMM model
        eps_LR = {'1a': 0.05, '1b': 0.05, '2': 0.1, '3': 0.06}  # LR model
        norm_constant_DL = {'1a': 0.3, '1b': 0.3, '2': 0.3, '3': None}
        norm_constant_SIC = {'1a': 0, '1b': 0, '2': 0, '3': 0.1}
        norm_constant_GMM = {'1a': 0, '1b': 0, '2': 0, '3': 0.2}
        norm_constant_LR = {'1a': 0, '1b': 0, '2': 0, '3': 0.2}

    # --------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------
    # TODO: Download deepwatermap checkpoints file and store in 'path_checkpoints_deepwatermap'
    path_checkpoints_deepwatermap = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/checkpoints/cp.135.ckpt"
    # TODO: Download files from zenodo and store in 'path_zenodo'
    path_zenodo = r"/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/2023nov"  # contains the folder downloaded from Zenodo
    #
    path_evaluation_results = os.path.join(path_zenodo, "evaluation_results")
    path_sentinel_images = os.path.join(path_zenodo, "Sentinel2_data")
    path_results_figures = os.path.join(path_evaluation_results, 'figures')
    path_watnet_pretrained_model = os.path.join(os.getcwd(), r"benchmark_models/watnet/model/pretrained/watnet.h5")
    path_log_files = os.path.join(path_zenodo, 'log')
    path_trained_models = os.path.join(os.getcwd(), "trained_models")

    # --------------------------------------------------------------------
    # Image Processing Parameters
    # --------------------------------------------------------------------
    #
    # Spectral Bands
    # The main selected bands are the following
    bands_to_read = ['2', '3', '4', '8A', '8', '11']  # band identifiers must be a string
    # In the paper associated to the 'watnet' algorithm the authors mention the selection of the following bands
    bands_watnet = [3, 4, 5, 6, 1, 2]
    # In the paper associated to the 'deepwaternet' algorithm the authors mention the selection of the following bands
    bands_deepwaternet = [3, 4, 5, 6, 1, 2]

    #
    # Image scaling factor
    scaling_factor_sentinel = 1e-4  # used when reading a Sentinel-2 image in image_reader.py
    scaling_factor_watnet = 1  # used to pre-process Sentinel-2 images to input the DL algorithms (DWM and WN)

    #
    # Histogram shifting factor
    #
    # This is explained in Section 2.3 from the paper "Recursive classification of satellite imaging time-series: An
    # application to land cover mapping". Text: " a time-varying bias is fitted to each image, for which an area
    # where the statistics are expected to be time-invariant (i.e., no clouds or disturbances are observed inside
    # that area for all evaluation dates) is selected."
    shifting_factor_available = True  # If True, a previously calculated offset is applied to pixel magnitudes (this is
    # required for the MultiEarth experiment). This should be set to False when running the image scaling tool.
    #
    # Coordinates of the area free of clouds/cloud shadows: The following are used in image_scaling.py to apply the
    # time-varying bias to each image as discussed in the paper and in cloud_filtering.py to check the
    # percentage of detected cloud/cloud shadow
    coord_scale_x = [70, 150]
    coord_scale_y = [0, 70]

    # --------------------------------------------------------------------
    # Scene-dependent Parameters
    # --------------------------------------------------------------------
    #
    # TODO: Select test site to evaluate
    test_site = '3'  # '1a' | '1b' | '2' | '3'
    #   - test_site = '1a' for Oroville Dam (water stream)
    #   - test_site = '1b' for Oroville Dam (peninsula)
    #   - test_site = '2' for Charles River
    #   - test_site = '3' for MULTIEARTH DATASET
    if test_site in ['1a', '1b']:
        scenario = 'oroville_dam'
    elif test_site == '2':
        scenario = 'charles_river'
    else:
        scenario = 'multiearth'
    # Classes to evaluate
    classes = {'charles_river': ['water', 'no water'],
               'charles_river_3class': ['water', 'land', 'vegetation'],
               'oroville_dam': ['water', 'no water'],
               'multiearth': ['forest', 'no forest']}
    # Prior probabilities
    class_marginal_probabilities = {'charles_river_3class': [0.33, 0.33, 0.33], 'oroville_dam': [0.5, 0.5],
                                    'multiearth': [0.5, 0.5], 'charles_river': [0.5, 0.5], }
    # Coordinates of the pixels to evaluate depending on the selected scene
    pixel_coords_to_evaluate = {'1a': {'x_coords': [1700, 1900], 'y_coords': [500, 1000]},
                                '1b': {'x_coords': [1400, 1550], 'y_coords': [1540, 1650]},
                                '2': {'x_coords': [0, 700], 'y_coords': [800, 2041]},
                                '3': {'x_coords': [0, 256], 'y_coords': [0, 256]}
                                }
    # Spectral Bands used to calculate the Spectral Indices
    # For NDVI, NIR (8) and Red (4) bands are used. Used for 'Charles River' scenario by the authors.
    # For NDWI, Green (3) and NIR (8) bands are used
    # For MNDWI, Green (3) and SWIR (11) bands are used. Used for 'Oroville Dam' scenario by the authors.
    bands_spectral_index = {'charles_river_3class': ['8', '4'],
                            'oroville_dam': ['3', '11'],
                            'charles_river': ['3', '11'],
                            'multiearth': ['3', '8']}  # band identifiers must be a string
    # Type of the images to be read
    image_types = {'charles_river': 'TCG', 'oroville_dam': 'SFJ', 'multiearth': ''}
    #   - 'charles_river': TCH|TCG
    #   - 'oroville_dam': SFJ|TFK
    # Dimensions of the images to read (pixels)
    image_dimensions = {'charles_river': {'dim_x': 927, 'dim_y': 2041},
                        'oroville_dam': {'dim_x': 2229, 'dim_y': 3341},
                        'multiearth': {'dim_x': 256, 'dim_y': 256}}
    # file extension of images to be read
    file_extension = {'charles_river': '.tif',
                      'oroville_dam': '.tif',
                      'multiearth': '.tiff'}

    # --------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------

    #
    # TODO: Change the pickle-related variables so that (a) the GMM and LR models are trained or (b) previously
    #  trained GMM and LR models are used in the experiments
    gmm_dump_pickle = False  # False if wanting to use a stored pretrained model for GMM - True if wanting to train the
    # GMM model again
    trained_lr_model_pickle = False  # False if wanting to use a stored pretrained model for LR - True if wanting to
    # train the LR model again

    #
    # Cloud Filtering:
    apply_cloud_mask_training = False  # True if wanting to discard training image pixels with clouds/cloud shadows in
    # training_main from training.py
    cloud_threshold_evaluation = 0.25  # maximum cloud percentage so that an image is considered for evaluation
    cloud_filtering = {'ci_2_bands': ['2', '3', '4', '11'], 't_2': 0.2, 'csi_bands': ['8', '11'], 't_3': 0.6,
                       'ndvi_bands': ['8', '4'],
                       't_ndvi': 0.5}  # Settings for the cloud/cloud shadow detection algorithms
    #
    # Training data cropping: The amount of pixels used for training, which belong to the training regions described
    # in the manuscript, can be cropped to speed up code execution
    training_data_crop_ratio = {'charles_river': 0.7, 'oroville_dam': 0.5, 'multiearth': 1} #
    #
    # Model Selection (GMM)
    # These values have been set after data inspection
    # The thresholds are used to compute the scaled index model mean and std values, and also for labeling
    gmm_num_components = {'charles_river_3class': [5, 3, 3], 'charles_river': [5, 3],
                          'oroville_dam': [5, 3],
                          'multiearth': [5, 3]}
    #
    # Spectral Index
    threshold_index_labels = {'charles_river_3class': [-0.05, 0.35],
                              'oroville_dam': [0.13], 'charles_river': [0.13],
                              # 'multiearth': [-0.67]}
                              'multiearth': [-0.65]}
    # https://www.geo.university/pages/spectral-indices-in-remote-sensing-and-how-to-interpret-them
    # NDWI (multiearth), NDVI (charles), MNDWI (oroville)
    # the value range of these indices is from -1 to 1
    # NDWI higher than 0.2 should be water

    # --------------------------------------------------------------------
    # Evaluation Parameters
    # --------------------------------------------------------------------

    #
    # TODO: Change the pickle-related variables so that (a) new results are generated and stored or (b) previously
    #  generated and stored results are loaded for the next code execution
    # Configuration Options
    evaluation_generate_results = True  # False if wanting to skip evaluation
    # If evaluation is skipped, previously stored evaluation results are plotted
    evaluation_store_results = False # True if wanting to store evaluation results for
    # further visualization
    evaluation_generate_results_likelihood = False  # True if wanting to generate instantaneous classifier results and
    # store them, False if wanting to use pre-generated and pre-stored likelihood results

    #
    # Offset of evaluation images to consider in the evaluation stage: For the manuscript we have skipped dates in the
    # case their associated images did not provide a good initialization for the recursive framework
    offset_eval_images = {'oroville_dam': 1, 'charles_river': 3, 'multiearth': 0}
    #
    # Index of images to include in the evaluation loop:
    # In the appendix figure we plot classification maps for all the images in the evaluation loop.
    index_images_to_evaluate = {'1b': [*range(offset_eval_images[scenario], 42, 1)],
                                '1a': [*range(offset_eval_images[scenario], 42, 1)],
                                '2': [*range(offset_eval_images[scenario], 28, 1)],
                                # '2': [*range(offset_eval_images[scenario], 5, 1)],
                                # '3': [0, 2, 3, 5, 7, 9, 11, 21, 49, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78,
                                # 81, 82, 83, 89, 90, 101, 103, 108, 130, 134, 135, 140, 142, 144, 145, 146, 147,
                                # 148, 150, 161, 175] # original vector from cloud filter
                                # '3': [0, 2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                # 82, 90, 101, 103, 108, 134, 145, 146,
                                # 150, 161, 175]}
                                # '3': [2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                #       82, 90, 101, 103, 108, 134, 145, 146,
                                #       150, 161, 175]  # plot all dates
                                '3': [2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                                      82, 90, 101, 103, 108, 134, 145, 146,
                                      150, 161, 175]  # plot all dates
                                }
    #
    # Index of images to plot in the classification results figure
    index_images_to_plot = {'1a': [*range(offset_eval_images[scenario], 42, 4)],
                            '1b': [*range(offset_eval_images[scenario], 42, 8)],
                            # '2': [*range(offset_eval_images[scenario], 28, 1)],
                            '2': [3, 5, 9, 13, 17, 21, 25],
                            # '3': [0, 2, 3, 5, 7, 9, 11, 21, 49, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78,
                            # 81, 82, 83, 89, 90, 101, 103, 108, 130, 134, 135, 140, 142, 144, 145, 146, 147,
                            # 148, 150, 161, 175] # original vector from cloud filter
                            # '3': [0, 2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                            # 82, 90, 101, 103, 108, 134, 145, 146,
                            # 150, 161, 175]}
                            '3': [5, 66, 76, 134, 150]  # plot dates for qualitative analysis experiment 3
                            }
    # If wanting to store evaluation results obtained with all dates (consider the previous important note regarding
    # big file sizes):
    index_images_to_store = {'1b': [*range(offset_eval_images[scenario], 42, 1)],
                             '1a': [*range(offset_eval_images[scenario], 42, 1)],
                             '2': [*range(offset_eval_images[scenario], 28, 1)],
                             '3': [*range(0, 217)]}
    #
    # If wanting to store only evaluation results to reproduce Figure 5 (these results are already provided in the Zenodo folder)
    # Config.index_images_to_store = Config.index_images_to_plot
    #
    #
    # Set the following parameter to a value different from 0 if wanting to read a specific number of images
    # different from the ones available
    num_evaluation_images_hardcoded = 0

    # --------------------------------------------------------------------
    # Quantitative Analysis (QA)
    # --------------------------------------------------------------------
    water_mapping_models = ["Scaled Index", 'GMM', "Logistic Regression", 'DeepWaterMap', 'WatNet', 'RScaled Index',
                            'RGMM', 'RLogistic Regression', 'RDeepWaterMap', 'RWatNet']
    mapping_models = ["Scaled Index", 'GMM', "Logistic Regression", 'RScaled Index', 'RGMM',
                      'RLogistic Regression']
    qa_settings = {
        # Options
        'save_results': False,  # If QA results are saved we can (1) compute boxplot and (2) update table afterwards
        # Accuracy metrics considered
        # 'metrics': ["accuracy", "balanced_accuracy", "f1"],
        'metrics': ["balanced_accuracy"],
        'main_metric': "balanced_accuracy",  # this metric is shown in tables/figures
        # Index of images considered for the quantitative analysis
        'index_quant_analysis': {'1a': index_images_to_plot['1a'],
                                 '1b': index_images_to_plot['1b'],
                                 '2': index_images_to_plot['2'],
                                 '3': {5: 0, 66: 1, 76: 2, 134: 3,
                                       150: 4}  # image_idx : label_idx  for Config.test_site = 4
                                 },
        # Models to evaluate
        'models': {'1a': water_mapping_models,
                   '1b': water_mapping_models,
                   '2': water_mapping_models,
                   '3': mapping_models  # no DL models are evaluated as this is a deforestation detection problem
                   }
    }

    # --------------------------------------------------------------------
    # Label Generation - [DEPRECATED FOR THIS IMPLEMENTATION]
    # --------------------------------------------------------------------
    #
    # Label generation consists in 3 main steps:
    #   (1) - Save RGB image as png
    #   (2) - Import RGB to LabelStudio Tool, create image label with the tool and export as png
    #   (3) - Read png label and generate array label as required
    label_generation_settings = {
        # Options
        'save_rgb': False,
        'generate_label': False
    }
