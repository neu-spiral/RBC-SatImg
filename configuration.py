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
    gmm_dump_pickle = False
    trained_lr_model_pickle = False

    @staticmethod
    def set_logging_file(time_now: datetime):
        """ Creates a log file for the current code execution.

        Parameters
        ----------
        time_now : datetime
            time at which the log file is created (year, month, day, hours of the day, etc.)

        """
        # Get log file path
        file_name = time_now.strftime("%d%m%Y_%H%M%S.log")
        file_path = os.path.join(Config.path_log_files, file_name)

        # Set log file basic configuration
        logging.basicConfig(filename=file_path, level=logging.DEBUG, format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

        # Write first commands in the log file
        logging.debug(f"Log file has been created at {time_now}")
        logging.debug(f"Selected scenario: {Config.scenario}")


class Config:
    """ Some configuration settings must be changed when executing the code with data that is
    different to the one provided by the authors.

    """

    # Paths
    path_sentinel_images = r"C:\Users\helen\Documents\Northeastern\Project_Water_Mapping\repo\sentinel_data"
    path_watnet_pretrained_model = os.path.join(os.getcwd(), r"benchmark\watnet\model\pretrained\watnet.h5")
    path_log_files = os.path.join(os.getcwd(), r"logs")
    path_training_data = os.path.join(os.getcwd(), r"training_data")

    # Scenario selection
    scenario = "oroville_dam"  # charles_river | oroville_dam

    # Scene selection
    scene_id = 1  # 0 | 1 | 2
    #   - scene_id = 0 if wanting to process the whole image (case of Charles River scenario).
    #   - scene_id = 1 for scene A in Oroville Dam (water stream)
    #   - scene_id = 2 for scene B in Oroville Dam (peninsula)

    # Coordinates of the pixels to evaluate depending on the selected scene
    # In the case of scene_id = 0, no coordinates are specified because the whole image wants to be
    # evaluated.
    pixel_coords_to_evaluate = {1: {'x_coords': [1700, 1900], 'y_coords': [500, 1000]}, 2: {'x_coords': [1400, 1550], 'y_coords': [1540, 1650]}}

    # Classes to evaluate
    classes = {'charles_river': ['water', 'land', 'vegetation'], 'oroville_dam': ['water', 'no water']}

    # Scaling factors
    scaling_factor_sentinel = 1e-4  # Sentinel2 image processing
    scaling_factor_watnet = 1e-4

    # Training data cropping
    training_data_crop_ratio = {'charles_river': 1, 'oroville_dam': 0.5}
    # Gaussian Mixtures Model Selection
    # These values have been set after data inspection
    gm_model_selection = {'charles_river': {'num_components': [5, 3, 3], 'thresholds': [0, 0.2]},
                          'oroville_dam': {'num_components': [5, 3], 'thresholds': [0.13]}}

    # Scaled Spectral Indices: values for their probabilistic model
    scaled_index_model = {'mean_values': [0.25, 0.55, 0.8], 'std_values': [0.25, 0.05, 0.2]}
    # For the 3 classes 'charles_river' scenario, the authors propose
    # *scaled_index_model = {'mean_values': [0.25, 0.55, 0.8], 'std_values': [0.25, 0.05, 0.2]}*
    # TODO: Define the following case (2 classes 'oroville dam')
    # For the 2 classes 'oroville_dam' scenario, the authors propose
    # *scaled_index_model = {'mean_values': [], 'std_values': []}*

    # Prior probabilities
    prior_probabilities = {'charles_river': [0.8, 0.2, 0.1],
                           'oroville_dam': [0.5, 0.5]}

    # Spectral Bands used to calculate the Spectral Indices
    # For NDVI, NIR (8) and Red (4) bands are used. Used for 'Charles River' scenario by the authors.
    # For NDWI, Green (3) and NIR (8) bands are used
    # For MNDWI, Green (3) and SWIR (11) bands are used. Used for 'Oroville Dam' scenario by the authors.
    bands_spectral_index = ['3', '11']  # band identifiers must be a string

    # Spectral Bands
    # The main selected bands are the following
    bands_to_read = ['2', '3', '4', '8A', '8', '11']  # band identifiers must be a string
    # In the paper associated to the 'watnet' algorithm the authors mention the selection of the following bands
    bands_watnet = [3, 4, 5, 6, 1, 2]
    # In the paper associated to the 'deepwaternet' algorithm the authors mention the selection of the following bands
    bands_deepwaternet = [3, 4, 5, 6, 1, 2]

    # Type of the images to be read
    image_types = {'charles_river': 'TCG', 'oroville_dam': 'SFJ'}
    #   - 'charles_river': TCH|TCG
    #   - 'oroville_dam': SFJ|TFK

    # Pixel coordinates of the images to read (dimensions)
    # TODO: Change the code so that this does not need to be hardcoded in the configuration file
    image_dimensions = {'charles_river': {'dim_x': 43, 'dim_y': 60}, 'oroville_dam': {'dim_x': 2229, 'dim_y': 3341}}

    # Transition Matrix
    # The transition matrix is given by the user (hardcoded in this configuration file)
    eps = 0.1
    transition_matrix = np.array([1 - eps, eps, eps, 1 - eps]).reshape(2, 2)