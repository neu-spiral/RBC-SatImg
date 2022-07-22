import random

from configuration import Config
from image_reader import ReadSentinel2
from training import training_main
from evaluation import evaluation_main
from datetime import datetime

from configuration import Debug

if __name__ == "__main__":
    """ Main function containing the main logic and flow of the code.
    
    Instead of calling this function, it is possible to run the jupyter notebook file
    provided in this project.
    
    """
    # Initialize random seed
    random.seed(1)

    # Set logging path
    Debug.set_logging_file(time_now=datetime.now())

    # Instance of Image Reader object
    image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                 Config.image_dimensions[Config.scenario]['dim_y'])

    # Training Stage
    labels, gmm_densities, trained_lr_model = training_main(image_reader)

    # Evaluation Stage
    # This stage includes the plotting of results
    evaluation_main(gmm_densities, trained_lr_model, image_reader)
