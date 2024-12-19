import random
from datetime import datetime

from configuration import Config, Debug
from image_reader import ReadSentinel2
from training import training_main
from plot_figures.evaluation import evaluation_main

"""
main.py

This script initializes the main logic and flow of the code, including the training and evaluation stages.

Corresponding Author:
- Helena Calatrava (Northeastern University, Boston, MA, USA)
Last Updated:
- December 2024

Usage:
- Run this script directly to perform the training and evaluation stages:
    `python main.py`
- Alternatively, use the provided Jupyter notebook for an interactive workflow.
"""


if __name__ == "__main__":
    """ Main function containing the main logic and flow of the code (training + evaluation).
    
    Instead of calling this function, it is possible to run the jupyter notebook file
    provided in this project.
    
    Classification accuracy results are calculated in the evaluation step.
    
    """
    # Initialize random seed
    random.seed(1)

    # Set logging path
    Debug.set_logging_file(time_now=datetime.now())

    # Instance of Image Reader object
    image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                 Config.image_dimensions[Config.scenario]['dim_y'])

    # Training Stage
    print("TRAINING STARTS")
    labels, gmm_densities, trained_lr_model = training_main(image_reader)

    # Evaluation Stage
    # This stage includes the plotting of plot_figures
    print("EVALUATION STARTS")
    evaluation_main(gmm_densities, trained_lr_model, image_reader)
