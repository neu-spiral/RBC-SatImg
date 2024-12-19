import random

from configuration import Config, Debug
from image_reader import ReadSentinel2
from training import training_main
from deprecated.evaluation_deprecated import evaluation_main
from datetime import datetime

# --------------------------------------------------------------------
# TO BE CHANGED BY USER
# --------------------------------------------------------------------
# TODO: Change accordingly:
# path_save_figure = os.path.join(Config.path_evaluation_results, "sensitivity_analysis", "multiearth", "sensitivity_analysis_multiearth.pdf")
# Config.path_evaluation_results = r"/Users/helena/Library/Mobile Documents/com~apple~CloudDocs/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset/-54.60_-4.05/evaluation_results"
Config.test_site = '1a'
# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
epsilon_evaluation_vector = [0.001]


# CALLING MAIN FOR ALL VALUES OF EPSILON
for eps_i in epsilon_evaluation_vector:

    # Initialize random seed
    random.seed(1)

    # Set logging path
    Debug.set_logging_file(time_now=datetime.now())

    # Instance of Image Reader object
    image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                 Config.image_dimensions[Config.scenario]['dim_y'])

    # Training Stage
    labels, gmm_densities, trained_lr_model = training_main(image_reader)

    # Change epsilon according to iteration value for sensitivity analysis
    Config.eps_SIC[Config.test_site] = eps_i
    Config.eps_DWM[Config.test_site] = eps_i
    Config.eps_GMM[Config.test_site] = eps_i
    Config.eps_LR[Config.test_site] = eps_i
    Config.eps_WN[Config.test_site] = eps_i
    # Config.eps_adaptive[Config.test_site] = eps_i
    # Config.eps_DWM_adaptive[Config.test_site] = eps_i
    # Config.eps_GMM_adaptive[Config.test_site] = eps_i
    # Config.eps_LR_adaptive[Config.test_site] = eps_i

    Config.fig_id = f"sensitivity_analysis_testsite_{Config.test_site}_eps_{eps_i}"

    # Evaluation Stage
    # This stage includes the plotting of plot_figures
    print("EVALUATION STARTS")
    evaluation_main(gmm_densities, trained_lr_model, image_reader)


