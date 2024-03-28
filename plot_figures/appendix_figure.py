import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from image_reader import ReadSentinel2
from configuration import Config, Debug, Visual
from matplotlib import colors
from figures import get_rgb_image, get_green_image
from plot_figures.quantitative_analysis.update_table import update_table

class AppendixFigure:
    def __init__(self):
        """ Initializes instance of RBC object with the corresponding class attributes.

        """

        # Layout/Style
        self.cmap = colors.ListedColormap(Visual.cmap[Config.scenario])
        plt.rc('font', family='serif')

        # Image Reader
        self.image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                          Config.image_dimensions[Config.scenario]['dim_y'])
        self.x_coords = Config.pixel_coords_to_evaluate[Config.test_site]['x_coords']
        self.y_coords = Config.pixel_coords_to_evaluate[Config.test_site]['y_coords']

        # Evaluated models (plot matrix columns)
        self.plot_legend = Visual.qa_fig_settings['legend_models'][Config.test_site].copy()
        self.plot_legend.append('RGB')
        self.models = Config.qa_settings['models'][Config.test_site]

        # Dates/Images plotted (plot matrix rows)
        self.index_results_to_plot = Config.index_images_to_evaluate[Config.test_site]
        self.n_results_to_plot = len(self.index_results_to_plot)

        self.f, self.axarr = plt.subplots(self.n_results_to_plot, len(self.models)+1, figsize=(9, 20))


    def plot_results(self, image_idx: int, image_all_bands: np.ndarray, date_string: str,
                              base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int):
    #use the following line if wanting to use the predicted probabilities (besides the predicted classes)
    #def plot_results_one_date(self, image_idx: int, image_all_bands: np.ndarray, date_string: str, base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int, base_model_predicted_probabilities: np.ndarray):
        """ Plots all the results that have been stored with the corresponding configuration.

        """

        #
        # Plot RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)
        # Scaling to increase illumination
        rgb_image = rgb_image * Visual.scaling_rgb[Config.test_site]
        self.axarr[result_idx, len(self.models)].imshow(
            rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], aspect='auto')
        self.axarr[result_idx, len(self.models)].axis('off')

        #
        # Plot Classification Map
        for idx, model_i in enumerate(self.models):
            if model_i[0] == 'R':
                # Plot posterior results for RBC classifiers
                result = posterior[model_i[1:]][self.x_coords[0]:self.x_coords[1],
                         self.y_coords[0]:self.y_coords[1]]
            else:
                # Plot likelihood results for instantaneous classifiers
                result = base_model_predicted_class[model_i][self.x_coords[0]:self.x_coords[1],
                         self.y_coords[0]:self.y_coords[1]]

            self.axarr[result_idx, idx].imshow(result, self.cmap, aspect='auto')

            # self.axarr[result_idx, 1].imshow(
            #     base_model_predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 1].axis('off')
            #
            # self.axarr[result_idx, 2].imshow(
            #     base_model_predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
            #     self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 2].axis('off')
            #
            # self.axarr[result_idx, 3].imshow(
            #     base_model_predicted_class["DeepWaterMap"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
            #     self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 3].axis('off')
            #
            # self.axarr[result_idx, 4].imshow(
            #     base_model_predicted_class["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 4].axis('off')
            #
            # self.axarr[result_idx, 5].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
            #                               self.y_coords[0]:self.y_coords[1]],
            #                               self.cmap, aspect='auto')
            # self.axarr[result_idx, 5].axis('off')
            #
            # self.axarr[result_idx, 6].imshow(
            #     posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
            #     self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 6].axis('off')
            #
            # self.axarr[result_idx, 7].imshow(
            #     posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
            #     self.y_coords[0]:self.y_coords[1]], self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 7].axis('off')
            #
            # self.axarr[result_idx, 8].imshow(posterior["DeepWaterMap"][self.x_coords[0]:self.x_coords[1],
            #                               self.y_coords[0]:self.y_coords[1]],
            #                               self.cmap, aspect='auto')
            # self.axarr[result_idx, 8].axis('off')
            #
            # self.axarr[result_idx, 9].imshow(
            #     posterior["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
            #     self.cmap,
            #     aspect='auto')
            # self.axarr[result_idx, 9].axis('off')

        #
        # Layout
        for axis in ['top', 'bottom', 'left', 'right']:
            self.axarr[result_idx, 0].spines[axis].set_linewidth(0)
        self.axarr[result_idx, 0].set_ylabel(date_string, rotation=0, fontsize=20, fontfamily='Times New Roman')
        # The label position must be changed accordingly, considering the amount of images plotted
        if True:
            # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
            self.axarr[result_idx, 0].yaxis.set_label_coords(-1.1, 0.35)
        else:
            # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
            self.axarr[result_idx, 0].yaxis.set_label_coords(-0.9, 0.4)

        #
        # Set figure labels
        for idx, label in enumerate(self.plot_legend):
            self.axarr[0, idx].title.set_fontfamily('Times New Roman')
            self.axarr[0, idx].title.set_fontsize(20)
            self.axarr[0, idx].title.set_text(self.plot_legend[idx])
            self.axarr[result_idx, idx].get_yaxis().set_ticks([])
            self.axarr[result_idx, idx].get_xaxis().set_ticks([])
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, idx].spines[axis].set_linewidth(0)
        self.f.show()

