import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from image_reader import ReadSentinel2
from configuration import Config, Debug, Visual
from matplotlib import colors
from figures import get_rgb_image, get_green_image

class ClassificationResultsFigure:
    def __init__(self):
        """ Initializes instance of RBC object with the corresponding class attributes.

        """
        self.cmap = colors.ListedColormap(Visual.cmap[Config.scenario])
        self.image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                          Config.image_dimensions[Config.scenario]['dim_y'])
        self.x_coords = Config.pixel_coords_to_evaluate[Config.test_site]['x_coords']
        self.y_coords = Config.pixel_coords_to_evaluate[Config.test_site]['y_coords']

    def create_quantitative_results_figure(self):
        """ Creates figure to show quantitative classification results.

        """
        self.f, self.axarr = plt.subplots(len(Config.index_quant_analysis[Config.test_site]), 13, figsize=(15, 15))

    def get_quantitative_results(self, label_image: np.ndarray, class_labels: np.ndarray):
        classification_map = (np.abs(class_labels - label_image) - 1)*(-1) # 1 means a match
        matched_pixel_percentage = np.sum(classification_map)/(classification_map.shape[0]*classification_map.shape[1])
        return classification_map, matched_pixel_percentage

    def create_figure(self):
        """ Creates figure to show qualitative classification results.

        """
        self.index_results_to_plot = Config.index_images_to_plot[Config.test_site]
        self.n_results_to_plot = len(self.index_results_to_plot)
        if self.n_results_to_plot > 15:
            self.big_figure = True
        else:
            self.big_figure = False

        # Plot classification results for Oroville Dam scenario
        if Config.test_site == 1 or Config.test_site == 2:
            print(f"Results for Oroville Dam, k = {Config.test_site}")

            # Vector with models
            self.plot_legend = ["Scaled Index", 'GMM', "Logistic Regression", 'DeepWaterMap', 'WatNet', 'RSIC', 'RGMM', 'RLR', 'RDeepWaterMap', 'RWatNet', 'RGB']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'DeepWaterMap', 'WatNet', 'RScaled Index', 'RGMM', 'RLogistic Regression', 'RDeepWaterMap', 'RWatNet']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'RScaled Index', 'RGMM',
                           'RLogistic Regression']

            # Create figure (size changes depending on the amount of plotted images)
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 11, figsize=(9, 30))
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 11, figsize=(9, 10))

        # Plot plot_figures for Charles River scenario
        elif Config.test_site == 3:
            print("Results for Charles River, Study Area C")

            # Create figure (size changes  depending on the amount of plotted images)
            if self.big_figure == False:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 7, figsize=(9, 10))
            else:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 7, figsize=(9, 30))

            # Vectors with models to plot
            self.plot_legend = ["Scaled Index", 'GMM', "Logistic Regression", 'RSIC', 'RGMM', 'RLR', 'RGB']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'RScaled Index', 'RGMM', 'RLogistic Regression']

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
            # plt.subplot_tool()
            # plt.show()
        elif Config.test_site == 4:
            print("Results for MultiEarth Dataset")

            # Create figure
            #self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(5, 50))
            if self.big_figure:
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(10, 35))  # full results
            else:
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(5, 7)) # small one only quantitative

            # Vectors with models to plot
            self.plot_legend = ['Acq. date      SIC', 'GMM', "Logistic Regression", 'RSIC', 'RGMM', 'RLR', 'RGB', 'Label     Label Date']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'RScaled Index', 'RGMM', 'RLogistic Regression']

            # Adjust space between subplots
            # plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.subplots_adjust(top=0.437, right=0.776)
        else:
            print("No plot_figures of this scene appearing in the publication.")
        return self.f

    def create_figure_errormap(self):
        """ Creates figure to show qualitative classification results.

        """
#
        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #
        self.index_results_to_plot = Config.index_images_to_plot[Config.test_site]
        self.n_results_to_plot = len(self.index_results_to_plot)*2
        if self.n_results_to_plot > 15:
            self.big_figure = True
        else:
            self.big_figure = False

        #
        # Plot classification results for Test Sites 1a and 1b
        if Config.test_site == 1 or Config.test_site == 2:
            print(f"Computing results for Oroville Dam (Test site {Config.test_site})")

            # Vector with models
            self.plot_legend = ["SIC", 'GMM', "LR", 'DWM', 'WN', 'RSIC', 'RGMM', 'RLR', 'RDWM', 'RWN']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'DeepWaterMap', 'WatNet', 'RScaled Index', 'RGMM', 'RLogistic Regression', 'RDeepWaterMap', 'RWatNet']

        # Plot classification results for Test Site 2
        elif Config.test_site == 3:
            print(f"Computing results for Charles River (Test site {Config.test_site})")

            # Vectors with models to plot
            self.plot_legend = ["SIC", 'GMM', "LR", 'DWM', 'WN', 'RSIC', 'RGMM', 'RLR', 'RDWM', 'RWN']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'DeepWaterMap', 'WatNet', 'RScaled Index', 'RGMM', 'RLogistic Regression', 'RDeepWaterMap', 'RWatNet']

        # Plot classification results for Test Site 3
        elif Config.test_site == 4:
            print(f"Computing results for MultiEarth Scenario (Test site {Config.test_site})")

            # Vectors with models to plot
            self.plot_legend =  ["SIC", 'GMM', "LR", 'RSIC', 'RGMM', 'RLR']
            self.models = ["Scaled Index", 'GMM', "Logistic Regression", 'RScaled Index', 'RGMM', 'RLogistic Regression']

        else:
            print("No plot_figures of this scene appearing in the publication.")

        #
        # Initialize Quantitative Analysis vector
        self.results_qa = dict()
        for acc in Config.metrics:
            self.results_qa[acc] = np.ndarray(shape=(int(self.n_results_to_plot / 2), len(self.models) + 1))

        #
        # Create figure (size changes depending on the amount of plotted images)
        if self.big_figure:
            # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
            self.f, self.axarr = plt.subplots(self.n_results_to_plot, len(self.models) + 1, figsize=(9, 30))
        else:
            # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
            self.f, self.axarr = plt.subplots(self.n_results_to_plot, len(self.models) + 1, figsize=(9, 10))

        return self.f

    def plot_stored_results(self):
        """ Plots all the results that have been stored with the corresponding configuration.

        """
        # Path with results
        path_results = os.path.join(Config.path_evaluation_results, "classification",
                                    f'{Config.scenario}_{Config.test_site}')
        path_images = os.path.join(Config.path_sentinel_images, f'{Config.scenario}', 'evaluation')

        # Plot classification results for Oroville Dam scenario
        if Config.test_site == 1 or Config.test_site == 2:
            print(f"Results for Oroville Dam, k = {Config.test_site}")

            for image_i in range(0, self.n_results_to_plot):
                # Read image again to be able to get RGB image
                image_all_bands, date_string = self.image_reader.read_image(path=path_images,
                                                                            image_idx=self.index_results_to_plot[
                                                                                image_i])

                # Get RGB Image
                rgb_image = get_rgb_image(image_all_bands=image_all_bands)
                # Scaling to increase illumination
                rgb_image = rgb_image * Visual.scaling_rgb[Config.test_site]

                # Read stored evaluation plot_figures to reproduce published figure
                pickle_file_path = os.path.join(path_results, f"oroville_dam_{Config.test_site}_image_{self.index_results_to_plot[image_i]}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl")
                [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_figures
                self.axarr[image_i, 0].imshow(
                    predicted_class["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 0].get_yaxis().set_ticks([])
                self.axarr[image_i, 0].get_xaxis().set_ticks([])

                # Remove axis
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[image_i, 0].spines[axis].set_linewidth(0)
                self.axarr[image_i, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

                # The label position must be changed accordingly, considering the amount of images plotted
                if self.big_figure:
                    # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                    self.axarr[image_i, 0].yaxis.set_label_coords(-0.9, 0.35)
                else:
                    # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                    self.axarr[image_i, 0].yaxis.set_label_coords(-0.9, 0.4)

                self.axarr[image_i, 1].imshow(
                    predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 1].axis('off')
                self.axarr[image_i, 2].imshow(
                    predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 2].axis('off')
                self.axarr[image_i, 3].imshow(
                    predicted_class["DeepWaterMap"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 3].axis('off')
                self.axarr[image_i, 4].imshow(
                    predicted_class["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 4].axis('off')
                self.axarr[image_i, 5].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap, aspect='auto')
                self.axarr[image_i, 5].axis('off')
                self.axarr[image_i, 6].imshow(
                    posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 6].axis('off')
                self.axarr[image_i, 7].imshow(
                    posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                    self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 7].axis('off')
                self.axarr[image_i, 8].imshow(posterior["DeepWaterMap"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap, aspect='auto')
                self.axarr[image_i, 8].axis('off')
                self.axarr[image_i, 9].imshow(
                    posterior["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 9].axis('off')
                self.axarr[image_i, 10].imshow(
                    rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], aspect='auto')
                self.axarr[image_i, 10].axis('off')

            # Set figure labels
            for idx, label in enumerate(self.plot_legend):
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(12)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])

        # Plot plot_figures for Charles River scenario
        elif Config.test_site == 3:
            print("Results for Charles River, Study Area C")

            for image_i in range(0, self.n_results_to_plot):

                # Read image again to be able to get RGB image
                image_all_bands, date_string = self.image_reader.read_image(path=path_images,
                                                                            image_idx=self.index_results_to_plot[
                                                                                image_i])

                # Get RGB Image
                rgb_image = get_rgb_image(image_all_bands=image_all_bands)

                # Read stored evaluation plot_figures to reproduce published figure
                pickle_file_path = os.path.join(path_results,
                                                f"charles_river_{Config.test_site}_image_{self.index_results_to_plot[image_i]}_epsilon_{Config.eps}.pkl")
                [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_figures
                self.axarr[image_i, 0].imshow(
                    predicted_class["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap)
                self.axarr[image_i, 0].get_yaxis().set_ticks([])
                self.axarr[image_i, 0].get_xaxis().set_ticks([])

                # Remove axis
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[image_i, 0].spines[axis].set_linewidth(0)

                self.axarr[image_i, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

                # The label position must be changed accordingly, considering the amount of images plotted
                if self.big_figure:
                    # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                    self.axarr[image_i, 0].yaxis.set_label_coords(-0.8, 0.25)
                else:
                    # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                    self.axarr[image_i, 0].yaxis.set_label_coords(-0.7, 0.25)

                self.axarr[image_i, 1].imshow(
                    predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
                self.axarr[image_i, 1].axis('off')
                self.axarr[image_i, 2].imshow(
                    predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap)
                self.axarr[image_i, 2].axis('off')
                self.axarr[image_i, 3].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap)
                self.axarr[image_i, 3].axis('off')
                self.axarr[image_i, 4].imshow(
                    posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap)
                self.axarr[image_i, 4].axis('off')
                self.axarr[image_i, 5].imshow(
                    posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                    self.y_coords[0]:self.y_coords[1]], self.cmap)
                self.axarr[image_i, 5].axis('off')
                self.axarr[image_i, 6].imshow(
                    rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
                self.axarr[image_i, 6].axis('off')

            # Set figure labels
            for idx, label in enumerate(self.plot_legend):
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(12)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
            # plt.subplot_tool()
            # plt.show()
        else:
            print("No plot_figures of this scene appearing in the publication.")

        # Save figure as pdf
        if Debug.save_figures:
            plt.savefig(os.path.join(Config.path_results_figures,
                                     f'classification_{Config.scenario}_{Config.test_site}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pdf'),
                        format="pdf", bbox_inches="tight", dpi=200)

    def plot_results_one_date(self, image_idx: int, image_all_bands: np.ndarray, date_string: str,
                              base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int):
    #use the following line if wanting to use the predicted probabilities (besides the predicted classes)
    #def plot_results_one_date(self, image_idx: int, image_all_bands: np.ndarray, date_string: str, base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int, base_model_predicted_probabilities: np.ndarray):
        """ Plots all the results that have been stored with the corresponding configuration.

        """

        # Plot classification results for Oroville Dam scenario
        if Config.test_site == 1 or Config.test_site == 2:
            print(f"Results for Oroville Dam, k = {Config.test_site}")

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands)
            # Scaling to increase illumination
            rgb_image = rgb_image * Visual.scaling_rgb[Config.test_site]

            # Plot plot_figures
            self.axarr[result_idx, 0].imshow(
                base_model_predicted_class["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 0].get_yaxis().set_ticks([])
            self.axarr[result_idx, 0].get_xaxis().set_ticks([])

            # Remove axis
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, 0].spines[axis].set_linewidth(0)
            self.axarr[result_idx, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

            # The label position must be changed accordingly, considering the amount of images plotted
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.9, 0.35)
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.9, 0.4)

            self.axarr[result_idx, 1].imshow(
                base_model_predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 1].axis('off')
            self.axarr[result_idx, 2].imshow(
                base_model_predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 2].axis('off')
            self.axarr[result_idx, 3].imshow(
                base_model_predicted_class["DeepWaterMap"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 3].axis('off')
            self.axarr[result_idx, 4].imshow(
                base_model_predicted_class["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 4].axis('off')
            self.axarr[result_idx, 5].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap, aspect='auto')
            self.axarr[result_idx, 5].axis('off')
            self.axarr[result_idx, 6].imshow(
                posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 6].axis('off')
            self.axarr[result_idx, 7].imshow(
                posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 7].axis('off')
            self.axarr[result_idx, 8].imshow(posterior["DeepWaterMap"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap, aspect='auto')
            self.axarr[result_idx, 8].axis('off')
            self.axarr[result_idx, 9].imshow(
                posterior["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 9].axis('off')
            self.axarr[result_idx, 10].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], aspect='auto')
            self.axarr[result_idx, 10].axis('off')

            # Set figure labels
            for idx, label in enumerate(self.plot_legend):
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(12)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])

        # Plot plot_figures for Charles River scenario
        elif Config.test_site == 3:
            print("Results for Charles River, Study Area C")

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands) * Visual.scaling_rgb[Config.test_site]

            # Plot Scaled Index (SIC)
            self.axarr[result_idx, 0].imshow(
                base_model_predicted_class["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 0].get_yaxis().set_ticks([])
            self.axarr[result_idx, 0].get_xaxis().set_ticks([])

            # Remove axis
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, 0].spines[axis].set_linewidth(0)

            self.axarr[result_idx, 0].set_ylabel(date_string, rotation=0, fontsize=10, fontfamily='Times New Roman')# new plot fontsize changed

            # The label position must be changed accordingly, considering the amount of images plotted
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.45, 0.3)
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.3, 0.3)
            self.axarr[result_idx, 0].yaxis.set_label_coords(-0.88, 0.32)  # new plot location changed

            # Plot GMM
            self.axarr[result_idx, 1].imshow(
                base_model_predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 1].axis('off')

            # Plot LR
            self.axarr[result_idx, 2].imshow(
                base_model_predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 2].axis('off')

            # Plot Recursive Scaled Index (RSIC)
            self.axarr[result_idx, 3].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap)
            self.axarr[result_idx, 3].axis('off')

            # Plot RGMM
            self.axarr[result_idx, 4].imshow(
                posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 4].axis('off')

            # Plot RLR
            self.axarr[result_idx, 5].imshow(
                posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 5].axis('off')

            # Plot RGB Image
            self.axarr[result_idx, 6].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, 6].axis('off')

            if Debug.save_rgb:
                rgb_cut = rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]


                # Save RGB Image without margin to create label with LabelStudio
                from PIL import Image
                plt.figure()
                plt.imshow(rgb_cut)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/myfig.png",
                            bbox_inches='tight', pad_inches=0)

                # Create water label (charles_river)
                water_label_1_path = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/water_label_1.png"
                water_label_2_path  = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/water_label_2.png"
                water_label_1 = np.array(Image.open(water_label_1_path).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                water_label_2 = np.array(Image.open(water_label_2_path).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                water_label = water_label_2 + water_label_1
                label_attempt = np.zeros(shape=rgb_cut[:, :, 0].flatten().shape)
                label_attempt[np.where(water_label.flatten() > 100)] = 1
                label = label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1])
                # plt.figure()
                # plt.imshow(label)
                # plt.gca().set_axis_off()
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                #                     hspace=0, wspace=0)
                # plt.margins(0, 0)
                # path_label = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/charles_river_water_label.png"
                # plt.savefig(path_label, bbox_inches='tight', pad_inches=0, dpi=1000)
                # read_label= np.array(Image.open(path_label).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                # plt.figure()
                # plt.imshow(read_label)
                import imageio
                path_label = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/charles_river_water_label.tiff"
                imageio.imwrite(path_label, label)
                # read_label = imageio.imread(path_label)
                # plt.figure(),
                # plt.imshow(read_label)

                # Label vegetation
                # scenario_vegetation = 'multiearth'
                # from tools.spectral_index import get_broadband_index, get_labels_from_index
                # sic = get_broadband_index(data=image_all_bands, bands=Config.bands_spectral_index[scenario_vegetation])
                # labels = get_labels_from_index(index=sic, num_classes=len(Config.classes[scenario_vegetation]),
                #                                threshold=[-0.58]).reshape([rgb_image.shape[0], rgb_image.shape[1]])
                # labels_vegetation = labels[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                # # Create global label
                # label_attempt = np.ones(shape=rgb_cut[:, :, 0].flatten().shape)*1
                # label_attempt[np.where(water_label.flatten() > 100)] = 0
                # plt.figure()
                # plt.imshow(label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1]))
                # #green_image = get_green_image(image_all_bands=image_all_bands)[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]].flatten()
                # label_attempt[np.intersect1d(np.where(label_attempt == 1)[0], np.where(green_image == 1)[0])] = 2
                # label_attempt[np.intersect1d(np.where(label_attempt == 1)[0], np.where(labels_vegetation.flatten() == 0)[0])] = 2
                # plt.imshow(label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1]),cmap= colors.ListedColormap(['#440154', 'yellow', 'green']))


                # rgb_saved_2 = mpimg.imread(
                #     r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/myfig.png")[
                #             :, :, 0:3]
                # im1= Image.open(r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/myfig.png")
                # im1 = im1.resize([rgb_cut.shape[1], rgb_cut.shape[0]])
                # label_img_toresize = Image.open(label_path)
                # label_resized = label_img_toresize.resize([rgb_cut.shape[1], rgb_cut.shape[0]])
                # plt.figure()
                # plt.imshow(label_resized)
                # plt.figure()
                # plt.imshow(im1)
                # plt.figure()
                # plt.imshow(rgb_cut)
                # im1_copy = np.array(im1)
                # label_resized = np.array(label_resized)
                # plt.figure()
                # plt.imshow(label_resized)
                # res = im1_copy[:,:,0].flatten()
                # res[np.where(label_resized.flatten() > 0)[0]] = 255
                # res = res.reshape([rgb_cut.shape[0], rgb_cut.shape[1]])
                # plt.figure()
                # plt.imshow(res)

                # label_attempt = np.zeros(shape=rgb_cut[:,:,0].flatten().shape)
                # label_attempt[np.where(label_resized.flatten() > 0)] = 1
                # plt.figure()
                # plt.imshow(label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1]))
                #
                # np.where(rgb_saved == [1,1,1])
                # plt.imshow(rgb_cut)
                # green_image = get_green_image(image_all_bands=image_all_bands)
                # plt.imshow(green_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
                # f.savefig(os.path.join(Config.path_results_figures,
                #                                   f'rgb_{result_idx}.png'),
                #                      format="png", dpi=500,bbox_inches='tight' )


            self.plot_legend = ["Scaled Index", 'GMM', "Logistic Regression", 'RSIC', 'RGMM', 'RLR', 'RGB']
            # Set figure labels
            for idx, label in enumerate(self.plot_legend):
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(12)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])
            # for idx, label in enumerate(self.plot_legend): # new plot
            #     self.axarr[0, idx].title.set_fontfamily('Times New Roman')
            #     self.axarr[0, idx].title.set_fontsize(8)
            #     self.axarr[0, idx].title.set_text(self.plot_legend[idx])

            #self.axarr[0, 0].text(-200, 0, 'Acq. date', fontsize=5, fontfamily='Times New Roman')

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
            # plt.subplot_tool()
            # plt.show()
        elif Config.test_site == 4:
            print("Results for MultiEarth dataset")

            # Get RGB Image
            rgb_image = 5*get_rgb_image(image_all_bands=image_all_bands)

            # Plot Scaled Index (SIC)
            self.axarr[result_idx, 0].imshow(
                base_model_predicted_class["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 0].get_yaxis().set_ticks([])
            self.axarr[result_idx, 0].get_xaxis().set_ticks([])

            # Remove axis
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, 0].spines[axis].set_linewidth(0)

            self.axarr[result_idx, 0].set_ylabel(date_string, rotation=0, fontsize=7.5, fontfamily='Times New Roman')# new plot fontsize changed

            # The label position must be changed accordingly, considering the amount of images plotted
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.8, 0.25)
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.7, 0.25)
            self.axarr[result_idx, 0].yaxis.set_label_coords(-0.88, 0.32)  # new plot location changed

            # Plot GMM
            self.axarr[result_idx, 1].imshow(
                base_model_predicted_class["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 1].axis('off')

            # Plot LR
            self.axarr[result_idx, 2].imshow(
                base_model_predicted_class["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 2].axis('off')

            # Plot Recursive Scaled Index (RSIC)
            self.axarr[result_idx, 3].imshow(posterior["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap)
            self.axarr[result_idx, 3].axis('off')

            # Plot RGMM
            self.axarr[result_idx, 4].imshow(
                posterior["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 4].axis('off')

            # Plot RLR
            self.axarr[result_idx, 5].imshow(
                posterior["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 5].axis('off')

            # Plot RGB Image
            self.axarr[result_idx, 6].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, 6].axis('off')

            # Plot MultiEarth label
            if image_idx in Config.index_quant_analysis[Config.test_site].keys():
                label_idx = Config.index_quant_analysis[Config.test_site][image_idx]
                # label_image, date_string_label = image_reader.read_image(path=path_label_images, image_idx=label_idx)
                for file_counter, file_name in enumerate(sorted(os.listdir(Config.path_label_images))):
                    if file_counter == label_idx:
                        path_label_i = os.path.join(Config.path_label_images, file_name)
                        label_image = self.image_reader.read_band(path_label_i)
                        label_date = file_name[-15:-5]
                        label_date_string = f'{label_date[0:4]}-{label_date[5:7]}-{label_date[8:10]}'
                #self.axarr[result_idx, 7].title.set_text(f'L {label_date_string[2:]}')
                self.axarr[result_idx, 7].imshow(label_image)
                self.axarr[result_idx, 7].set_ylabel(label_date_string, rotation=0, fontsize=7.5,
                                                     fontfamily='Times New Roman')  # new plot fontsize changed
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[result_idx, 7].spines[axis].set_linewidth(0)
                self.axarr[result_idx, 7].yaxis.set_label_coords(1.95, 0.325)  # new plot location changed
                self.axarr[result_idx, 7].get_yaxis().set_ticks([])
                self.axarr[result_idx, 7].get_xaxis().set_ticks([])
            else:
                #for axis in ['top', 'bottom', 'left', 'right']:
                    #self.axarr[result_idx, 7].spines[axis].set_linewidth(0)
                #self.axarr[result_idx, 7].yaxis.set_label_coords(1.95, 0.325)  # new plot location changed
                self.axarr[result_idx, 7].get_yaxis().set_ticks([])
                self.axarr[result_idx, 7].get_xaxis().set_ticks([])

            self.plot_legend = ["Scaled Index", 'GMM', "Logistic Regression", 'RSIC', 'RGMM', 'RLR', 'RGB', 'Label']
            # Set figure labels
            for idx, label in enumerate(self.plot_legend):
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(12)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])
            for idx, label in enumerate(self.plot_legend): # new plot
                self.axarr[0, idx].title.set_fontfamily('Times New Roman')
                self.axarr[0, idx].title.set_fontsize(8)
                self.axarr[0, idx].title.set_text(self.plot_legend[idx])

            #self.axarr[0, 0].text(-200, 0, 'Acq. date', fontsize=5, fontfamily='Times New Roman')

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
            # plt.subplot_tool()
            # plt.show()
        else:
            print("No plot_figures of this scene appearing in the publication.")
        self.f.show()

    def plot_results_errormap(self, image_idx: int, image_all_bands: np.ndarray, date_string: str,
                              base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int):
        """ Plots all the results that have been stored with the corresponding configuration.

        """

        #
        # Plot classification results for Test Sites 1a and 1b
        if Config.test_site == 1 or Config.test_site == 2:
            print(f"Computing results for Oroville Dam (Test site {Config.test_site})")
            result_idx = result_idx * 2

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands)
            # Scaling to increase illumination
            rgb_image = rgb_image * Visual.scaling_rgb[Config.test_site]
            # Plot RGB Image
            self.axarr[result_idx, 10].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, 10].axis('off')


            if Debug.save_rgb:
                rgb_cut = rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                # Save RGB Image without margin to create label with LabelStudio
                plt.figure()
                plt.imshow(rgb_cut)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                path_rgb_image = f"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/evaluation/labels/oroville_dam_{Config.test_site}/{Debug.label_folder}/rgb_{image_idx}.png"
                plt.savefig(path_rgb_image,
                            bbox_inches='tight', pad_inches=0)

            # Create label
            if Debug.create_label:
                import imageio
                rgb_cut = rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                from PIL import Image
                path_label = f"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/evaluation/labels/oroville_dam_{Config.test_site}/{Debug.label_folder}/label_{image_idx}.png"
                #path_label = f"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/evaluation/labels/oroville_dam_2/label_1.png"
                water_label = np.array(Image.open(path_label).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                label_attempt = np.zeros(shape=rgb_cut[:, :, 0].flatten().shape)
                label_attempt[np.where(water_label.flatten() > 100)] = 1
                label = label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1])
                path_label = f"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/evaluation/labels/oroville_dam_{Config.test_site}/{Debug.label_folder}/label_{image_idx}.tiff"
                import imageio
                imageio.imwrite(path_label, label)
            try:
                # from PIL import Image
                import imageio
                path_label = f"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/oroville_dam/evaluation/labels/oroville_dam_{Config.test_site}/{Debug.label_folder}/label_{image_idx}.tiff"
                label_image = imageio.imread(path_label)
                self.axarr[result_idx + 1, 10].imshow(label_image, cmap=colors.ListedColormap(Visual.cmap['label']))
            except:
                print('except')

            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx+1, 10].spines[axis].set_linewidth(0)
            self.axarr[result_idx + 1, 10].xaxis.set_label_coords(0, 0.325)
            self.axarr[result_idx+1, 10].get_yaxis().set_ticks([])
            self.axarr[result_idx+1, 10].get_xaxis().set_ticks([])
            # Classification results
            # (likelihood for instantaneous classifiers, posterior for RBCs and classification error map for all
            # classifiers)
            for idx, model_i in enumerate(self.models):
                if model_i[0] == 'R':
                    # Plot posterior results for RBC classifiers
                    result = posterior[model_i[1:]][self.x_coords[0]:self.x_coords[1],
                             self.y_coords[0]:self.y_coords[1]]
                else:
                    # Plot likelihood results for instantaneous classifiers
                    result = base_model_predicted_class[model_i][self.x_coords[0]:self.x_coords[1],
                             self.y_coords[0]:self.y_coords[1]]
                    #       # Merge the two non-water (vegetation and land) classes in 1
                if Config.test_site == 1:
                    result[np.where(result != 0)] = 1
                else:
                    result[np.where(result!=0)] = -1
                    result = result + np.ones(result.shape)
                self.axarr[result_idx, idx].imshow(result,  colors.ListedColormap(Visual.cmap['multiearth']))  #multiearth map because it is a 2 class problem now
                self.axarr[result_idx, idx].get_yaxis().set_ticks([]), self.axarr[result_idx, idx].get_xaxis().set_ticks([])
                try:
                    # Balanced accuracy
                    from sklearn.metrics import balanced_accuracy_score, f1_score
                    balanced_accuracy = balanced_accuracy_score(y_true=label_image.flatten(), y_pred=result.flatten())
                    f1_score_i = f1_score(y_true=label_image.flatten(), y_pred=result.flatten())
                    # Plot error map
                    error_map = label_image != result
                    self.axarr[result_idx+1, idx].imshow(error_map, colors.ListedColormap(Visual.cmap['error_map']))
                    self.axarr[result_idx+1, idx].get_yaxis().set_ticks([]), self.axarr[result_idx+1, idx].get_xaxis().set_ticks([])
                    # Remove axis
                    for axis in ['top', 'bottom', 'left', 'right']:
                        self.axarr[result_idx, idx].spines[axis].set_linewidth(0)
                        self.axarr[result_idx+1, idx].spines[axis].set_linewidth(0)
                        # self.axarr[result_idx, idx].set_aspect('equal')
                        # self.axarr[result_idx, idx].set_aspect('equal')
                    accuracy = np.floor((1 - np.sum(error_map) / (error_map.shape[0] * error_map.shape[1])) * 10000) / 100
                    #                     # self.axarr[result_idx + 1, idx].set_xlabel(f'{accuracy}')
                    self.results_qa['accuracy'][int(result_idx/2), idx] = accuracy
                    self.results_qa['balanced_accuracy'][int(result_idx / 2), idx] = np.floor((balanced_accuracy*10000))/100
                    self.results_qa['f1'][int(result_idx / 2), idx]  = np.floor((f1_score_i*10000))/100
                except:
                    print('except')

        # Plot plot_figures for Charles River scenario
        elif Config.test_site == 3:
            print("Results for Charles River errormap, Study Area C")
            result_idx = result_idx * 2

            position_label = len(self.models)

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands) * Visual.scaling_rgb[Config.test_site]
            # Plot RGB Image
            self.axarr[result_idx, position_label].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, position_label].axis('off')

            # Plot label
            import imageio
            path_label = "/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/classification/charles_river_3/configuration_paper/figures/charles_river_water_label.tiff"
            label_image = imageio.imread(path_label)
            #label_image = (label_image - np.ones(label_image.shape))*-1
            #plt.figure(), plt.imshow(label_image)
            self.axarr[result_idx + 1, position_label].imshow(label_image, cmap=colors.ListedColormap(Visual.cmap['label']))
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx+1, position_label].spines[axis].set_linewidth(0)
            self.axarr[result_idx + 1, position_label].xaxis.set_label_coords(0, 0.325)
            self.axarr[result_idx+1, position_label].get_yaxis().set_ticks([])
            self.axarr[result_idx+1, position_label].get_xaxis().set_ticks([])

            # Classification results
            # (likelihood for instantaneous classifiers, posterior for RBCs and classification error map for all
            # classifiers)
            for idx, model_i in enumerate(self.models):
                if model_i[0] == 'R':
                    # Plot posterior results for RBC classifiers
                    result = posterior[model_i[1:]][self.x_coords[0]:self.x_coords[1],
                             self.y_coords[0]:self.y_coords[1]]
                else:
                    # Plot likelihood results for instantaneous classifiers
                    result = base_model_predicted_class[model_i][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                # Merge the two non-water (vegetation and land) classes in 1
                if Config.test_site != 3:
                    result[np.where(result!=0)] = -1
                    result = result + np.ones(result.shape)
                self.axarr[result_idx, idx].imshow(result,  colors.ListedColormap(Visual.cmap['multiearth']))  #multiearth map because it is a 2 class problem now
                self.axarr[result_idx, idx].get_yaxis().set_ticks([]), self.axarr[result_idx, idx].get_xaxis().set_ticks([])
                # Balanced accuracy
                from sklearn.metrics import balanced_accuracy_score, f1_score
                balanced_accuracy = balanced_accuracy_score(y_true=label_image.flatten(), y_pred=result.flatten())
                balanced_accuracy = np.floor(
                    (balanced_accuracy * 10000)) / 100
                f1_score_i = f1_score(y_true=label_image.flatten(), y_pred=result.flatten())
                # Plot error map
                error_map = label_image != result
                self.axarr[result_idx+1, idx].imshow(error_map, colors.ListedColormap(Visual.cmap['error_map']))
                self.axarr[result_idx+1, idx].get_yaxis().set_ticks([]), self.axarr[result_idx+1, idx].get_xaxis().set_ticks([])
                # Remove axis
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[result_idx, idx].spines[axis].set_linewidth(0)
                    self.axarr[result_idx+1, idx].spines[axis].set_linewidth(0)
                    # self.axarr[result_idx, idx].set_aspect('equal')
                    # self.axarr[result_idx, idx].set_aspect('equal')
                accuracy = np.floor((1 - np.sum(error_map) / (error_map.shape[0] * error_map.shape[1])) * 10000) / 100
                # self.axarr[result_idx + 1, idx].set_xlabel(f'{accuracy}')
                self.results_qa['accuracy'][int(result_idx / 2), idx] = accuracy
                self.results_qa['balanced_accuracy'][int(result_idx / 2), idx] = balanced_accuracy
                self.results_qa['f1'][int(result_idx / 2), idx] = np.floor((f1_score_i * 10000)) / 100

            for idx, model_i in enumerate(self.models[0:int(len(self.models)/2)]):
                nonrecursive_metric = self.results_qa['balanced_accuracy'][int(result_idx / 2), idx]
                recursive_metric = self.results_qa['balanced_accuracy'][int(result_idx / 2), idx + int(len(self.models)/2)]
                print(model_i, nonrecursive_metric, recursive_metric)

                fontweight = Results.class_fig_settings['font_highest_value']['fontweight']
                fontcolor = Results.class_fig_settings['font_highest_value']['fontcolor']
                if nonrecursive_metric>recursive_metric:
                    self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric, fontweight=fontweight, color=fontcolor)
                    self.axarr[result_idx + 1, idx+ int(len(self.models)/2)].set_xlabel(recursive_metric)
                elif nonrecursive_metric<recursive_metric:
                    self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric)
                    self.axarr[result_idx + 1, idx+ int(len(self.models)/2)].set_xlabel(recursive_metric, fontweight=fontweight, color=fontcolor)
                else:
                    self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric)
                    self.axarr[result_idx + 1, idx+ int(len(self.models)/2)].set_xlabel(recursive_metric)




            if Debug.save_rgb:
                rgb_cut = rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                # Save RGB Image without margin to create label with LabelStudio
                from PIL import Image
                plt.figure()
                plt.imshow(rgb_cut)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig(r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/myfig.png",
                            bbox_inches='tight', pad_inches=0)
                # Create water label (charles_river)
                water_label_1_path = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/charles_river/evaluation/labels/water_label_1.png"
                water_label_2_path  = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/Sentinel2_data/charles_river/evaluation/labels/water_label_2.png"
                water_label_1 = np.array(Image.open(water_label_1_path).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                water_label_2 = np.array(Image.open(water_label_2_path).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
                water_label = water_label_2 + water_label_1
                label_attempt = np.zeros(shape=rgb_cut[:, :, 0].flatten().shape)
                label_attempt[np.where(water_label.flatten() > 100)] = 1
                label = label_attempt.reshape(rgb_cut.shape[0], rgb_cut.shape[1])
                import imageio
                path_label = r"/Users/helena/Documents/RESEARCH/Recursive_Bayesian_Image_Classification/2023nov/evaluation_results/figures/charles_river_water_label.tiff"
                imageio.imwrite(path_label, label)

            plt.subplots_adjust(wspace=0, hspace=0)
        elif Config.test_site == 4:
            print("Results for MultiEarth errormap dataset")
            result_idx = result_idx*2

            # Get RGB Image
            rgb_image = 5*get_rgb_image(image_all_bands=image_all_bands)
            # Plot RGB Image
            self.axarr[result_idx, 6].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, 6].axis('off')
            # self.axarr[result_idx, 6].set_aspect('equal')

            # Show string with acquisition date
            #self.axarr[result_idx, 0].yaxis.set_label_coords(-0.88, -0.3)  # new plot location changed

            # Plot MultiEarth label
            if image_idx in Config.index_quant_analysis[Config.test_site].keys():
                label_idx = Config.index_quant_analysis[Config.test_site][image_idx]
                for file_counter, file_name in enumerate(sorted(os.listdir(Config.path_label_images))):
                    if file_counter == label_idx:
                        path_label_i = os.path.join(Config.path_label_images, file_name)
                        label_image = self.image_reader.read_band(path_label_i)
                self.axarr[result_idx+1, 6].imshow(label_image, cmap=colors.ListedColormap(Visual.cmap['label']))
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[result_idx+1, 6].spines[axis].set_linewidth(0)
                self.axarr[result_idx + 1, 6].xaxis.set_label_coords(0, 0.325)
            self.axarr[result_idx+1, 6].get_yaxis().set_ticks([])
            self.axarr[result_idx+1, 6].get_xaxis().set_ticks([])
            # self.axarr[result_idx+1, 6].set_aspect('equal')


            # Classification results
            # (likelihood for instantaneous classifiers, posterior for RBCs and classification error map for all
            # classifiers)
            for idx, model_i in enumerate(self.models):
                if model_i[0] == 'R':
                    # Plot posterior results for RBC classifiers
                    result = posterior[model_i[1:]][self.x_coords[0]:self.x_coords[1],
                             self.y_coords[0]:self.y_coords[1]]
                else:
                    # Plot likelihood results for instantaneous classifiers
                    result = base_model_predicted_class[model_i][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
                self.axarr[result_idx, idx].imshow(result, self.cmap)
                self.axarr[result_idx, idx].get_yaxis().set_ticks([]), self.axarr[result_idx, idx].get_xaxis().set_ticks([])
                # Balanced Accuracy
                from sklearn.metrics import balanced_accuracy_score, f1_score
                balanced_accuracy = balanced_accuracy_score(y_true=label_image.flatten(), y_pred=result.flatten())
                f1_score_i = f1_score(y_true=label_image.flatten(), y_pred=result.flatten())
                # Plot error map
                error_map = label_image != result
                self.axarr[result_idx+1, idx].imshow(error_map, colors.ListedColormap(Visual.cmap['error_map']))
                self.axarr[result_idx+1, idx].get_yaxis().set_ticks([]), self.axarr[result_idx+1, idx].get_xaxis().set_ticks([])
                # Remove axis
                for axis in ['top', 'bottom', 'left', 'right']:
                    self.axarr[result_idx, idx].spines[axis].set_linewidth(0)
                    self.axarr[result_idx+1, idx].spines[axis].set_linewidth(0)
                    # self.axarr[result_idx, idx].set_aspect('equal')
                    # self.axarr[result_idx, idx].set_aspect('equal')
                accuracy = np.floor((1 - np.sum(error_map) / (error_map.shape[0] * error_map.shape[1])) * 10000) / 100
                # self.axarr[result_idx + 1, idx].set_xlabel(f'{accuracy}')
                self.results_qa['accuracy'][int(result_idx / 2), idx] = accuracy
                self.results_qa['balanced_accuracy'][int(result_idx / 2), idx] = np.floor(
                    (balanced_accuracy * 10000)) / 100
                self.results_qa['f1'][int(result_idx / 2), idx] = np.floor((f1_score_i * 10000)) / 100

            # Legend
            # self.plot_legend = ["SIC", 'GMM', "LR", 'RSIC', 'RGMM', 'RLR', 'RGB/Label']
            # for idx, label in enumerate(self.plot_legend): # new plot
            #     self.axarr[0, idx].title.set_fontfamily('Times New Roman')
            #     self.axarr[0, idx].title.set_fontsize(8)
            #     self.axarr[0, idx].title.set_text(self.plot_legend[idx])
            # Adjust space between subplots
            #plt.subplots_adjust(top=0.437, right=0.776)
            plt.subplots_adjust(wspace=0, hspace=0)
        else:
            print("No plot_figures of this scene appearing in the publication.")
        self.f.show()
