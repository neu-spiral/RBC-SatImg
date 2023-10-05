import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from image_reader import ReadSentinel2
from configuration import Config, Debug
from matplotlib import colors
from figures import get_rgb_image


class ClassificationResultsFigure:
    def __init__(self):
        """ Initializes instance of RBC object with the corresponding class attributes.

        """
        self.cmap = colors.ListedColormap(Config.cmap[Config.scenario])
        self.image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                                          Config.image_dimensions[Config.scenario]['dim_y'])
        self.x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
        self.y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']

    def create_quantitative_results_figure(self):
        """ Creates figure to show quantitative classification results.

        """
        self.f, self.axarr = plt.subplots(len(Config.index_quant_analysis), 13, figsize=(15, 15))

    def get_quantitative_results(self, label_image: np.ndarray, class_labels: np.ndarray):
        classification_map = (np.abs(class_labels - label_image) - 1)*(-1) # 1 means a match
        matched_pixel_percentage = np.sum(classification_map)/(classification_map.shape[0]*classification_map.shape[1])
        return classification_map, matched_pixel_percentage

    def create_figure(self):
        """ Creates figure to show qualitative classification results.

        """
        self.index_results_to_plot = Config.index_images_to_plot[Config.scene_id]
        self.n_results_to_plot = len(self.index_results_to_plot)
        if self.n_results_to_plot > 15:
            self.big_figure = True
        else:
            self.big_figure = False

        # Plot classification results for Oroville Dam scenario
        if Config.scene_id == 1 or Config.scene_id == 2:
            print(f"Results for Oroville Dam, k = {Config.scene_id}")

            # Vector with models
            self.plot_legend = ['SIC', 'GMM', 'LR', 'DWM', 'WN', 'RSIC', 'RGMM', 'RLR', 'RDWM', 'RWN', 'RGB']
            self.models = ['Scaled Index', 'GMM', 'Logistic Regression']  # TODO: Use this vector to clean code

            # Create figure (size changes depending on the amount of plotted images)
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 11, figsize=(9, 30))
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 11, figsize=(9, 10))

        # Plot plot_results for Charles River scenario
        elif Config.scene_id == 3:
            print("Results for Charles River, Study Area C")

            # Create figure (size changes  depending on the amount of plotted images)
            if self.big_figure == False:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 7, figsize=(9, 10))
            else:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.f, self.axarr = plt.subplots(self.n_results_to_plot, 7, figsize=(9, 30))

            # Vectors with models to plot
            self.plot_legend = ['SIC', 'GMM', 'LR', 'RSIC', 'RGMM', 'RLR', 'RGB']
            self.models = ['Scaled Index', 'GMM', 'Logistic Regression']  # TODO: Use this vector to clean code

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
            # plt.subplot_tool()
            # plt.show()
        elif Config.scene_id == 4:
            print("Results for MultiEarth Dataset")

            # Create figure
            self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(5, 50))
            self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(5, 7)) # small one only quantitative
            self.f, self.axarr = plt.subplots(self.n_results_to_plot, 8, figsize=(10, 35))  # full results

            # Vectors with models to plot
            self.plot_legend = ['Acq. date      SIC', 'GMM', 'LR', 'RSIC', 'RGMM', 'RLR', 'RGB', 'Label     Label Date']
            self.models = ['Scaled Index', 'GMM', 'Logistic Regression']  # TODO: Use this vector to clean code

            # Adjust space between subplots
            plt.subplots_adjust(top=0.437, right=0.776)
        else:
            print("No plot_results of this scene appearing in the publication.")

    def plot_stored_results(self):
        """ Plots all the results that have been stored with the corresponding configuration.

        """
        # Path with results
        path_results = os.path.join(Config.path_evaluation_results, "classification",
                                    f'{Config.scenario}_{Config.scene_id}')
        path_images = os.path.join(Config.path_sentinel_images, f'{Config.scenario}', 'evaluation')

        # Plot classification results for Oroville Dam scenario
        if Config.scene_id == 1 or Config.scene_id == 2:
            print(f"Results for Oroville Dam, k = {Config.scene_id}")

            for image_i in range(0, self.n_results_to_plot):
                # Read image again to be able to get RGB image
                image_all_bands, date_string = self.image_reader.read_image(path=path_images,
                                                                            image_idx=self.index_results_to_plot[
                                                                                image_i])

                # Get RGB Image
                rgb_image = get_rgb_image(image_all_bands=image_all_bands)
                # Scaling to increase illumination
                rgb_image = rgb_image * Config.scaling_rgb[Config.scene_id]

                # Read stored evaluation plot_results to reproduce published figure
                pickle_file_path = os.path.join(path_results, f"oroville_dam_{Config.scene_id}_image_{self.index_results_to_plot[image_i]}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pkl")
                [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_results
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

        # Plot plot_results for Charles River scenario
        elif Config.scene_id == 3:
            print("Results for Charles River, Study Area C")

            for image_i in range(0, self.n_results_to_plot):

                # Read image again to be able to get RGB image
                image_all_bands, date_string = self.image_reader.read_image(path=path_images,
                                                                            image_idx=self.index_results_to_plot[
                                                                                image_i])

                # Get RGB Image
                rgb_image = get_rgb_image(image_all_bands=image_all_bands)

                # Read stored evaluation plot_results to reproduce published figure
                pickle_file_path = os.path.join(path_results,
                                                f"charles_river_{Config.scene_id}_image_{self.index_results_to_plot[image_i]}_epsilon_{Config.eps}.pkl")
                [predicted_class, posterior] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_results
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
            print("No plot_results of this scene appearing in the publication.")

        # Save figure as pdf
        if Debug.save_figures:
            plt.savefig(os.path.join(Config.path_results_figures,
                                     f'classification_{Config.scenario}_{Config.scene_id}_epsilon_{Config.eps}_norm_constant_{Config.norm_constant}.pdf'),
                        format="pdf", bbox_inches="tight", dpi=200)

    def plot_results_one_date(self, image_idx: int, image_all_bands: np.ndarray, date_string: str,
                              base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int):
    #use the following line if wanting to use the predicted probabilities (besides the predicted classes)
    #def plot_results_one_date(self, image_idx: int, image_all_bands: np.ndarray, date_string: str, base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int, base_model_predicted_probabilities: np.ndarray):
        """ Plots all the results that have been stored with the corresponding configuration.

        """

        # Plot classification results for Oroville Dam scenario
        if Config.scene_id == 1 or Config.scene_id == 2:
            print(f"Results for Oroville Dam, k = {Config.scene_id}")

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands)
            # Scaling to increase illumination
            rgb_image = rgb_image * Config.scaling_rgb[Config.scene_id]

            # Plot plot_results
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

        # Plot plot_results for Charles River scenario
        elif Config.scene_id == 3 or Config.scene_id == 4:
            print("Results for Charles River, Study Area C")

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
            if image_idx in Config.index_quant_analysis.keys():
                label_idx = Config.index_quant_analysis[image_idx]
                path_label_images = os.path.join(Config.path_sentinel_images, 'deforestation_labels')# index of the label we want to use for comparison
                # label_image, date_string_label = image_reader.read_image(path=path_label_images, image_idx=label_idx)
                for file_counter, file_name in enumerate(sorted(os.listdir(path_label_images))):
                    if file_counter == label_idx:
                        path_label_i = os.path.join(path_label_images, file_name)
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

            self.plot_legend = ['SIC', 'GMM', 'LR', 'RSIC', 'RGMM', 'RLR', 'RGB', 'Label']
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
            print("No plot_results of this scene appearing in the publication.")
        self.f.show()
