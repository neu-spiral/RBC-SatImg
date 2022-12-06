import pickle
import os

import matplotlib.pyplot as plt

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

    def create_figure(self):
        """ Creates figure to show classification results.

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

            # Create figure (size changes depending on the amount of plotted images)
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
                [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_results
                self.axarr[image_i, 0].imshow(
                    y_pred["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
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
                    y_pred["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 1].axis('off')
                self.axarr[image_i, 2].imshow(
                    y_pred["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 2].axis('off')
                self.axarr[image_i, 3].imshow(
                    y_pred["DeepWaterMap"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 3].axis('off')
                self.axarr[image_i, 4].imshow(
                    y_pred["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 4].axis('off')
                self.axarr[image_i, 5].imshow(predicted_image["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap, aspect='auto')
                self.axarr[image_i, 5].axis('off')
                self.axarr[image_i, 6].imshow(
                    predicted_image["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap,
                    aspect='auto')
                self.axarr[image_i, 6].axis('off')
                self.axarr[image_i, 7].imshow(
                    predicted_image["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                    self.y_coords[0]:self.y_coords[1]], self.cmap,
                    aspect='auto')
                self.axarr[image_i, 7].axis('off')
                self.axarr[image_i, 8].imshow(predicted_image["DeepWaterMap"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap, aspect='auto')
                self.axarr[image_i, 8].axis('off')
                self.axarr[image_i, 9].imshow(
                    predicted_image["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
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
                [y_pred, predicted_image] = pickle.load(open(pickle_file_path, 'rb'))
                print(pickle_file_path)

                # Plot plot_results
                self.axarr[image_i, 0].imshow(
                    y_pred["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
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
                    y_pred["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
                self.axarr[image_i, 1].axis('off')
                self.axarr[image_i, 2].imshow(
                    y_pred["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap)
                self.axarr[image_i, 2].axis('off')
                self.axarr[image_i, 3].imshow(predicted_image["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                              self.y_coords[0]:self.y_coords[1]],
                                              self.cmap)
                self.axarr[image_i, 3].axis('off')
                self.axarr[image_i, 4].imshow(
                    predicted_image["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                    self.cmap)
                self.axarr[image_i, 4].axis('off')
                self.axarr[image_i, 5].imshow(
                    predicted_image["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
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

    def plot_results_one_date(self, image_idx, image_all_bands, date_string, y_pred, predicted_image, result_idx):
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
                y_pred["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
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
                y_pred["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 1].axis('off')
            self.axarr[result_idx, 2].imshow(
                y_pred["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 2].axis('off')
            self.axarr[result_idx, 3].imshow(
                y_pred["DeepWaterMap"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 3].axis('off')
            self.axarr[result_idx, 4].imshow(
                y_pred["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 4].axis('off')
            self.axarr[result_idx, 5].imshow(predicted_image["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap, aspect='auto')
            self.axarr[result_idx, 5].axis('off')
            self.axarr[result_idx, 6].imshow(
                predicted_image["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap,
                aspect='auto')
            self.axarr[result_idx, 6].axis('off')
            self.axarr[result_idx, 7].imshow(
                predicted_image["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                self.y_coords[0]:self.y_coords[1]], self.cmap,
                aspect='auto')
            self.axarr[result_idx, 7].axis('off')
            self.axarr[result_idx, 8].imshow(predicted_image["DeepWaterMap"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap, aspect='auto')
            self.axarr[result_idx, 8].axis('off')
            self.axarr[result_idx, 9].imshow(
                predicted_image["WatNet"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
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
        elif Config.scene_id == 3:
            print("Results for Charles River, Study Area C")

            # Get RGB Image
            rgb_image = get_rgb_image(image_all_bands=image_all_bands)

            # Plot plot_results
            self.axarr[result_idx, 0].imshow(
                y_pred["Scaled Index"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 0].get_yaxis().set_ticks([])
            self.axarr[result_idx, 0].get_xaxis().set_ticks([])

            # Remove axis
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, 0].spines[axis].set_linewidth(0)

                self.axarr[result_idx, 0].set_ylabel(date_string, rotation=0, fontsize=11.7, fontfamily='Times New Roman')

            # The label position must be changed accordingly, considering the amount of images plotted
            if self.big_figure:
                # if wanting to plot all images (all dates are plotted, like in the end of the manuscript arxiv version)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.8, 0.25)
            else:
                # if wanting to plot the same images plotted in the manuscript (1 every four dates are selected)
                self.axarr[result_idx, 0].yaxis.set_label_coords(-0.7, 0.25)

            self.axarr[result_idx, 1].imshow(
                y_pred["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 1].axis('off')
            self.axarr[result_idx, 2].imshow(
                y_pred["Logistic Regression"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 2].axis('off')
            self.axarr[result_idx, 3].imshow(predicted_image["Scaled Index"][self.x_coords[0]:self.x_coords[1],
                                          self.y_coords[0]:self.y_coords[1]],
                                          self.cmap)
            self.axarr[result_idx, 3].axis('off')
            self.axarr[result_idx, 4].imshow(
                predicted_image["GMM"][self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]],
                self.cmap)
            self.axarr[result_idx, 4].axis('off')
            self.axarr[result_idx, 5].imshow(
                predicted_image["Logistic Regression"][self.x_coords[0]:self.x_coords[1],
                self.y_coords[0]:self.y_coords[1]], self.cmap)
            self.axarr[result_idx, 5].axis('off')
            self.axarr[result_idx, 6].imshow(
                rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
            self.axarr[result_idx, 6].axis('off')

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
        self.f.show()
