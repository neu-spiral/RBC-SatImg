import os
import imageio

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from PIL import Image
import numpy as np

from image_reader import ReadSentinel2
from configuration import Config, Debug, Visual
from matplotlib import colors
from figures import get_rgb_image



class ClassificationFigure:

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
        self.plot_legend = Visual.qa_fig_settings['legend_models'][Config.test_site]
        self.models = Config.qa_settings['models'][Config.test_site]
        self.n_total_models = len(self.models)
        self.n_inst_models = int(self.n_total_models / 2)

        # Dates/Images plotted (plot matrix rows)
        self.index_results_to_plot = Config.index_images_to_plot[Config.test_site]
        self.n_results_to_plot = len(self.index_results_to_plot) * 2  # multiply by 2 because we show classification map + error map

        # Initialize Quantitative Analysis vector
        self.results_qa = dict()
        dim_x = int(self.n_results_to_plot / 2)  # we divide by 2 because self.n_results_to_plot includes 2 per deate (classification map + error classification map)
        for acc in Config.qa_settings['metrics']:
            self.results_qa[acc] = np.ndarray(shape=(dim_x, len(self.models) + 1))

        # Create figure
        self.f, self.axarr = plt.subplots(self.n_results_to_plot, len(self.models) + 1, figsize=(9, 10))

    def plot_results(self, image_idx: int, image_all_bands: np.ndarray,
                              base_model_predicted_class: np.ndarray, posterior: np.ndarray, result_idx: int):
        """ Plots all the results that have been stored with the corresponding configuration.

        """

        #
        result_idx = result_idx * 2
        last_col = self.n_total_models

        #
        # Plot RGB Image
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)
        rgb_image = rgb_image * Visual.scaling_rgb[Config.test_site]
        self.axarr[result_idx, last_col].imshow(
            rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
        self.axarr[result_idx, last_col].axis('off')

        #
        # Plot Label
        if Config.test_site == '2':
            path_label = os.path.join(Config.path_sentinel_images, Config.scenario, f"labels_{Config.test_site}", "charles_river_water_label.tiff")
            label_image = imageio.imread(path_label)
        elif Config.test_site in ['1a', '1b']:
            path_label = os.path.join(Config.path_sentinel_images, Config.scenario, f"labels_{Config.test_site}", f"label_{image_idx}.tiff")
            label_image = imageio.imread(path_label)
        elif Config.test_site == '3':
            path_labels = os.path.join(Config.path_sentinel_images, Config.scenario, f"labels")
            label_idx = Config.qa_settings['index_quant_analysis'][Config.test_site][image_idx]
            for file_counter, file_name in enumerate(sorted(os.listdir(path_labels))):
                if file_counter == label_idx:
                    path_label_i = os.path.join(path_labels, file_name)
                    label_image = self.image_reader.read_band(path_label_i)
        self.axarr[result_idx + 1, last_col].imshow(label_image,
                                                          cmap=colors.ListedColormap(Visual.cmap['label']))
        for axis in ['top', 'bottom', 'left', 'right']:
            self.axarr[result_idx + 1, last_col].spines[axis].set_linewidth(0)
        self.axarr[result_idx + 1, last_col].xaxis.set_label_coords(0, 0.325)
        self.axarr[result_idx + 1, last_col].get_yaxis().set_ticks([])
        self.axarr[result_idx + 1, last_col].get_xaxis().set_ticks([])

        # TODO: cmap call can be replaced by self.cmap
        #
        # Plot Classification Map + Classification Error Map
        # We plot:
        #   (1) - likelihood in the case of instantaneous classifiers (first row, first half columns)
        #   (2) - posterior in the case of RBCs (first row, second half columns)
        #   (3) - classification error map for all classifiers (second row, all columns)
        for idx, model_i in enumerate(self.models):
            if model_i[0] == 'R':
                # Plot posterior results for RBC classifiers
                result = posterior[model_i[1:]][self.x_coords[0]:self.x_coords[1],
                         self.y_coords[0]:self.y_coords[1]]
            else:
                # Plot likelihood results for instantaneous classifiers
                result = base_model_predicted_class[model_i][self.x_coords[0]:self.x_coords[1],
                         self.y_coords[0]:self.y_coords[1]]

            # Scale result to compare with label accordingly
            if Config.test_site in [ '?']:
                # result[np.where(result != 0)] = -1
                # result = result + np.ones(result.shape)
                result = result * -1
            elif Config.test_site in ['1b']:
                result[np.where(result != 0)] = -1
                result = result + np.ones(result.shape)
            # elif Config.test_site in ['2']:
            #     result = result*-1

            # Plot result (classification map)
            if Config.test_site == '1b':
                self.axarr[result_idx, idx].imshow(result*-1, colors.ListedColormap(
                    Visual.cmap[Config.scenario]))
            else:
                self.axarr[result_idx, idx].imshow(result, colors.ListedColormap(
                    Visual.cmap[Config.scenario]))
            self.axarr[result_idx, idx].get_yaxis().set_ticks([]), self.axarr[
                result_idx, idx].get_xaxis().set_ticks([]) # remove ticks

            # Plot error map
            error_map = label_image != result
            self.axarr[result_idx + 1, idx].imshow(error_map, colors.ListedColormap(Visual.cmap['error_map']))
            self.axarr[result_idx + 1, idx].get_yaxis().set_ticks([]), self.axarr[
                result_idx + 1, idx].get_xaxis().set_ticks([])

            from sklearn.metrics import confusion_matrix
            conf_mat = confusion_matrix(y_true=label_image.flatten(), y_pred=result.flatten())

            # Quantitative Analysis (QA)
            # Calculate and store QA metrics
            balanced_accuracy = balanced_accuracy_score(y_true=label_image.flatten(), y_pred=result.flatten())
            balanced_accuracy = np.floor(
                (balanced_accuracy * 10000)) / 100
            f1_score_i = f1_score(y_true=label_image.flatten(), y_pred=result.flatten())
            accuracy = np.floor((1 - np.sum(error_map) / (error_map.shape[0] * error_map.shape[1])) * 10000) / 100
            #self.results_qa['accuracy'][int(result_idx / 2), idx] = accuracy
            self.results_qa['balanced_accuracy'][int(result_idx / 2), idx] = balanced_accuracy
            #self.results_qa['f1'][int(result_idx / 2), idx] = np.floor((f1_score_i * 10000)) / 100

            # Remove axis
            for axis in ['top', 'bottom', 'left', 'right']:
                self.axarr[result_idx, idx].spines[axis].set_linewidth(0)
                self.axarr[result_idx + 1, idx].spines[axis].set_linewidth(0)

        #
        # Plot selected metric result in x_label
        main_metric = Config.qa_settings['main_metric']
        for idx, model_i in enumerate(self.models[0:int(len(self.models) / 2)]):
            nonrecursive_metric = self.results_qa[main_metric][int(result_idx / 2), idx]
            recursive_metric = self.results_qa[main_metric][
                int(result_idx / 2), idx + int(len(self.models) / 2)]
            print(model_i, nonrecursive_metric, recursive_metric)

            fontweight = Visual.class_fig_settings['font_highest_value']['fontweight']
            fontcolor = Visual.class_fig_settings['font_highest_value']['fontcolor']
            fontsize = Visual.class_fig_settings['fontsize'][Config.test_site]
            if nonrecursive_metric > recursive_metric:
                self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric, fontweight=fontweight,
                                                           color=fontcolor, fontsize=fontsize)
                self.axarr[result_idx + 1, idx + int(len(self.models) / 2)].set_xlabel(recursive_metric, fontsize=fontsize)
            elif nonrecursive_metric < recursive_metric:
                self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric, fontsize=fontsize)
                self.axarr[result_idx + 1, idx + int(len(self.models) / 2)].set_xlabel(recursive_metric,
                                                                                       fontweight=fontweight,
                                                                                       color=fontcolor, fontsize=fontsize)
            else:
                self.axarr[result_idx + 1, idx].set_xlabel(nonrecursive_metric, fontsize=fontsize)
                self.axarr[result_idx + 1, idx + int(len(self.models) / 2)].set_xlabel(recursive_metric, fontsize=fontsize)

        # TODO: Clean this part of code by creating a label generation module
        if Config.label_generation_settings['save_rgb']:
            rgb_cut = rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]
            plt.figure()
            plt.imshow(rgb_cut)
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            path_rgb_prelabel = os.path.join(Config.path_sentinel_images, Config.scenario, "labels",
                                      "charles_river_water_label.tiff")
            plt.savefig(path_rgb_prelabel,
                bbox_inches='tight', pad_inches=0)

        # TODO: Clean this part of code by creating a label generation module
        if Config.label_generation_settings['generate_label']:

            # Create Label
            path_labelstudio_label = ""
            labelstudio_label = np.array(Image.open(path_labelstudio_label).resize([rgb_cut.shape[1], rgb_cut.shape[0]]))
            label = np.zeros(shape=rgb_cut[:, :, 0].flatten().shape)
            label[np.where(labelstudio_label.flatten() > 100)] = 1
            label = label.reshape(rgb_cut.shape[0], rgb_cut.shape[1])

            # Write Label
            path_label = os.path.join(Config.path_sentinel_images, Config.scenario, "labels",
                                      "charles_river_water_label.tiff")
            imageio.imwrite(path_label, label)

        plt.subplots_adjust(wspace=0, hspace=0)
        self.f.show()

    def adjust_figure(self):
        set = Visual.class_fig_settings[Config.test_site]
        self.f.subplots_adjust(wspace=set['tuned_wspace'], hspace=set['tuned_hspace'], top=1,
                                         right=1, left=0, bottom=0)
        n_cols = len(self.models) + 1
        for ax in self.f.axes:
            if hasattr(ax, 'get_subplotspec'):
                ss = ax.get_subplotspec()
                row, col = ss.num1 // n_cols, ss.num1 % n_cols

                # Error Classification Map/Label row
                if (row % 2 == 1):
                    x0_lower, y0_lower, width_lower, height_lower = ss.get_position(
                        self.f).bounds

                    # Add vertical space every 2 rows
                    if row != 1:
                        y0_lower = y0_upper - set['dist_aux'] + set['height_image']
                        ax.set_position(pos=[x0_lower, y0_lower, width_lower, height_lower])
                    else:
                        y0_lower = y0_upper - set['dist_aux'] + set['height_image']
                        ax.set_position(pos=[x0_lower, y0_lower, width_lower, height_lower])

                    # Add horizontal space to separate between nonrecursive-recursive-Label
                    if (col // int(len(self.models) / 2) > 0 and col != 0):
                        ax.set_position(
                            pos=[x0_lower + set['dist_separation'],
                                 y0_lower, width_lower, height_lower])
                    if col == int(len(self.models)):
                        ax.set_position(
                            pos=[x0_lower + set['dist_separation'] * 2,
                                 y0_lower, width_lower, height_lower])

                # Results/RGB Image row
                elif (row % 2 == 0):  # lower-half row (all subplots)
                    x0_upper, y0_upper, width_upper, height_upper = ss.get_position(
                        self.f).bounds

                    # Add vertical space every 2 rows
                    if row != 0:
                        y0_upper = y0_lower - set['dist_aux']
                        ax.set_position(pos=[x0_upper, y0_upper, width_upper, height_upper])
                    # else:
                    #     y0_upper = y0_upper - y0_upper
                    #     ax.set_position(pos=[x0_upper, y0_upper, width_upper, height_upper])

                    # Add horizontal space to separate between nonrecursive-recursive-RGB
                    if (col // int(len(self.models) / 2) > 0 and col != 0):
                        ax.set_position(
                            pos=[x0_upper + set['dist_separation'],
                                 y0_upper, width_upper, height_upper])
                    if col == int(len(self.models)):
                        ax.set_position(
                            pos=[x0_upper + set['dist_separation'] * 2,
                                 y0_upper, width_upper, height_upper])