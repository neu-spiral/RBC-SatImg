import logging
import os
import imageio

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np

from image_reader import ReadSentinel2
from configuration import Config, Visual
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

    def calculate_accuracy_metrics(self, label_image: np.ndarray, predicted_image: np.ndarray) -> dict:
        """
        Calculate classification accuracy metrics.

        Parameters
        ----------
        label_image : np.ndarray
            Ground truth labels.
        predicted_image : np.ndarray
            Predicted classification results.

        Returns
        -------
        dict
            Dictionary containing accuracy, balanced_accuracy, and f1_score.
        """
        balanced_accuracy = balanced_accuracy_score(label_image.flatten(), predicted_image.flatten())
        f1 = f1_score(label_image.flatten(), predicted_image.flatten(), average="weighted")
        accuracy = np.mean(label_image == predicted_image) * 100

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy * 100,
            "f1_score": f1 * 100
        }

    def add_metrics_to_plot(self, metrics: dict, result_idx: int, model_idx: int):
        """
        Add accuracy metrics to the plots.

        Parameters
        ----------
        metrics : dict
            Dictionary containing accuracy metrics.
        result_idx : int
            Index to place the results in the plot grid.
        model_idx : int
            Index of the model being processed.
        """
        fontweight = Visual.class_fig_settings['font_highest_value']['fontweight']
        fontcolor = Visual.class_fig_settings['font_highest_value']['fontcolor']
        fontsize = Visual.class_fig_settings['fontsize'][Config.test_site]

        balanced_accuracy = metrics['balanced_accuracy']

        self.axarr[result_idx + 1, model_idx].set_xlabel(
            f"{balanced_accuracy:.2f}%", fontweight=fontweight, color=fontcolor, fontsize=fontsize
        )

    def load_label_image(self, image_idx: int) -> np.ndarray:
        """
        Load ground truth labels for the given image index.

        Parameters
        ----------
        image_idx : int
            Index of the image being processed.

        Returns
        -------
        np.ndarray or None
            Ground truth label image or None if the label is not found.
        """
        if Config.test_site in ['2', '1a', '1b']:
            path_label = os.path.join(Config.path_zenodo, 'RBC-WatData', f"labels_{Config.scenario}_{Config.test_site}",
                                      f"label_{image_idx}.tiff")
            if not os.path.exists(path_label):
                logging.warning(f"Label file not found: {path_label}")
                return None
            return imageio.imread(path_label)
        elif Config.test_site == '3':
            path_labels = os.path.join(Config.path_sentinel_images, Config.scenario, "labels")
            label_idx = Config.qa_settings['index_quant_analysis'][Config.test_site][image_idx]
            for file_counter, file_name in enumerate(sorted(os.listdir(path_labels))):
                if file_counter == label_idx:
                    full_path = os.path.join(path_labels, file_name)
                    if not os.path.exists(full_path):
                        logging.warning(f"Label file not found: {full_path}")
                        return None
                    return self.image_reader.read_band(full_path)
        return None

    def plot_classification_results(self, image_idx: int, image_all_bands: np.ndarray,
                                    base_model_predicted_class: dict, posterior: dict, result_idx: int):
        """
        Plot classification results including RGB images, label images, classification maps, and error maps.

        Parameters
        ----------
        image_idx : int
            Index of the image being processed.
        image_all_bands : np.ndarray
            Array containing all bands of the image.
        base_model_predicted_class : dict
            Dictionary containing base model predictions.
        posterior : dict
            Dictionary containing posterior probabilities for recursive models.
        result_idx : int
            Row index to place the results in the plot grid.
        """
        # Calculate the starting rows for classification/error maps
        classification_row = result_idx * 2
        error_map_row = classification_row + 1

        # Plot RGB image in the last column of the classification row
        rgb_image = get_rgb_image(image_all_bands=image_all_bands)
        rgb_image *= Visual.scaling_rgb[Config.test_site]
        ax_rgb = self.axarr[classification_row, -1]
        ax_rgb.imshow(rgb_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]])
        ax_rgb.set_xticks([]), ax_rgb.set_yticks([])
        for spine in ax_rgb.spines.values():
            spine.set_visible(False)

        # Plot label image (or black if not available) in the last column of the error map row
        label_image = self.load_label_image(image_idx)
        ax_label = self.axarr[error_map_row, -1]
        if label_image is None:
            black_image = np.zeros((self.x_coords[1] - self.x_coords[0], self.y_coords[1] - self.y_coords[0]))
            ax_label.imshow(black_image, cmap="gray")
        else:
            ax_label.imshow(label_image, cmap=colors.ListedColormap(Visual.cmap['label']))
        ax_label.set_xticks([]), ax_label.set_yticks([])
        for spine in ax_label.spines.values():
            spine.set_visible(False)

        # Plot classification maps and error maps for each model
        for idx, model_name in enumerate(self.models):
            if model_name.startswith("R"):
                predicted_image = posterior[model_name[1:]]
            else:
                predicted_image = base_model_predicted_class[model_name]

            # Crop the predicted image for display
            cropped_prediction = predicted_image[self.x_coords[0]:self.x_coords[1], self.y_coords[0]:self.y_coords[1]]

            # Classification map in classification_row
            ax_class = self.axarr[classification_row, idx]
            ax_class.imshow(cropped_prediction, cmap=self.cmap)
            ax_class.set_xticks([]), ax_class.set_yticks([])
            for spine in ax_class.spines.values():
                spine.set_visible(False)

            # Error map in error_map_row
            ax_error = self.axarr[error_map_row, idx]
            if label_image is None:
                # If no label is available, plot a black error map
                black_image = np.zeros(cropped_prediction.shape)
                ax_error.imshow(black_image, cmap="gray")
            else:
                error_map = cropped_prediction != label_image[self.x_coords[0]:self.x_coords[1],
                                                  self.y_coords[0]:self.y_coords[1]]
                ax_error.imshow(error_map, cmap=colors.ListedColormap(Visual.cmap['error_map']))
            ax_error.set_xticks([]), ax_error.set_yticks([])
            for spine in ax_error.spines.values():
                spine.set_visible(False)

            # Add metrics to the error map row with an offset
            if label_image is not None:
                metrics = self.calculate_accuracy_metrics(label_image, predicted_image)
                self.results_qa['balanced_accuracy'][result_idx // 2, idx] = metrics['balanced_accuracy']
                fontweight = Visual.class_fig_settings['font_highest_value']['fontweight']
                fontcolor = Visual.class_fig_settings['font_highest_value']['fontcolor']
                fontsize = Visual.class_fig_settings['fontsize'][Config.test_site]

                # Set the metrics as x-labels for the error map row
                ax_error.set_xlabel(
                    f"{metrics['balanced_accuracy']:.2f}%", fontweight=fontweight, color=fontcolor, fontsize=fontsize
                )

    def process_and_plot_results(self, image_idx: int, image_all_bands: np.ndarray,
                                 base_model_predicted_class: dict, posterior: dict, result_idx: int):
        """
        Process results, calculate accuracy metrics, and plot classification results.

        This function handles the visualization of RGB images, classification maps, error maps,
        and accuracy metrics for a given image. Accuracy metrics are calculated and added to
        the error map plots as labels for easy reference.

        Parameters
        ----------
        image_idx : int
            Index of the image being processed.
        image_all_bands : np.ndarray
            Array containing all bands of the image.
        base_model_predicted_class : dict
            Dictionary containing base model predictions.
        posterior : dict
            Dictionary containing posterior probabilities for recursive models.
        result_idx : int
            Index to place the results in the plot grid.

        Returns
        -------
        None
        """

        # Load the ground truth label
        label_image = self.load_label_image(image_idx)

        # Plot classification results
        self.plot_classification_results(image_idx, image_all_bands, base_model_predicted_class, posterior, result_idx)

        # If label does not exist, skip metrics calculation
        if label_image is None:
            logging.warning(f"Skipping accuracy metrics calculation for image {image_idx} as label is missing.")
            return

        # Calculate accuracy metrics and store them
        for idx, model_name in enumerate(self.models):
            if model_name.startswith("R"):
                predicted_image = posterior[model_name[1:]]
            else:
                predicted_image = base_model_predicted_class[model_name]

            metrics = self.calculate_accuracy_metrics(label_image, predicted_image)
            self.results_qa['balanced_accuracy'][result_idx // 2, idx] = metrics['balanced_accuracy']

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