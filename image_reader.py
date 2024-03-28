import os
import logging

import numpy as np
import skimage
import pickle

from osgeo import gdal
from skimage import transform
from abc import ABC, abstractmethod
from configuration import Config
from tools.path_operations import get_path_image


class ImageReader(ABC):
    """ Abstract class ImageReader. All compatible image readers must inherit from
    this class and implement the abstract methods.

    Abstract Methods
    ----------------
    read_band(self, path_band)
    read_image(self, path, image_idx)

    """
    def __init__(self):
        """ Initialize an abstract ImageReader.

        """
        super().__init__()
        logging.debug("Creating instance of Image Reader object...")

    @abstractmethod
    def read_band(self, path_band: str):
        """ Abstract method to read the band of an image by knowing its path.

        Parameters
        ----------
        path_band : str
            path of the band to be read

        Returns
        -------
        None (Abstract method)

        """

    @abstractmethod
    def read_image(self, path: str, image_idx: int):
        """ Abstract method to read the bands of the image with index *image_idx*.

        Parameters
        ----------
        path : str
            path that includes the folders with the available bands
        image_idx : int
            index linked to the image to be read, if sorting by file_name in ascending order, and only
            considering the images of type and file extension specified in the configuration file.

        Returns
        -------
        None (Abstract method)

        """


class ReadSentinel2(ImageReader):
    """ Reads Sentinel2 images. The bands to be read must be specified in the configuration
    file.

    Class Attributes
    ----------
    dim_x : int
        horizontal dimension of images to be read
    dim_y : int
        vertical dimension of images to be read
    """

    def __init__(self, dim_x: int, dim_y: int):
        """ Initializes instance of ReadSentinel2 object with the corresponding
        class attributes.

        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        if Config.scenario == "multiearth" and Config.shifting_factor_available:
            Config.shifting_factor_evaluation = pickle.load(open(os.path.join(Config.path_sentinel_images, Config.scenario, 'shifting_factor_evaluation.pkl'), 'rb'))
            Config.shifting_factor_training = pickle.load(open(os.path.join(Config.path_sentinel_images, Config.scenario, 'shifting_factor_training.pkl'), 'rb'))

    def read_band(self, path_band: str):
        """ Reads and returns the band with path *path_band*. Each image is linked to
        one date, and for each date there are multiple available bands.

        Parameters
        ----------
        path_band : str
            path of the band to be read

        Returns
        -------
        band : np.ndarray
            read band

        """
        band = gdal.Open(path_band).ReadAsArray()
        #print(f'Mean Value Band {np.mean(band)}')
        # Check if the image dimensions are the proper ones
        if band.shape != (self.dim_x, self.dim_y):
            band = transform.resize(band, (self.dim_x, self.dim_y), anti_aliasing=True, preserve_range=True)
            #print('issue')
        return band

    def read_image(self, path: str, image_idx: int, shift_option: str = None):
        """ Reads and returns all the bands for the image with index *image_idx*.

        Parameters
        ----------
        path : str
            path including the folders with the available bands
        image_idx : int
            index linked to the image to be read, if sorting by file_name in ascending order, and only
            considering the images of type and file extension specified in the configuration file.
        shift_option : str
            if "training" or "evaluation", applies pixel shifting corresponding to the input image_idx

        Returns
        -------
        image : np.ndarray
            all the read bands for the image with index *image_idx*
        date_string: str
            string with the date of read image

        """
        # Empty list
        image_all_bands = []

        # Loop through bands
        # Each band sub-folder must contain the image linked to image_idx
        # taken at the corresponding frequency band.
        if Config.scenario == 'multiearth':
            file_extension = '.tiff'
        else:
            file_extension = '.tif'
        for band_id in Config.bands_to_read:
            # Get path of image to be read in this iteration, which depends on the image index
            path_band_folder = os.path.join(path, f'Sentinel2_B{band_id}')
            path_image_band, date_string = get_path_image(path_folder=path_band_folder,
                                             image_type=Config.image_types[Config.scenario],
                                             file_extension=file_extension, image_index=image_idx, band_id=band_id)

            # Read the corresponding band
            image_band = self.read_band(path_band=path_image_band)

            # Add the read band to the *image_all_bands* array
            # Image is flattened and stored as an array (no image dimensions)
            image_all_bands.extend([image_band.flatten()])

        # Transpose for proper dimensions
        image_all_bands = np.transpose(image_all_bands)

        # Scale images
        image_all_bands = image_all_bands * Config.scaling_factor_sentinel

        # Apply image shifting if required
        if shift_option == "training" or shift_option == "evaluation":
            image_all_bands = self.shift_image(image=image_all_bands, image_idx=image_idx, shift_option=shift_option)
        return image_all_bands, date_string

    def shift_image(self, image: np.ndarray, image_idx: int, shift_option: str):
        """ Shifts image pixel values according to previously stored shifting factor and image index.

        Parameters
        ----------
        image : np.ndarray
            all the read bands for the image with index *image_idx*
        image_idx : int
            index linked to the image to be read, if sorting by file_name in ascending order, and only
            considering the images of type and file extension specified in the configuration file.
        shift_option : str
            if "training" or "evaluation", applies pixel shifting corresponding to the input image_idx

        Returns
        -------
        shifted_image : np.ndarray
            image after shifting operation

        """
        if shift_option == "training":
            shifting_vector = Config.shifting_factor_training[image_idx]
        else: #shift_option == "evaluation"
            shifting_vector = Config.shifting_factor_evaluation[image_idx]
        shifted_image = np.ndarray(image.shape)
        for idx, band_i in enumerate(Config.bands_to_read):
            hist_offset = shifting_vector[idx]
            print(shifting_vector[idx])
            shifted_image[:, idx] = image[:, idx] + hist_offset
        return shifted_image

"""
    def check_clouds(self, image: np.ndarray, threshold: float):
        """ """ Checks whether the input image has too many clouds, according to an illumination threshold and by checking
        the image histogram.

        Parameters
        ----------
        image: np.ndarray
            image to check for clouds
        threshold: float
            threshold on pixel illumination

        Returns
        -------
        above_threshold : bool
            True if the image has too many clouds according to the provided threshold. False otherwise.

        """ """
        print('Checking if image has too many clouds according to the threshold')
        # Normalize image
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Add up normalized pixels
        sum_pixels = np.sum(image_normalized)
        if sum_pixels > threshold:
            above_threshold = True
        else:
            above_threshold = False
        return above_threshold, sum_pixels
"""