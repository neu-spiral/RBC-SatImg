import os
import logging

import numpy as np
import skimage

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
        # Check if the image dimensions are the proper ones
        if band.shape != (self.dim_x, self.dim_y):
            band = transform.resize(band, (self.dim_x, self.dim_y), anti_aliasing=True, preserve_range=True)
        return band

    def read_image(self, path: str, image_idx: int):
        """ Reads and returns all the bands for the image with index *image_idx*.

        Parameters
        ----------
        path : str
            path including the folders with the available bands
        image_idx : int
            index linked to the image to be read, if sorting by file_name in ascending order, and only
            considering the images of type and file extension specified in the configuration file.

        Returns
        -------
        image : np.ndarray
            all the read bands for the image with index *image_idx*

        """
        # Empty list
        image_all_bands = []

        # Loop through bands
        # Each band sub-folder must contain the image linked to image_idx
        # taken at the corresponding frequency band.
        for band_id in Config.bands_to_read:
            # Get path of image to be read in this iteration, which depends on the image index
            path_band_folder = os.path.join(path, f'Sentinel2_B{band_id}')
            path_image_band = get_path_image(path_folder=path_band_folder,
                                             image_type=Config.image_types[Config.scenario],
                                             file_extension='.tif', image_index=image_idx, band_id=band_id)

            # Read the corresponding band
            image_band = self.read_band(path_band=path_image_band)

            # Add the read band to the *image_all_bands* array
            image_all_bands.extend([image_band.flatten()])

        # Transpose for proper dimensions
        image_all_bands = np.transpose(image_all_bands)

        # Scale images
        image_all_bands = image_all_bands * Config.scaling_factor_sentinel
        return image_all_bands
