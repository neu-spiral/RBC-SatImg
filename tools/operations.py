import numpy as np

from configuration import Config


def normalize(array: np.ndarray):
    """
    TODO: Document this function
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def get_index_pixels_of_interest(image_all_bands: np.ndarray, scene_id: int = 0):
    """ Returns the positional index of the pixels of interest according to the selected scene.

    Parameters
    ----------
    scene_id : int
        ID of the selected scene
    image_all_bands : np.ndarray
        pixel values for all the bands of the target image.
        This parameter must have size (dim_x*dim_y, n_bands):
            - dim_x = Config.image_dimensions[Config.scenario]['dim_x']
            - dim_y = Config.image_dimensions[Config.scenario]['dim_y']
            - n_bands = len(Config.bands_to_read))

    Returns
    -------
    index_pixels_of_interest : TODO: define correct type
        positional index of the pixels of interest

    """
    # Assign a number to each pixel
    # This number is used to select the pixels of interest
    row_index = np.array(range(image_all_bands.shape[0])).reshape(image_all_bands.shape[0], 1)
    image_band = np.concatenate((image_all_bands, row_index), axis=1)

    # Get index of all pixels
    index_all_pixels = image_band[:, -1].reshape(Config.image_dimensions[Config.scenario]['dim_x'],
                                                 Config.image_dimensions[Config.scenario]['dim_y'])

    # If the scene_id is 0, the whole image wants to be evaluated
    if scene_id == 0:
        index_pixels_of_interest = index_all_pixels[:, :].flatten().astype('int')
    else:
        # If the scene_id is other than 0, the coordinates of the pixels to evaluate are defined
        # in the configuration file.
        x_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['x_coords']
        y_coords = Config.pixel_coords_to_evaluate[Config.scene_id]['y_coords']
        index_pixels_of_interest = index_all_pixels[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]].flatten().astype(
            'int')
    return index_pixels_of_interest
