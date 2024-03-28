import os
from configuration import Config


def get_num_images_in_folder(path_folder: str, image_type: str, file_extension: str):
    """ Returns the number of images with type *image_type* and file extension *file_extension*
    in the folder with path *path_folder*.

    Parameters
    ----------
    path_folder : str
        path of the folder from which images are counted
    image_type : str
        type that images must have to be counted
    file_extension : str
        file extension that the image files must have to be counted

    Returns
    -------
    image_counter : int
        number of images with type *image_type* and file extension *file_extension*
        in the folder with path *path_folder*.

    """
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in os.listdir(path_folder):
        if file_name.endswith(image_type + file_extension):
            file_counter = file_counter + 1
    return file_counter


def get_path_image(path_folder: str, image_type: str, file_extension: str, image_index: int, band_id: str):
    """ Returns the path of the image stored in the folder *path_folder*, with type
    *image_type* and file extension *file_extension*. If sorting by file_name in ascending order,
    and only considering the images of the specified type and file extension, the returned
    path is linked to the image with index *image_index*.

    Parameters
    ----------
    path_folder: str
        path of the folder where the target image is stored, which depends on the band
    image_type: str
        type of the target image
    file_extension: str
        extension of the target image file
    image_index: int
        index of the target image within the folder with path *path_folder*

    Returns
    -------
    output_path : str
        path of the target image

    """
    date_string = ''
    output_path = -1  # if image with specified type, file extension and index is not found, -1 is returned
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in sorted(os.listdir(path_folder)):
        if file_name.endswith(image_type + file_extension):
            #print(file_name)
            #print(file_counter)
            if file_counter == image_index:
                date_image = get_date_from_file_name(file_name)
                print(f"Image with index {image_index} from date {date_image['year']}/{date_image['month']}/{date_image['day']} (band {band_id})")
                output_path = os.path.join(path_folder, file_name)
                date_string = f"{date_image['year']}-{date_image['month']}-{date_image['day']}"
                break  # if the corresponding image is found, the loop is automatically stopped
            file_counter = file_counter + 1
    return output_path, date_string


def get_date_from_file_name(file_name: str):
    """ Returns date information from file name.
    Parameters
    ----------
    file_name: str
        type of the target image

    Returns
    -------
    date : dict
        date information

    """
    if Config.scenario == 'multiearth':
        # File names have a different format for the multiearth dataset
        date_aux = file_name[-15:-5]
        date = {'year': date_aux[0:4], 'month': date_aux[5:7], 'day': date_aux[8:10]}
    else:
        date_aux = file_name.split("_")[2]
        date = {'year': date_aux[0:4], 'month': date_aux[4:6], 'day': date_aux[6:8]}
    return date
